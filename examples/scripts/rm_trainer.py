from datasets import load_dataset
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
from tqdm import tqdm

# https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoConfig
# https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForCausalLM
# https://huggingface.co/docs/transformers/main_classes/quantization#transformers.BitsAndBytesConfig
# https://huggingface.co/docs/transformers/internal/generation_utils#transformers.StoppingCriteria
# https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# Transformer Reinforcement Learning
# https://huggingface.co/docs/trl/index
from trl import RewardTrainer, RewardConfig
import torch

# load tokenizer..

def train(model_name_or_path,
          dataset_name_or_path,
          output_dir,
          num_epochs,
          per_device_train_batch_size):
        
    # https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoTokenizer.from_pretrained
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    ##############
    # Load quant
    ##############
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,  # if not set will throw a warning about slow speeds when training
    )

    ##############
    # Load model
    ##############
    config = AutoConfig.from_pretrained(
        model_name_or_path, 
        torchscript=True, 
        trust_remote_code=True, 
        num_labels=1, 
    )

    # https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoModelForCausalLM.from_pretrained
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        torch_dtype="auto",
        quantization_config=bnb_config,
        config=config,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model = prepare_model_for_kbit_training(model)

    print(model)

    ##############
    # Load lora 
    ##############
    lora_alpha = 32
    lora_dropout = 0.1
    lora_r = 4

    """
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    """
    
    target_modules = 'all-linear'

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        bias="none",
        lora_dropout=lora_dropout,  # Conventional
        task_type=TaskType.SEQ_CLS,
    )

    ##############
    # Load dataset
    ##############
    train_dataset = load_dataset(dataset_name_or_path, split='train') #.to(local_rank)
    eval_dataset = load_dataset(dataset_name_or_path, split='validation') #.to(local_rank)

    print('#'*100)
    print(config)
    print('#'*100)
    print(peft_config)
    print('#'*100)

    training_arguments = RewardConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        bf16=True,
        save_strategy="epoch",
        report_to="none",
        # create checkpoint directories
        save_on_each_node=True,
    )
    training_arguments.center_rewards_coefficient=0.01

    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    trainer.train()
    torch.cuda.empty_cache()

    trainer.save_model(training_arguments.output_dir)
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    trainer.save_model(training_arguments.output_dir)

if __name__ == "__main__":
    train(
        '/proj/checkpoints/bathen/models/base/granite-3.0-8b-instruct',
        '/proj/checkpoints/bathen/data/helpsteer2/rm_regular',
        '/proj/checkpoints/bathen/models/reward/granite_3.0_8b_instruct_rm',
        2,
        8)