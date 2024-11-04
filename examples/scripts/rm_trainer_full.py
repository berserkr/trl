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
          per_device_train_batch_size,
          learning_rate=2e-06):
        
    # https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoTokenizer.from_pretrained
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

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
        config=config,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map='auto'
    )

    print(model)

    ##############
    # Load dataset
    ##############
    train_dataset = load_dataset(dataset_name_or_path, split='train') #.to(local_rank)
    eval_dataset = load_dataset(dataset_name_or_path, split='validation') #.to(local_rank)

    print('#'*100)
    print(config)
    print('#'*100)

    training_arguments = RewardConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        bf16=True,
        save_strategy="steps",
        save_steps=1000,
        gradient_accumulation_steps=2,
        report_to="none",
        center_rewards_coefficient=0.01,
        # create checkpoint directories
        save_on_each_node=True,
        learning_rate=learning_rate
    )

    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    torch.cuda.empty_cache()

    trainer.save_model(training_arguments.output_dir)
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    trainer.save_model(training_arguments.output_dir)

if __name__ == "__main__":
    dataset='/proj/checkpoints/bathen/data/helpsteer2/rm_regular'
    #dataset='/proj/checkpoints/bathen/data/rm_mixtures/tahira_best'
    #dataset='/proj/checkpoints/bathen/data/helpsteer2/rm_regular_twoepochs'
    #dataset='/proj/checkpoints/bathen/data/rm_mixtures/tahira_best'
    #dataset='/proj/checkpoints/bathen/data/rm_mixtures/golden_only'

    base_model='/proj/checkpoints/bathen/models/base/granite-3.0-8b-instruct'
    #rm='/proj/checkpoints/bathen/models/reward/granite_3.0_8b_instruct_rm'
    rm='/proj/checkpoints/bathen/models/reward/granite_3.0_8b_instruct_rm_golden_lr2en6_full'
    #rm='/proj/checkpoints/bathen/models/reward/granite_3.0_8b_instruct_rm_helpsteer2_3epoch'
    
    train(
        base_model,
        dataset,
        rm,
        2,
        8)