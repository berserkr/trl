from datasets import load_dataset
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
from tqdm import tqdm
import argparse

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
          per_device_train_batch_size,
          learning_rate=2e-06):
        
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
        device_map='auto'
    )
    model = prepare_model_for_kbit_training(model)

    print(model)

    ##############
    # Load lora 
    ##############
    lora_alpha = 32
    lora_dropout = 0.1
    lora_r = 8

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
    try: # assume test and val in the splits...
        train_dataset = load_dataset(dataset_name_or_path, split='train') #.to(local_rank)
        eval_dataset = load_dataset(dataset_name_or_path, split='validation') #.to(local_rank)
    except ValueError:
        dataset = load_dataset(dataset_name_or_path, split='train')
        train_val_split = dataset.train_test_split(test_size=0.1)
        train_dataset = train_val_split['train']
        eval_dataset = train_val_split['test']
 
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
        peft_config=peft_config,
    )
    trainer.train()
    torch.cuda.empty_cache()

    trainer.save_model(training_arguments.output_dir)
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    trainer.save_model(training_arguments.output_dir)


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_path',
        action="store", dest="data_path",
        required=True,
        help="path to dataset folder")

parser.add_argument('-m', '--base_model',
        action="store", dest="base_model",
        required=True,
        help="path to base model")

parser.add_argument('-r', '--reward_model',
        action="store", dest="reward_model",
        required=True,
        help="path to reward model")

parser.add_argument('-e', '--epochs',
        action="store", dest="epochs",
        required=False,
        default=2,
        help="Number of epochs")

parser.add_argument('-b', '--batch_size',
         action="store", dest="batch_size",
         default=8,
         required=False,
         help="Batch size for training")  

if __name__ == "__main__":
    args = parser.parse_args()
    train(
        args.base_model,
        args.data_path,
        args.reward_model,
        args.epochs,
        args.batch_size)
