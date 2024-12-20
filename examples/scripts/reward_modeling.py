# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Full training:
python examples/scripts/reward_modeling.py \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --output_dir Qwen2-0.5B-Reward \
    --per_device_train_batch_size 8 \
    --num_train_epochs 1 \
    --gradient_checkpointing True \
    --learning_rate 1.0e-5 \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 50 \
    --max_length 2048

LoRA:
python examples/scripts/reward_modeling.py \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --output_dir Qwen2-0.5B-Reward-LoRA \
    --per_device_train_batch_size 8 \
    --num_train_epochs 1 \
    --gradient_checkpointing True \
    --learning_rate 1.0e-4 \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 50 \
    --max_length 2048 \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16
"""

import warnings

import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser, BitsAndBytesConfig
import torch.distributed as dist
import os

from trl import (
    ModelConfig,
    RewardConfig,
    RewardTrainer,
    get_kbit_device_map,
    get_quantization_config,
    setup_chat_format,
)
from trl.commands.cli_utils import RewardScriptArguments
from deepspeed.utils import set_z3_leaf_modules
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
from peft import LoraConfig, TaskType

def setup(rank, world_size):
    # Initialize the process group

    if rank == 0:
        dist.init_process_group(backend="gloo|nccl", rank=rank, world_size=world_size)
    
    torch.cuda.set_device(local_rank)

def cleanup(rank):

    if rank == 0:
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = HfArgumentParser((RewardScriptArguments, RewardConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_into_dataclasses()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.center_rewards_coefficient=0.01
    training_args.bf16=True

    # init distributed...

    world_size = torch.cuda.device_count()
    local_rank = int(os.environ["LOCAL_RANK"])

    setup(os.environ["RANK"], world_size)
    dist.barrier()

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )

    quant_storage_dtype = torch.bfloat16
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_quant_storage=quant_storage_dtype,
    )

    model_kwargs = dict(
        revision=model_config.model_revision,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        use_cache=False if training_args.gradient_checkpointing else True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, use_fast=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name_or_path, 
        num_labels=1, 
        trust_remote_code=model_config.trust_remote_code, 
        **model_kwargs
    )
    # Align padding tokens between tokenizer and model
    model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.pad_token = '<pad>'
    #tokenizer.pas_token = tokenizer.eos_token
    
    #set_z3_leaf_modules(model, [MixtralSparseMoeBlock])

    model = model.to(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank)

    # If post-training a base model, use ChatML as the default template
    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)
        
    """
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
    """
    peft_config = LoraConfig(
        r=32,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        bias="none",
        lora_dropout=0.05,  # Conventional
        task_type=TaskType.SEQ_CLS,
    )

    ##############
    # Load dataset
    ##############
    train_dataset = load_dataset(script_args.dataset_name, split='train') #.to(local_rank)
    eval_dataset = load_dataset(script_args.dataset_name, split='validation') #.to(local_rank)

    ##########
    # Training
    ##########
    print(model_config)
    
    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    trainer.train()
    torch.cuda.empty_cache()

    # fsdp
    # saving final model
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    ############################
    # Save model and push to Hub
    ############################
    trainer.save_model(training_args.output_dir)
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    """
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
    """

    cleanup(os.environ["RANK"])
