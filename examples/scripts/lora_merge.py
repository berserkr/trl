import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from peft import PeftModel
import argparse

#model_name_or_path = "/proj/checkpoints/bathen/models/base/mistral7b_inst_v3"
model_name_or_path = "/proj/checkpoints/bathen/models/base/granite-3.0-8b-instruct"

#peft_id = "data/outputs/granite-3b-instruct-preview-16k-100krt/checkpoint-3602"
#peft_id = '/proj/checkpoints/bathen/models/reward/granite_3.0_8b_instruct_rm/checkpoint-4006'
#peft_id = '/proj/checkpoints/bathen/models/reward/granite_3.0_8b_instruct_rm_tm'
#peft_id = '/proj/checkpoints/bathen/models/reward/granite_3.0_8b_instruct_rm_golden_only'
peft_id = '/proj/checkpoints/bathen/models/reward/granite_3.0_8b_instruct_rm_helpsteer2_3epoch'

#peft_id = "/proj/checkpoints/bathen/models/reward/granite_3.0_8b_instruct_rm"
#merged_model_path = "data/outputs/granite-3b-instruct-preview-16k-100krt/merged/" 
#merged_model_path = "/proj/checkpoints/bathen/models/reward/granite_3.0_8b_instruct_rm/merged_rhlf/"
merged_model_path = f"{peft_id}/merged"


def convert(model_name_or_path, peft_id, merged_model_path=None):
    if merged_model_path is None:
        merged_model_path = f'{peft_id}/merged'
    
    print(f'Saving merged model to {merged_model_path}')

    torch_dtype = torch.bfloat16
    # place the model on GPU
    device_map = {"": "cuda"}

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    config = AutoConfig.from_pretrained(
        model_name_or_path, 
        torchscript=True, 
        trust_remote_code=True, 
        num_labels=1, 
    )

    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        torch_dtype="auto",
        config=config,
    )

    model = PeftModel.from_pretrained(
        base_model, 
        peft_id,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )

    # NOTE: merge LoRA weights
    merged_model = model.merge_and_unload().eval()

    #p print save the model and tokenizer...
    merged_model.save_pretrained(merged_model_path)
    tokenizer.save_pretrained(merged_model_path)


"""
Test logic
model = AutoModelForSequenceClassification.from_pretrained(merged_model_path,trust_remote_code=True,num_labels=1)
tokenizer = AutoTokenizer.from_pretrained(merged_model_path, trust_remote_code=True)
ids=tokenizer.apply_chat_template(data, return_tensors="pt")
with torch.no_grad():
    result=model(ids)
result[0]
>>> tensor([[-0.2455]])
"""

if __name__ =='__main__':
    parser = argparse.ArgumentParser(
        description="Simple lora adapter merger."
    )

    parser.add_argument(
        "--model_name_or_path", required=True,
        help="Full or relative path to base model."
    )

    parser.add_argument(
        "--peft_id", required=True,
        help="Full or relative path to the lora adapters."
    )

    parser.add_argument(
        "--merged_model_path", required=False, default=None,
        help="Path for merged model. Default is peft_id/merged"
    )

    args = parser.parse_args()

    convert(
            model_name_or_path=args.model_name_or_path, 
            peft_id=args.peft_id, 
            merged_model_path=args.merged_model_path)
