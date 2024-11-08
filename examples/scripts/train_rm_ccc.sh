export HF_DATASETS_CACHE="/dccstor/distillation/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="/dccstor/distillation/.cache/huggingface/transformers"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

export TOKENIZERS_PARALLELISM=true

export CUDA_HOME="/opt/share/cuda-12.2"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export WANDB_MODE=disabled

# conda stuff
ARCH=$(uname -m)
if [ -f "/opt/share/anaconda3-2019.03/$ARCH/etc/profile.d/conda.sh" ]; then
    . "/opt/share/anaconda3-2019.03/$ARCH/etc/profile.d/conda.sh"
else
     export PATH="/opt/share/anaconda3-2019.03/$ARCH/bin:$PATH"
fi

data_path="/dccstor/distillation/data/rm/synthetic/Skywork-Reward-Preference-80K-v0.2"
base_model="/dccstor/distillation/models/base/granite-3.0-8b-instruct"
reward_model="/dccstor/distillation/models/rm/granite-3.0-8b-instruct-Skywork-Reward-Preference-80K-v0.2"

cd /dccstor/distillation/code/trl/examples/scripts
LOG=/dccstor/distillation/logs/Skywork-Reward-Preference-Granite8b-RM.out

GPUs=4+1
MEM=100g
EXPERIMENT_NAME=Skywork-Reward-Preference-Granite8b-RM-test
Q=nonstandard

port=$(shuf -i25000-30000 -n1)
jbsub -cores $GPUs -mem 100g -q $Q -require a100_80gb -name $EXPERIMENT_NAME \
    -out "$LOG" \
    python rm_trainer_full.py --data_path ${data_path} --base_model ${base_model} --reward_model ${reward_model}

