#!/bin/bash

startTime=$(date +%s) #mark the start of job 
hostname=`hostname`
verbose=0
runtime=$(date "+%Y.%m.%d-%H.%M")

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TOKENIZERS_PARALLELISM=true
export CUDA_HOME="/opt/share/cuda-12.1"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export WANDB_MODE=disabled
export NVIDIA_PYTORCH_VERSION=3.10
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/share/cudnn-linux-x86_64-8.9.7.29_cuda12/lib
export PYTHONPATH=$PYTHONPATH:/proj/checkpoints/shared_data/envs/repos/NeMo:/proj/checkpoints/shared_data/envs/repos/NeMo-Aligner:/proj/checkpoints/shared_data/envs/repos/Megatron-LM:/proj/checkpoints/shared_data/envs/repos/apex

MASTER_ADDR=$(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | head -n 1)
MASTER_PORT=28444 #5${LSB_JOBID: -5:-1}
NNODES=$(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | sed 'n; d' | wc -w)
#NNODES=2
GPUS_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -w)
NODE_RANK=$(($(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | sed 'n; d' | grep -n -m1 $HOSTNAME | cut -d':' -f1)-1))
JOB_ID=${LSB_JOBID}

MODEL=mistral7b_inst_v3
MODEL_PATH_IN=/proj/checkpoints/bathen/models/base/${MODEL}
DATA=/proj/checkpoints/tahira/data/rm-data-msgs/gold/anthropic_hh
RUN_DIR=/proj/checkpoints/bathen/developer/trl
MODEL_PATH_OUT=/proj/checkpoints/bathen/models/reward/${MODEL}_rm

source /proj/checkpoints/bathen/envs/run.env

. /u/bathen/miniconda3/etc/profile.d/conda.sh
conda_env_path="/proj/checkpoints/bathen/envs/conda/rewarding"
echo "conda activate ${conda_env_path}"
conda activate ${conda_env_path}

vf_coef=1.0

exp_name=${MODEL}_RM

outerr_file=${exp_name}".out"
GPUS_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -w)

cd $RUN_DIR
mkdir -p $exp_name

echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "NODE_RANK=${NODE_RANK}"
echo "JOB_ID=${JOB_ID}"

torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes  $NNODES\
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    examples/scripts/reward_modeling.py \
    --model_name_or_path $MODEL_PATH_IN \
    --dataset_name $DATA \
    --output_dir $MODEL_PATH_OUT \
    --per_device_train_batch_size 8 \
    --num_train_epochs 1 \
    --learning_rate 1.0e-4 \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 50 \
    --max_length 1024 \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16 