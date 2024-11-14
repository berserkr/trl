cd /dccstor/distillation/code/trl/examples/scripts

GPUs=8+1
MEM=32g
#EXPERIMENT_NAME=anthropic_hh-rm
EXPERIMENT_NAME=Skywork-Reward-Preference-Granite8b-RM-3epochs-lora-cosine
LOG=/dccstor/distillation/logs/${EXPERIMENT_NAME}.out
#Q=x86_24h
Q=nonstandard
#Q=x86_6h

port=$(shuf -i25000-30000 -n1)
jbsub -cores $GPUs -mem $MEM -q $Q -require a100_80gb -name $EXPERIMENT_NAME \
    -out "$LOG" \
    ./ccc_trainer_lora.sh

