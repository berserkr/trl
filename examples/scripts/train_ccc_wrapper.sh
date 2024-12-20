cd /dccstor/distillation/code/trl/examples/scripts

GPUs=4+8
MEM=32g
EXPERIMENT_NAME=Skywork-Reward-Preference-Granite8b-RM-full
LOG=/dccstor/distillation/logs/${EXPERIMENT_NAME}.out
Q=nonstandard

port=$(shuf -i25000-30000 -n1)
jbsub -cores $GPUs -mem $MEM -q $Q -require a100_80gb -name $EXPERIMENT_NAME \
    -out "$LOG" \
    ./ccc_trainer.sh

