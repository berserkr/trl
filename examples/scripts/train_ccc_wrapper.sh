cd /dccstor/distillation/code/trl/examples/scripts

GPUs=4+4
MEM=32g
EXPERIMENT_NAME=Skywork-Reward-Preference-Granite8b-RM-test-shell
LOG=/dccstor/distillation/logs/${EXPERIMENT_NAME}.out
Q=x86_24h

port=$(shuf -i25000-30000 -n1)
jbsub -cores $GPUs -mem $MEM -q $Q -require a100_80gb -name $EXPERIMENT_NAME \
    -out "$LOG" \
    ./ccc_trainer.sh

