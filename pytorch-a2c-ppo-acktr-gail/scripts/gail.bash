#!/bin/bash

eval_eps=50
eval_interval=100000
num_env_steps=4000000
expert_file="../gail_experts/expert-data/${1}/reset/800_steps-90_sp_point5_play_open_200_extra_lasts/int_2.gz"
env=$1
num_processes=1
#seed=$2

seeds=(1 2 3 4 5)
# seeds=(3 4 5)
# seeds=(10 11)
job_type="$1"

for seed in "${seeds[@]}"
do
    
python ../main.py \
  --seed "$seed" \
  --num-steps 2048 \
  --lr 3e-4 \
  --entropy-coef 0 \
  --value-loss-coef 0.5 \
  --ppo-epoch 10 \
  --num-mini-batch 32 \
  --gamma 0.99 \
  --gae-lambda 0.95 \
  --use-linear-lr-decay \
  --use-proper-time-limits \
  --num-processes="$num_processes" \
  --use-gae \
  --algo ppo \
  --gail \
  --eval-interval="$eval_interval" \
  --num-env-steps="$num_env_steps" \
  --gail-experts-file="$expert_file" \
  --env-name="$env" \
  --log-interval 10 \
  --eval-eps "$eval_eps" \
  --no-cuda &
#  --train-render


done
