#!/bin/bash
project_name='MemSifter'
exp_name='ep1-MemSifter-Qwen3-4B-Task-Reward'
ckpt_steps=(20 30 40)

for ckpt_step in "${ckpt_steps[@]}"; do
  python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir ../ckpt/${project_name}/${exp_name}/global_step_${ckpt_step}/actor \
    --target_dir ../models/${project_name}/${exp_name}-step-${ckpt_step}
done
