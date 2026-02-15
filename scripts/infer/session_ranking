#!/bin/bash

RUNTIME_ENV=${RUNTIME_ENV:-"./configs/runtime_env.yaml"}

dataset_names=(locomo split_longmemeval_s personamem_128k personamem_1M zh4o personamem_v2_32k personamem_v2_128k split_longmemeval_m HotpotQA WebWalkerQA-Silver WebDancer)
dataset_splits=(test)

model_name="MemSifter/ep1-DAPO-Qwen3-4B-Task-Reward-step-80"

max_num_batched_tokens=$((130*1024))
max_model_len=$((146*1024))
max_output_token=$((16*1024))

for dataset_name in "${dataset_names[@]}"; do
  for dataset_split in "${dataset_splits[@]}"; do
    echo "========================================================"
    echo "Running session ranking for $dataset_name $dataset_split"
    echo "========================================================"

    ray job submit \
    --working-dir . \
    --runtime-env="${RUNTIME_ENV}" \
    -- \
    python -B memretriever/ray_vllm_infer.py \
    --prompt_name "gen_retrieve_instruct_prompt.yaml" \
    --dataset_name $dataset_name \
    --dataset_split $dataset_split \
    --embed_model bge-m3 \
    --output_dir "../data/results" \
    --model_path "../models/${model_name}" \
    --model_name $model_name \
    --concurrency 8 \
    --max_num_batched_tokens $max_num_batched_tokens \
    --max_model_len $max_model_len \
    --tp 1 \
    --pp 1 \
    --batch_size 1 \
    --max_output_token $max_output_token \
    --temperature 0.6 \
    --override_num 1
  done
done