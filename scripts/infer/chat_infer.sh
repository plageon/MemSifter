#!/bin/bash
set -euo pipefail

RUNTIME_ENV=${RUNTIME_ENV:-"./configs/runtime_env.yaml"}

export API_KEY=${API_KEY:-"EMPTY"}
export BASE_URL=${BASE_URL:-"http://BASE_URL/v1"}

prompt_name=${PROMPT_NAME:-"gen_retrieve_instruct_prompt.yaml"}
embed_model=${EMBED_MODEL:-"bge-m3"}
output_dir=${OUTPUT_DIR:-"../data/results"}
model_path=${MODEL_PATH:-"../models/MemSifter/ep1-DAPO-Qwen3-4B-Task-Reward-step-80"}
model_name=${MODEL_NAME:-"MemSifter/ep1-DAPO-Qwen3-4B-Task-Reward-step-80"}
override_num=${OVERRIDE_NUM:-1}
max_concurrent=${MAX_CONCURRENT:-64}
max_output_token=${MAX_OUTPUT_TOKEN:-4096}
temperature=${TEMPERATURE:-0.6}
filter_col=${FILTER_COL:-"similarities"}
context_limit=${CONTEXT_LIMIT:-"128k"}


dataset_names=(locomo split_longmemeval_s personamem_128k personamem_1M zh4o personamem_v2_32k personamem_v2_128k split_longmemeval_m HotpotQA WebWalkerQA-Silver WebDancer)
dataset_splits=(test)


for dataset_split in "${dataset_splits[@]}"; do
  for dataset_name in "${dataset_names[@]}"; do
    echo "========================================================"
    echo "Running remote ranking inference for $dataset_name $dataset_split"
    echo "========================================================"

    python -B memretriever/ray_remote_vllm_infer.py \
      --prompt_name "${prompt_name}" \
      --dataset_name "${dataset_name}" \
      --dataset_split "${dataset_split}" \
      --embed_model "${embed_model}" \
      --output_dir "${output_dir}" \
      --override_num ${override_num} \
      --model_path "${model_path}" \
      --model_name "${model_name}" \
      --max_output_token ${max_output_token} \
      --temperature ${temperature} \
      --max_concurrent ${max_concurrent} \
      --filter_col "${filter_col}" \
      --context_limit "${context_limit}" \
      --infer_batch_size 1024
  done
  echo "Completed remote inference for $dataset_split"
done

echo "All remote vLLM ranking jobs submitted. Output base: ${output_dir}/${model_name}"
