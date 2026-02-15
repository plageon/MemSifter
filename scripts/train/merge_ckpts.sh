#!/bin/bash
# 
# Merge Multiple Checkpoint Models Script
# Average merge weights from multiple checkpoints
#
set -euo pipefail

# ============== Configuration Variables ==============
project_name=${PROJECT_NAME:-'MemSifter'}
model_name=${MODEL_NAME:-'MemSifter-Qwen3-4B-Task-Reward'}
model_dir=${MODEL_DIR:-"../models/${project_name}"}

# Checkpoint steps array, can be overridden by environment variable CKPT_STEPS
# Example: export CKPT_STEPS="20 30 40"
if [ -z "${CKPT_STEPS:-}" ]; then
  ckpt_steps=(20 30 40)
else
  # Convert environment variable string to array
  read -ra ckpt_steps <<< "${CKPT_STEPS}"
fi

# ============== Parameter Processing ==============
# Convert array to comma-separated string
ckpt_steps_str=""
for i in "${!ckpt_steps[@]}"; do
  if [ $i -eq 0 ]; then
    ckpt_steps_str="${ckpt_steps[$i]}"
  else
    ckpt_steps_str="${ckpt_steps_str},${ckpt_steps[$i]}"
  fi
done

echo "========================================================"
echo "Merge Checkpoint Models"
echo "========================================================"
echo "Project Name: ${project_name}"
echo "Model Name: ${model_name}"
echo "Model Directory: ${model_dir}"
echo "Checkpoint Steps: ${ckpt_steps_str}"
echo "========================================================"

# ============== Call Python Script ==============
python -m genrank_verl.merge_ckpts \
  --model_dir "${model_dir}" \
  --model_name "${model_name}" \
  --ckpt_steps "${ckpt_steps_str}"

echo "========================================================"
echo "Merge completed!"
echo "Output directory: ${model_dir}/${model_name}-merged"
echo "========================================================"
