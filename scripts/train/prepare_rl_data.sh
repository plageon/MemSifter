#!/bin/bash
# 
# RL Training Data Preparation Script
# Filter and prepare reinforcement learning training data from ranking/embedding data
# Supports anchor sampling strategy (based on NDCG score) and random sampling strategy
#
set -euo pipefail

# ============== Configuration Variables ==============
recipe=${RECIPE:-"configs/dataset_recipe_deepsearch_v1.yaml"}
primary_data_dir=${PRIMARY_DATA_DIR:-"../data/results/DAPO-GenRank/ep3-DAPO-Qwen3-4B-Thinking-merged"}
fallback_data_dir=${FALLBACK_DATA_DIR:-"../data/results/bge-m3"}
output_dir=${OUTPUT_DIR:-"../data"}
version=${VERSION:-""}  # If empty, will auto-generate v{MMDD}-0

echo "========================================================"
echo "RL Training Data Preparation"
echo "========================================================"
echo "Dataset Recipe: ${recipe}"
echo "Primary Data Directory: ${primary_data_dir}"
echo "Fallback Data Directory: ${fallback_data_dir}"
echo "Output Directory: ${output_dir}"
if [ -n "${version}" ]; then
  echo "Version: ${version}"
else
  echo "Version: Auto-generate (v{MMDD}-0)"
fi
echo "========================================================"

# ============== Call Python Script ==============
if [ -n "${version}" ]; then
  python -m data.rl_train_data_propare \
    --recipe "${recipe}" \
    --primary_data_dir "${primary_data_dir}" \
    --fallback_data_dir "${fallback_data_dir}" \
    --output_dir "${output_dir}" \
    --version "${version}"
else
  python -m data.rl_train_data_propare \
    --recipe "${recipe}" \
    --primary_data_dir "${primary_data_dir}" \
    --fallback_data_dir "${fallback_data_dir}" \
    --output_dir "${output_dir}"
fi

echo "========================================================"
echo "Data preparation completed!"
echo "Output directory: ${output_dir}/rl_train_data/"
echo "========================================================"
