#!/bin/bash

# deepsearch datasets
dataset_names=(locomo split_longmemeval_s personamem_128k personamem_1M zh4o personamem_v2_32k personamem_v2_128k split_longmemeval_m HotpotQA WebWalkerQA-Silver WebDancer)
dataset_splits=(test train)

for dataset_name in "${dataset_names[@]}"; do
  for dataset_split in "${dataset_splits[@]}"; do
    echo "Running session embedding for $dataset_name $dataset_split"

    python memretriever/ray_emb.py \
      --dataset_name $dataset_name \
      --dataset_split $dataset_split \
      --data_dir ../data \
      --output_dir ../data/results \
      --embed_store_path ../data/embedding_store \
      --embedding_model_name bge-m3 \
      --emb_concurrency 32 \
      --emb_batch_size 256 \
      --max_seq_len 512 \
      --batch_size 256 \
      --use_gpu
  done
done
