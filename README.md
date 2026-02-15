# MemSifter: Offloading LLM Memory Retrieval via Outcome-Driven Proxy Reasoning

[English](README.md) | [中文](README_ZH.md)

## Project Introduction

MemSifter is an LLM memory retrieval offloading system based on outcome-driven proxy reasoning.

## Scripts Overview

This project provides complete training and inference workflow scripts located in the `scripts/` directory.

### Inference Workflow (scripts/infer/)

The inference workflow consists of three stages, executed in order:

#### 1. Session Embedding (`session_embedding.sh`)
Computes session embeddings using an embedding model (e.g., bge-m3) for initial filtering.

**Features:**
- Computes session embeddings for multiple datasets (locomo, split_longmemeval_s, personamem_128k, etc.)
- Generates embedding storage files for subsequent similarity retrieval
- Supports batch processing of train and test datasets

**Usage:**
```bash
cd scripts/infer
./session_embedding.sh
```

**Environment Variables:**
- `EMBEDDING_MODEL_NAME`: Embedding model name (default: bge-m3)
- `DATA_DIR`: Data directory (default: ../data)
- `OUTPUT_DIR`: Output directory (default: ../data/results)
- `EMBED_STORE_PATH`: Embedding storage path (default: ../data/embedding_store)

#### 2. Session Ranking (`session_ranking`)
Computes fine-grained session ranking using a generative ranking model.

**Features:**
- Uses generative ranking model (e.g., MemSifter) to perform fine-grained ranking of sessions based on embedding results
- Supports Ray distributed inference
- Generates ranking result files containing ranking scores for each session

**Usage:**
```bash
cd scripts/infer
./session_ranking
```

**Environment Variables:**
- `MODEL_NAME`: Model name (default: MemSifter/ep1-DAPO-Qwen3-4B-Task-Reward-step-80)
- `RUNTIME_ENV`: Ray runtime environment configuration (default: ./configs/runtime_env.yaml)

#### 3. Chat Inference (`chat_infer.sh`)
Calls chat LLM to generate final answers based on ranked sessions.

**Features:**
- Uses ranked sessions as context
- Calls LLM API to generate final answers
- Supports batch inference for multiple datasets

**Usage:**
```bash
cd scripts/infer
./chat_infer.sh
```

**Environment Variables:**
- `MODEL_PATH`: Model path
- `MODEL_NAME`: Model name
- `API_KEY`: API key
- `BASE_URL`: API base URL
- `MAX_OUTPUT_TOKEN`: Maximum output tokens (default: 4096)
- `TEMPERATURE`: Temperature parameter (default: 0.6)

### Training Workflow (scripts/train/)

#### 1. RL Training Data Preparation (`prepare_rl_data.sh`)
Filters and prepares reinforcement learning training data from ranking/embedding data.

**Features:**
- Loads datasets according to dataset recipe configuration file
- Supports anchor sampling strategy (based on NDCG score) and random sampling strategy
- Automatically selects primary data directory or fallback data directory
- Generates training and test data files

**Usage:**
```bash
cd scripts/train
./prepare_rl_data.sh
```

**Environment Variables:**
- `RECIPE`: Dataset recipe yaml file path (default: configs/dataset_recipe_deepsearch_v1.yaml)
- `PRIMARY_DATA_DIR`: Primary data directory, uses anchor sampling (default: ../data/results/DAPO-GenRank/ep3-DAPO-Qwen3-4B-Thinking-merged)
- `FALLBACK_DATA_DIR`: Fallback data directory, uses random sampling (default: ../data/results/bge-m3)
- `OUTPUT_DIR`: Output directory (default: ../data)
- `VERSION`: Version string (default: auto-generate v{MMDD}-0)

**Output:**
- Training data: `{output_dir}/rl_train_data/{version}/train_{0..N}.parquet`
- Test data: `{output_dir}/rl_train_data/{version}/test.parquet`

#### 2. Main Training Script (`qwen3_4b_task_reward.sh`)
Performs reinforcement learning training using DAPO algorithm.

**Features:**
- Trains using task reward mode (Marginal Utility Reward + Rank-Sensitive Reward)
- Supports multi-node distributed training
- Automatically saves checkpoints

**Usage:**
```bash
cd scripts/train
./qwen3_4b_task_reward.sh
```

**Environment Variables:**
- `WORKING_DIR`: Working directory
- `RUNTIME_ENV`: Ray runtime environment configuration
- `NNODES`: Number of nodes (default: 1)
- `MODEL_PATH`: Base model path
- `CKPTS_DIR`: Checkpoint save directory
- `TRAIN_FILE`: Training data file path
- `TEST_FILE`: Test data file path

#### 3. Collect VERL Checkpoint (`collect_verl_ckpt.sh`)
Converts VERL training checkpoints to standard model format.

**Features:**
- Converts VERL's FSDP checkpoint to HuggingFace format
- Supports batch processing of multiple checkpoint steps

**Usage:**
```bash
cd scripts/train
./collect_verl_ckpt.sh
```

**Configuration:**
- `project_name`: Project name (default: MemSifter)
- `exp_name`: Experiment name (default: ep1-MemSifter-Qwen3-4B-Task-Reward)
- `ckpt_steps`: Checkpoint steps array (default: 20 30 40)

#### 4. Merge Checkpoints (`merge_ckpts.sh`)
Averages and merges weights from multiple checkpoints.

**Features:**
- Loads multiple checkpoint models
- Computes average weights (using Float32 high precision)
- Saves merged model

**Usage:**
```bash
cd scripts/train
./merge_ckpts.sh
```

**Environment Variables:**
- `PROJECT_NAME`: Project name (default: MemSifter)
- `MODEL_NAME`: Model name (default: MemSifter-Qwen3-4B-Task-Reward)
- `MODEL_DIR`: Model directory (default: ../models/MemSifter)
- `CKPT_STEPS`: Checkpoint steps, space-separated (default: 20 30 40)

**Example:**
```bash
export CKPT_STEPS="50 60 70"
export MODEL_NAME="MyModel-Name"
./merge_ckpts.sh
```

**Output:**
- Merged model saved at: `{model_dir}/{model_name}-merged`

## Workflow

### Complete Training Workflow

1. **Data Preparation**
   ```bash
   # 1. Compute session embedding (initial filtering)
   cd scripts/infer
   ./session_embedding.sh
   
   # 2. Compute session ranking (fine-grained ranking)
   ./session_ranking
   
   # 3. Prepare RL training data
   cd ../train
   ./prepare_rl_data.sh
   ```

2. **Model Training**
   ```bash
   # Train using DAPO algorithm
   ./qwen3_4b_task_reward.sh
   ```

3. **Model Post-processing**
   ```bash
   # Collect checkpoints
   ./collect_verl_ckpt.sh
   
   # Merge checkpoints (optional)
   ./merge_ckpts.sh
   ```

### Complete Inference Workflow

1. **Session Embedding** → 2. **Session Ranking** → 3. **Chat Inference**

Execute the three scripts in the `scripts/infer/` directory in order.

## Environment Configuration

### Python Environment

This project requires Python 3.8 or higher.

#### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 2. Start Ray Cluster

Before starting training or inference, you need to start a Ray cluster. For a single-machine environment, you can start a head node using:

```bash
ray start --head
```

This will start a head node of a Ray cluster. If you need to connect to a remote Ray cluster, you can use:

```bash
ray start --address=<head-node-address>:<port>
```

To stop the Ray cluster:

```bash
ray stop
```

#### 3. Environment Variable Configuration

Depending on the script, you may need to set the following environment variables:

- `API_KEY`: OpenAI API key (for chat inference)
- `BASE_URL`: API base URL
- `CUDA_VISIBLE_DEVICES`: Specify GPU devices to use
- `RAY_ADDRESS`: Ray cluster address (if using remote cluster)

### Dependencies

Main dependencies include:
- Python 3.8+
- PyTorch
- Ray (with data, train, tune, serve, llm extensions)
- Transformers
- Other dependencies see `requirements.txt`

## Configuration

Main configuration files are located in the `configs/` directory:
- `runtime_env.yaml`: Ray runtime environment configuration
- `dataset_recipe_*.yaml`: Dataset recipe configuration
- `*_prompt.yaml`: Prompt template configuration

## License

[Add license information]
