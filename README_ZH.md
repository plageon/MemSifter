[English](README.md) | [中文](README_ZH.md)

# MemSifter: Offloading LLM Memory Retrieval via Outcome-Driven Proxy Reasoning

## 项目简介

MemSifter 是一个基于结果驱动的代理推理的 LLM 内存检索卸载系统。

## 脚本说明

本项目提供了完整的训练和推理流程脚本，位于 `scripts/` 目录下。

### 推理流程 (scripts/infer/)

推理流程分为三个阶段，按顺序执行：

#### 1. Session Embedding (`session_embedding.sh`)
使用 embedding 模型（如 bge-m3）计算 session embedding，进行初步过滤。

**功能：**
- 对多个数据集（locomo, split_longmemeval_s, personamem_128k 等）进行 session embedding 计算
- 生成 embedding 存储文件，用于后续的相似度检索
- 支持 train 和 test 数据集的批量处理

**使用方法：**
```bash
cd scripts/infer
./session_embedding.sh
```

**环境变量配置：**
- `EMBEDDING_MODEL_NAME`: embedding 模型名称（默认：bge-m3）
- `DATA_DIR`: 数据目录（默认：../data）
- `OUTPUT_DIR`: 输出目录（默认：../data/results）
- `EMBED_STORE_PATH`: embedding 存储路径（默认：../data/embedding_store）

#### 2. Session Ranking (`session_ranking`)
使用 generative ranking 模型计算精细的 session ranking。

**功能：**
- 基于 embedding 结果，使用生成式排序模型（如 MemSifter）对 session 进行精细排序
- 支持 Ray 分布式推理
- 生成 ranking 结果文件，包含每个 session 的排序分数

**使用方法：**
```bash
cd scripts/infer
./session_ranking
```

**环境变量配置：**
- `MODEL_NAME`: 模型名称（默认：MemSifter/ep1-DAPO-Qwen3-4B-Task-Reward-step-80）
- `RUNTIME_ENV`: Ray 运行时环境配置（默认：./configs/runtime_env.yaml）

#### 3. Chat Inference (`chat_infer.sh`)
调用 chat LLM 基于排序后的 session 做出最终回答。

**功能：**
- 使用排序后的 session 作为上下文
- 调用 LLM API 生成最终回答
- 支持多个数据集的批量推理

**使用方法：**
```bash
cd scripts/infer
./chat_infer.sh
```

**环境变量配置：**
- `MODEL_PATH`: 模型路径
- `MODEL_NAME`: 模型名称
- `API_KEY`: API 密钥
- `BASE_URL`: API 基础 URL
- `MAX_OUTPUT_TOKEN`: 最大输出 token 数（默认：4096）
- `TEMPERATURE`: 温度参数（默认：0.6）

### 训练流程 (scripts/train/)

#### 1. RL 训练数据准备 (`prepare_rl_data.sh`)
从 ranking/embedding 数据中筛选和准备强化学习训练数据。

**功能：**
- 根据 dataset recipe 配置文件加载数据集
- 支持 anchor 采样策略（基于 NDCG 分数）和随机采样策略
- 自动选择主要数据目录或备用数据目录
- 生成训练和测试数据文件

**使用方法：**
```bash
cd scripts/train
./prepare_rl_data.sh
```

**环境变量配置：**
- `RECIPE`: dataset recipe yaml 文件路径（默认：configs/dataset_recipe_deepsearch_v1.yaml）
- `PRIMARY_DATA_DIR`: 主要数据目录，使用 anchor 采样（默认：../data/results/DAPO-GenRank/ep3-DAPO-Qwen3-4B-Thinking-merged）
- `FALLBACK_DATA_DIR`: 备用数据目录，使用随机采样（默认：../data/results/bge-m3）
- `OUTPUT_DIR`: 输出目录（默认：../data）
- `VERSION`: 版本字符串（默认：自动生成 v{MMDD}-0）

**输出：**
- 训练数据：`{output_dir}/rl_train_data/{version}/train_{0..N}.parquet`
- 测试数据：`{output_dir}/rl_train_data/{version}/test.parquet`

#### 2. 主训练脚本 (`qwen3_4b_task_reward.sh`)
使用 DAPO 算法进行强化学习训练。

**功能：**
- 使用任务奖励模式训练（Marginal Utility Reward + Rank-Sensitive Reward）
- 支持多节点分布式训练
- 自动保存 checkpoint

**使用方法：**
```bash
cd scripts/train
./qwen3_4b_task_reward.sh
```

**环境变量配置：**
- `WORKING_DIR`: 工作目录
- `RUNTIME_ENV`: Ray 运行时环境配置
- `NNODES`: 节点数量（默认：1）
- `MODEL_PATH`: 基础模型路径
- `CKPTS_DIR`: checkpoint 保存目录
- `TRAIN_FILE`: 训练数据文件路径
- `TEST_FILE`: 测试数据文件路径

#### 3. 收集 VERL Checkpoint (`collect_verl_ckpt.sh`)
将 VERL 训练产生的 checkpoint 转换为标准模型格式。

**功能：**
- 将 VERL 的 FSDP checkpoint 转换为 HuggingFace 格式
- 支持批量处理多个 checkpoint 步骤

**使用方法：**
```bash
cd scripts/train
./collect_verl_ckpt.sh
```

**配置：**
- `project_name`: 项目名称（默认：MemSifter）
- `exp_name`: 实验名称（默认：ep1-MemSifter-Qwen3-4B-Task-Reward）
- `ckpt_steps`: checkpoint 步骤数组（默认：20 30 40）

#### 4. 合并 Checkpoint (`merge_ckpts.sh`)
将多个 checkpoint 的权重进行平均合并。

**功能：**
- 加载多个 checkpoint 模型
- 计算平均权重（使用 Float32 高精度计算）
- 保存合并后的模型

**使用方法：**
```bash
cd scripts/train
./merge_ckpts.sh
```

**环境变量配置：**
- `PROJECT_NAME`: 项目名称（默认：MemSifter）
- `MODEL_NAME`: 模型名称（默认：MemSifter-Qwen3-4B-Task-Reward）
- `MODEL_DIR`: 模型目录（默认：../models/MemSifter）
- `CKPT_STEPS`: checkpoint 步骤，空格分隔（默认：20 30 40）

**示例：**
```bash
export CKPT_STEPS="50 60 70"
export MODEL_NAME="MyModel-Name"
./merge_ckpts.sh
```

**输出：**
- 合并后的模型保存在：`{model_dir}/{model_name}-merged`

## 工作流程

### 完整训练流程

1. **数据准备**
   ```bash
   # 1. 计算 session embedding（初步过滤）
   cd scripts/infer
   ./session_embedding.sh
   
   # 2. 计算 session ranking（精细排序）
   ./session_ranking
   
   # 3. 准备 RL 训练数据
   cd ../train
   ./prepare_rl_data.sh
   ```

2. **模型训练**
   ```bash
   # 使用 DAPO 算法训练
   ./qwen3_4b_task_reward.sh
   ```

3. **模型后处理**
   ```bash
   # 收集 checkpoint
   ./collect_verl_ckpt.sh
   
   # 合并 checkpoint（可选）
   ./merge_ckpts.sh
   ```

### 完整推理流程

1. **Session Embedding** → 2. **Session Ranking** → 3. **Chat Inference**

按顺序执行 `scripts/infer/` 目录下的三个脚本即可。

## 环境配置

### Python 环境

本项目需要 Python 3.8 或更高版本。

#### 1. 安装依赖

```bash
pip install -r requirements.txt
```

#### 2. 启动 Ray 集群

在开始训练或推理之前，需要启动 Ray 集群。对于单机环境，可以使用以下命令启动 head 节点：

```bash
ray start --head
```

这将启动一个 Ray 集群的 head 节点。如果需要连接到远程 Ray 集群，可以使用：

```bash
ray start --address=<head-node-address>:<port>
```

停止 Ray 集群：

```bash
ray stop
```

#### 3. 环境变量配置

根据不同的脚本，可能需要设置以下环境变量：

- `API_KEY`: OpenAI API 密钥（用于 chat inference）
- `BASE_URL`: API 基础 URL
- `CUDA_VISIBLE_DEVICES`: 指定使用的 GPU 设备
- `RAY_ADDRESS`: Ray 集群地址（如果使用远程集群）

### 依赖包

主要依赖包括：
- Python 3.8+
- PyTorch
- Ray (包含 data, train, tune, serve, llm 扩展)
- Transformers
- 其他依赖见 `requirements.txt`

## 配置说明

主要配置文件位于 `configs/` 目录：
- `runtime_env.yaml`: Ray 运行时环境配置
- `dataset_recipe_*.yaml`: 数据集配方配置
- `*_prompt.yaml`: Prompt 模板配置

## 许可证

[添加许可证信息]