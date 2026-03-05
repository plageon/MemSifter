# MemSifter: 基于结果驱动代理推理的 LLM 记忆检索卸载系统

<div align="center">
<a href="https://arxiv.org/abs/2603.03379" target="_blank"><img src=https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv></a>
<a href="https://huggingface.co/zstanjj/MemSifter-4B-Thinking" target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace%20Models-27b3b4.svg></a>
<a href="https://www.modelscope.cn/models/zstanjj/MemSifter-4B-Thinking" target="_blank"><img src=https://custom-icon-badges.demolab.com/badge/ModelScope%20Models-624aff?style=flat&logo=modelscope&logoColor=white></a>
<a href="https://github.com/plageon/MemSifter/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green"></a>
<a><img alt="Static Badge" src="https://img.shields.io/badge/made_with-Python-blue"></a>
</div>

[English](README.md) | [中文](README_ZH.md)

## 📖 目录

- [项目简介](#-项目简介)
- [最新动态](#-最新动态)
- [安装](#-安装)
- [快速入门](#-快速入门)
- [复现实验结果](#-复现实验结果)
- [模型训练](#-模型训练)
- [引用](#-引用)

## ✨ 项目简介

**MemSifter** 是一个基于结果驱动代理推理的 LLM 记忆检索卸载系统。给定大量个人对话会话（"大海捞针"场景），MemSifter 能够高效识别与用户查询最相关的会话，并将其作为上下文传递给下游对话 LLM——无需由 LLM 本身承担检索工作。

系统遵循三阶段流水线：

```
会话嵌入 (Session Embedding)  →  会话排序 (Session Ranking)  →  对话 LLM
        (bge-m3)                     (生成式重排序器)              (任意 LLM)
```

1. **会话嵌入** — 使用稠密嵌入模型（bge-m3）对所有会话进行粗粒度相似度预筛选。
2. **会话排序** — MemSifter 是一个使用 DAPO 强化学习训练的轻量级生成式模型，对预筛选候选项进行细粒度重排序。
3. **对话 LLM** — 将排名靠前的会话组装成上下文窗口，传递给任意兼容 OpenAI 接口的对话模型，以生成最终答案。

## 🗞 最新动态

- **[2026/03/03]** 论文已发布于 [arXiv](https://arxiv.org/abs/2603.03379)。
- **[2026/02/20]** 代码、模型与数据正式发布。

## 🔧 安装

**环境要求：** Python 3.8+，两块支持 CUDA 的 GPU（用于快速入门的单样本推理）。

```bash
git clone https://github.com/plageon/MemSifter.git
cd MemSifter
pip install -r requirements.txt
```

将所需模型下载到本地 `models/` 目录：

| 模型 | 用途 | 来源 |
|---|---|---|
| `bge-m3` | 会话嵌入 | [HuggingFace](https://huggingface.co/BAAI/bge-m3) |
| `zstanjj/MemSifter-4B-Thinking` | 生成式会话排序器 | [HuggingFace](https://huggingface.co/zstanjj/MemSifter-4B-Thinking) |

```bash
# 使用 huggingface-cli 下载示例
huggingface-cli download BAAI/bge-m3 --local-dir models/bge-m3
huggingface-cli download zstanjj/MemSifter-4B-Thinking \
    --local-dir models/zstanjj/MemSifter-4B-Thinking
```

如仅运行快速入门的单样本推理（无需 Ray），可只安装以下最小依赖：

```bash
pip install torch sentence-transformers vllm openai pyyaml loguru numpy pandas
```

## 🚀 快速入门

`memsifter/toolkit.py` 中的工具包提供了三个类，用于**无需 Ray 依赖**的单样本推理。假设你有两块 GPU：嵌入模型运行在 `cuda:0`，MemSifter 排序器运行在 `cuda:1`。

```python
import json
from memsifter.toolkit import SessionEmbedder, SessionRanker, LLMChat

# 加载一个样本并解包所有字段
with open("data/test_memory.json") as f:
    entry = json.load(f)[0]

question             = entry["question"]
haystack_sessions    = entry["haystack_sessions"]
haystack_dates       = entry["haystack_dates"]
haystack_session_ids = entry["haystack_session_ids"]
answer_session_ids   = entry["answer_session_ids"]

# 初始化模型（加载一次，可复用）
embedder = SessionEmbedder(model_path="models/bge-m3", device="cuda:0")
ranker   = SessionRanker(
    model_path="models/zstanjj/MemSifter-4B-Thinking",
    device="cuda:1",
)
chat = LLMChat(api_key="YOUR_KEY", base_url="YOUR_BASE_URL", model_name="YOUR_MODEL")

# 阶段 1 — 嵌入预筛选
top_sessions = embedder.get_top_sessions(
    question=question, sessions=haystack_sessions, dates=haystack_dates, top_k=20
)

# 阶段 2 — 生成式重排序
ranked_sessions = ranker.rerank(
    question=question, pre_ranked_sessions=top_sessions, top_k=5
)

# 阶段 3 — LLM 生成答案
predicted_answer = chat.answer(question=question, ranked_sessions=ranked_sessions)

print("问题：", question)
print("答案：", predicted_answer)
```

### 输入数据格式

`data/test_memory.json` 中每条数据包含以下字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `question` | `str` | 用户查询 |
| `haystack_sessions` | `List[List[dict]]` | 所有候选会话；每个会话是由 `{"role": ..., "content": ...}` 组成的轮次列表 |
| `haystack_dates` | `List[str]` | 每个会话的时间戳 |
| `haystack_session_ids` | `List[str]` | 每个会话的唯一 ID |
| `answer` | `str` | 标准答案（仅用于评估） |
| `answer_session_ids` | `List[str]` | 包含答案的会话 ID（仅用于评估） |

### Toolkit API 简介

**`SessionEmbedder(model_path, device="cuda:0")`**
- `get_top_sessions(question, sessions, dates=None, top_k=20)` → `List[(idx, session_turns, date, score)]`

**`SessionRanker(model_path, device="cuda:1")`**
- `rerank(question, pre_ranked_sessions, top_k=5)` → `List[(idx, session_turns, date)]`

**`LLMChat(api_key, base_url, model_name)`**
- `answer(question, ranked_sessions)` → `str`

## 📊 复现实验结果

本节介绍如何使用已发布的 MemSifter 检查点对所有基准数据集进行批量推理。批量推理流水线使用 Ray 进行分布式多 GPU 推理。

### 前置条件

运行脚本前，请先启动 Ray 集群：

```bash
ray start --head
```

设置所需的环境变量：

```bash
export API_KEY="YOUR_LLM_API_KEY"
export BASE_URL="YOUR_LLM_BASE_URL"
export CUDA_VISIBLE_DEVICES=0,1
```

### 步骤 1 — 会话嵌入

计算基准数据集中所有会话的 bge-m3 嵌入，并存储相似度分数。

```bash
cd scripts/infer
./session_embedding.sh
```

关键变量（在脚本内编辑或运行前导出）：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `EMBEDDING_MODEL_NAME` | `bge-m3` | `models/` 下的嵌入模型名称 |
| `DATA_DIR` | `../data` | 根数据目录 |
| `OUTPUT_DIR` | `../data/results` | 嵌入结果保存路径 |
| `EMBED_STORE_PATH` | `../data/embedding_store` | 持久化嵌入缓存路径 |

### 步骤 2 — 会话排序

对嵌入预筛选后的候选项运行 MemSifter 生成式排序器。

```bash
cd scripts/infer
./session_ranking.sh
```

关键变量：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `MODEL_NAME` | `zstanjj/MemSifter-4B-Thinking` | `models/` 下的 MemSifter 检查点名称 |
| `RUNTIME_ENV` | `./configs/runtime_env.yaml` | Ray 运行时环境配置 |

### 步骤 3 — 对话推理

将排序后的会话传递给对话 LLM，收集生成的答案。

```bash
cd scripts/infer
./chat_infer.sh
```

关键变量：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `MODEL_NAME` | — | 对话模型名称（传递给 API） |
| `MODEL_PATH` | — | 本地模型路径（用于 tokenizer） |
| `API_KEY` | — | LLM API 密钥 |
| `BASE_URL` | — | LLM API 基础 URL |
| `MAX_OUTPUT_TOKEN` | `4096` | 最大生成 token 数 |
| `TEMPERATURE` | `0.6` | 采样温度 |

推理完成后停止 Ray：

```bash
ray stop
```

## 🏋️ 模型训练

本节介绍如何在自定义数据上训练你自己的 MemSifter 排序器。

### 步骤 1 — 准备强化学习训练数据

首先在自己的数据集上运行嵌入和排序步骤（参见[复现实验结果](#-复现实验结果)），然后准备 DAPO 训练数据：

```bash
cd scripts/train
./prepare_rl_data.sh
```

关键变量：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `RECIPE` | `configs/dataset_recipe_v1.yaml` | 数据集配方 YAML 文件 |
| `PRIMARY_DATA_DIR` | `../data/results/DAPO-GenRank/...` | 使用锚点采样的数据（基于 NDCG） |
| `FALLBACK_DATA_DIR` | `../data/results/bge-m3` | 使用随机采样的备用数据 |
| `OUTPUT_DIR` | `../data` | 输出根目录 |
| `VERSION` | 自动生成（`v{MMDD}-0`） | 生成数据集的版本标签 |

输出文件：
- `{OUTPUT_DIR}/rl_train_data/{VERSION}/train_*.parquet`
- `{OUTPUT_DIR}/rl_train_data/{VERSION}/test.parquet`

### 步骤 2 — DAPO 强化学习训练

```bash
cd scripts/train
./qwen3_4b_task_reward.sh
```

关键变量：

| 变量 | 说明 |
|---|---|
| `MODEL_PATH` | 基础模型路径（例如 `Qwen3-4B`） |
| `CKPTS_DIR` | 检查点保存目录 |
| `TRAIN_FILE` | 训练 parquet 文件路径 |
| `TEST_FILE` | 测试 parquet 文件路径 |
| `NNODES` | 训练节点数（默认：`1`） |
| `RUNTIME_ENV` | Ray 运行时环境配置 |

训练使用 DAPO 算法的**任务奖励**模式（边际效用奖励 + 排序敏感奖励）。

### 步骤 3 — 转换与合并检查点

**将 VERL 检查点转换为 HuggingFace 格式：**

```bash
cd scripts/train
./collect_verl_ckpt.sh
```

**通过权重平均合并多个检查点步骤（可选但推荐）：**

```bash
export CKPT_STEPS="20 30 40"
export MODEL_NAME="MemSifter-Qwen3-4B-Task-Reward"
./merge_ckpts.sh
```

合并后的模型保存至 `{MODEL_DIR}/{MODEL_NAME}-merged`。

## 📝 引用

如果你在研究中使用了 MemSifter，请引用：

```bibtex
@misc{memsifter,
      title={MemSifter: Offloading LLM Memory Retrieval via Outcome-Driven Proxy Reasoning}, 
      author={Jiejun Tan and Zhicheng Dou and Liancheng Zhang and Yuyang Hu and Yiruo Cheng and Ji-Rong Wen},
      year={2026},
      eprint={2603.03379},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2603.03379}, 
}
```
