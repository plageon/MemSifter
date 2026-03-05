# MemSifter: Offloading LLM Memory Retrieval via Outcome-Driven Proxy Reasoning

<div align="center">
<a href="https://arxiv.org/abs/2603.03379" target="_blank"><img src=https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv></a>
<a href="https://huggingface.co/zstanjj/MemSifter-4B-Thinking" target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace%20Models-27b3b4.svg></a>
<a href="https://www.modelscope.cn/models/zstanjj/MemSifter-4B-Thinking" target="_blank"><img src=https://custom-icon-badges.demolab.com/badge/ModelScope%20Models-624aff?style=flat&logo=modelscope&logoColor=white></a>
<a href="https://github.com/plageon/MemSifter/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green"></a>
<a><img alt="Static Badge" src="https://img.shields.io/badge/made_with-Python-blue"></a>
</div>

[English](README.md) | [中文](README_ZH.md)

## 📖 Table of Contents

- [Introduction](#-introduction)
- [Latest News](#-latest-news)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Reproduce Results](#-reproduce-results)
- [Training](#-training)
- [Citation](#-citation)

## ✨ Introduction

**MemSifter** is an LLM memory retrieval offloading system based on outcome-driven proxy reasoning.  Given a large pool of personal conversation sessions (the "haystack"), MemSifter efficiently identifies the sessions most relevant to a user query and passes them as context to a downstream chat LLM—without burdening the LLM itself with retrieval.

The system follows a three-stage pipeline:

```
Session Embedding  →  Session Ranking (MemSifter)  →  Chat LLM
   (bge-m3)           (generative reranker)          (any LLM)
```

1. **Session Embedding** — a dense embedding model (bge-m3) performs a coarse similarity pre-filter across all sessions.
2. **Session Ranking** — MemSifter, a lightweight generative model trained with DAPO reinforcement learning, performs fine-grained reranking of the pre-filtered candidates.
3. **Chat LLM** — the top-ranked sessions are assembled into a context window and passed to any OpenAI-compatible chat model to produce the final answer.

## 🗞 Latest News

- **[03/03/2026]** Our paper is available on [arXiv](https://arxiv.org/abs/2603.03379).
- **[20/02/2026]** Code, models, and data are released.

## 🔧 Installation

**Requirements:** Python 3.8+, two CUDA-capable GPUs (for Quick Start single-sample inference).

```bash
git clone https://github.com/plageon/MemSifter.git
cd MemSifter
pip install -r requirements.txt
```

Download the required models into a local `models/` directory:

| Model | Purpose | Source |
|---|---|---|
| `bge-m3` | Session embedding | [HuggingFace](https://huggingface.co/BAAI/bge-m3) |
| `zstanjj/MemSifter-4B-Thinking` | Generative session ranker | [HuggingFace](https://huggingface.co/zstanjj/MemSifter-4B-Thinking) |

```bash
# Example using huggingface-cli
huggingface-cli download BAAI/bge-m3 --local-dir models/bge-m3
huggingface-cli download zstanjj/MemSifter-4B-Thinking \
    --local-dir models/zstanjj/MemSifter-4B-Thinking
```

For the Quick Start single-sample inference (no Ray), install only the required packages:

```bash
pip install torch sentence-transformers vllm openai pyyaml loguru numpy pandas
```

## 🚀 Quick Start

The toolkit in `memsifter/toolkit.py` provides three classes for single-sample inference with **no Ray dependency**.  It assumes you have two GPUs: the embedding model runs on `cuda:0` and the MemSifter ranker runs on `cuda:1`.

```python
import json
from memsifter.toolkit import SessionEmbedder, SessionRanker, LLMChat

# Load one sample and unpack all fields
with open("data/test_memory.json") as f:
    entry = json.load(f)[0]

question             = entry["question"]
haystack_sessions    = entry["haystack_sessions"]
haystack_dates       = entry["haystack_dates"]
haystack_session_ids = entry["haystack_session_ids"]
answer_session_ids   = entry["answer_session_ids"]

# Initialise models (loaded once, reusable)
embedder = SessionEmbedder(model_path="models/bge-m3", device="cuda:0")
ranker   = SessionRanker(
    model_path="models/zstanjj/MemSifter-4B-Thinking",
    device="cuda:1",
)
chat = LLMChat(api_key="YOUR_KEY", base_url="YOUR_BASE_URL", model_name="YOUR_MODEL")

# Stage 1 — embedding pre-filter
top_sessions = embedder.get_top_sessions(
    question=question, sessions=haystack_sessions, dates=haystack_dates, top_k=20
)

# Stage 2 — generative reranking
ranked_sessions = ranker.rerank(
    question=question, pre_ranked_sessions=top_sessions, top_k=5
)

# Stage 3 — LLM answer
predicted_answer = chat.answer(question=question, ranked_sessions=ranked_sessions)

print("Question:", question)
print("Answer:  ", predicted_answer)
```

### Input data format

Each entry in `data/test_memory.json` has the following fields:

| Field | Type | Description |
|---|---|---|
| `question` | `str` | User query |
| `haystack_sessions` | `List[List[dict]]` | All candidate sessions; each session is a list of `{"role": ..., "content": ...}` turns |
| `haystack_dates` | `List[str]` | Timestamp for each session |
| `haystack_session_ids` | `List[str]` | Unique ID for each session |
| `answer` | `str` | Ground-truth answer (evaluation only) |
| `answer_session_ids` | `List[str]` | IDs of sessions containing the answer (evaluation only) |

### Toolkit API summary

**`SessionEmbedder(model_path, device="cuda:0")`**
- `get_top_sessions(question, sessions, dates=None, top_k=20)` → `List[(idx, session_turns, date, score)]`

**`SessionRanker(model_path, device="cuda:1")`**
- `rerank(question, pre_ranked_sessions, top_k=5)` → `List[(idx, session_turns, date)]`

**`LLMChat(api_key, base_url, model_name)`**
- `answer(question, ranked_sessions)` → `str`

## 📊 Reproduce Results

This section covers batch inference across all benchmark datasets using the released MemSifter checkpoint.  The batch pipeline uses Ray for distributed multi-GPU inference.

### Prerequisites

Start a Ray cluster before running the scripts:

```bash
ray start --head
```

Set the required environment variables:

```bash
export API_KEY="YOUR_LLM_API_KEY"
export BASE_URL="YOUR_LLM_BASE_URL"
export CUDA_VISIBLE_DEVICES=0,1
```

### Step 1 — Session Embedding

Computes bge-m3 embeddings for all sessions in the benchmark datasets and stores similarity scores.

```bash
cd scripts/infer
./session_embedding.sh
```

Key variables (edit inside the script or export before running):

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL_NAME` | `bge-m3` | Embedding model name under `models/` |
| `DATA_DIR` | `../data` | Root data directory |
| `OUTPUT_DIR` | `../data/results` | Where to save embedding results |
| `EMBED_STORE_PATH` | `../data/embedding_store` | Persistent embedding cache |

### Step 2 — Session Ranking

Runs the MemSifter generative ranker over the embedding-pre-filtered candidates.

```bash
cd scripts/infer
./session_ranking.sh
```

Key variables:

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `zstanjj/MemSifter-4B-Thinking` | MemSifter checkpoint name under `models/` |
| `RUNTIME_ENV` | `./configs/runtime_env.yaml` | Ray runtime environment config |

### Step 3 — Chat Inference

Passes the ranked sessions to a chat LLM and collects the generated answers.

```bash
cd scripts/infer
./chat_infer.sh
```

Key variables:

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | — | Chat model name (passed to the API) |
| `MODEL_PATH` | — | Local model path (for tokenizer) |
| `API_KEY` | — | LLM API key |
| `BASE_URL` | — | LLM API base URL |
| `MAX_OUTPUT_TOKEN` | `4096` | Max tokens to generate |
| `TEMPERATURE` | `0.6` | Sampling temperature |

Stop Ray after inference is complete:

```bash
ray stop
```

## 🏋️ Training

This section describes how to train your own MemSifter ranker on custom data.

### Step 1 — Prepare RL Training Data

Run the embedding and ranking steps on your own datasets first (see [Reproduce Results](#-reproduce-results)), then prepare the DAPO training data:

```bash
cd scripts/train
./prepare_rl_data.sh
```

Key variables:

| Variable | Default | Description |
|---|---|---|
| `RECIPE` | `configs/dataset_recipe_v1.yaml` | Dataset recipe YAML |
| `PRIMARY_DATA_DIR` | `../data/results/DAPO-GenRank/...` | Data with anchor sampling (NDCG-based) |
| `FALLBACK_DATA_DIR` | `../data/results/bge-m3` | Fallback data with random sampling |
| `OUTPUT_DIR` | `../data` | Output root |
| `VERSION` | auto (`v{MMDD}-0`) | Version tag for the generated split |

Outputs:
- `{OUTPUT_DIR}/rl_train_data/{VERSION}/train_*.parquet`
- `{OUTPUT_DIR}/rl_train_data/{VERSION}/test.parquet`

### Step 2 — DAPO Reinforcement Learning Training

```bash
cd scripts/train
./qwen3_4b_task_reward.sh
```

Key variables:

| Variable | Description |
|---|---|
| `MODEL_PATH` | Path to the base model (e.g. `Qwen3-4B`) |
| `CKPTS_DIR` | Directory to save checkpoints |
| `TRAIN_FILE` | Path to training parquet |
| `TEST_FILE` | Path to test parquet |
| `NNODES` | Number of training nodes (default: `1`) |
| `RUNTIME_ENV` | Ray runtime environment config |

The training uses **task reward** mode (Marginal Utility Reward + Rank-Sensitive Reward) via the DAPO algorithm.

### Step 3 — Convert & Merge Checkpoints

**Convert VERL checkpoints to HuggingFace format:**

```bash
cd scripts/train
./collect_verl_ckpt.sh
```

**Merge multiple checkpoint steps by weight averaging (optional but recommended):**

```bash
export CKPT_STEPS="20 30 40"
export MODEL_NAME="MemSifter-Qwen3-4B-Task-Reward"
./merge_ckpts.sh
```

The merged model is saved to `{MODEL_DIR}/{MODEL_NAME}-merged`.

## 📝 Citation

If you use MemSifter in your research, please cite:

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
