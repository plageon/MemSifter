"""
memsifter/toolkit.py — Single-sample inference toolkit for MemSifter.

No Ray required. Designed for a 2-GPU setup:
  - GPU 0 (cuda:0): SessionEmbedder  — bge-m3 embedding model
  - GPU 1 (cuda:1): SessionRanker    — MemSifter generative ranking model
  - LLMChat                          — OpenAI-compatible API (no GPU required)

Quick usage::

    import json
    from memsifter.toolkit import SessionEmbedder, SessionRanker, LLMChat

    with open("data/test_memory.json") as f:
        entry = json.load(f)[0]

    question           = entry["question"]
    haystack_sessions  = entry["haystack_sessions"]
    haystack_dates     = entry["haystack_dates"]
    haystack_session_ids = entry["haystack_session_ids"]
    answer_session_ids = entry["answer_session_ids"]
    answer             = entry["answer"]   # ground-truth, for evaluation

    embedder = SessionEmbedder(model_path="models/bge-m3", device="cuda:0")
    ranker   = SessionRanker(model_path="models/zstanjj/MemSifter-4B-Thinking", device="cuda:1")
    chat     = LLMChat(api_key="YOUR_KEY", base_url="YOUR_BASE_URL", model_name="YOUR_MODEL")

    top_sessions    = embedder.get_top_sessions(question=question, sessions=haystack_sessions, dates=haystack_dates, top_k=20)
    ranked_sessions = ranker.rerank(question=question, pre_ranked_sessions=top_sessions, top_k=5)
    predicted_answer = chat.answer(question=question, ranked_sessions=ranked_sessions)
    print(predicted_answer)

Note on GPU assignment
----------------------
vLLM spawns its own worker processes that inherit the value of
``CUDA_VISIBLE_DEVICES`` at the time :class:`SessionRanker` is constructed.
The constructor temporarily sets ``CUDA_VISIBLE_DEVICES`` to the target GPU
index so that the vLLM workers run on the correct device.  The original value
is restored immediately after the engine is initialised.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from loguru import logger

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from utils.embed_utils import STEmbConfig, STEmbeddingModel, compute_cosine_similarity
from utils.eval_utils import robust_session_ranking_parse
from utils.session_process import construct_history_text, construct_session_text

# ---------------------------------------------------------------------------
# Ranking prompt used by the MemSifter model
# ---------------------------------------------------------------------------
_RANKER_USER_PROMPT = (
    "Given the following conversation sessions and a query, rank the sessions "
    "from most relevant to least relevant.\n\n"
    "{history}\n\n"
    "Query: {current}\n\n"
    "Output your reasoning inside <think>...</think> tags, then output a "
    "comma-separated list of 0-based session indices in descending order of "
    "relevance inside <ranking>...</ranking> tags.  Example for 3 sessions: "
    "<think> reasoning </think><ranking>2,0,1</ranking>"
)


# ---------------------------------------------------------------------------
# SessionEmbedder
# ---------------------------------------------------------------------------

class SessionEmbedder:
    """Rank sessions by embedding-based cosine similarity.

    Args:
        model_path: Path to the sentence-transformer embedding model
                    (e.g. ``"models/bge-m3"``).
        device:     Torch device string for the embedding model
                    (default ``"cuda:0"``).
        max_seq_len: Maximum sequence length fed to the encoder.
        batch_size:  Encoding batch size.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        max_seq_len: int = 512,
        batch_size: int = 32,
    ) -> None:
        config = STEmbConfig()
        config.embedding_model_name = model_path
        config.embedding_max_seq_len = max_seq_len
        config.embedding_batch_size = batch_size
        config.embedding_model_device = device
        self.model = STEmbeddingModel(config, embedding_model_name=model_path)
        logger.info(f"SessionEmbedder loaded on {device}: {model_path}")

    def get_top_sessions(
        self,
        question: str,
        sessions: List[List],
        dates: Optional[List[str]] = None,
        top_k: int = 20,
    ) -> List[Tuple[int, List, Optional[str], float]]:
        """Embed the query and all sessions; return top-*k* by cosine similarity.

        Args:
            question: The user query string.
            sessions: List of sessions (``haystack_sessions`` from the JSON).
                      Each session is a list of ``{"role": ..., "content": ...}`` turns.
            dates:    Optional list of date strings (``haystack_dates`` from the JSON),
                      one per session.
            top_k:    Maximum number of sessions to return.

        Returns:
            List of ``(original_idx, session_turns, date_or_None, score)``
            tuples sorted by descending similarity score.
        """

        session_texts = [construct_session_text(s) for s in sessions]
        all_texts = [question] + session_texts

        embeddings = self.model.batch_encode(all_texts)
        query_emb = embeddings[0]
        session_embs = embeddings[1:]

        similarities = compute_cosine_similarity(query_emb, session_embs)
        sorted_indices = np.argsort(similarities)[::-1][:top_k]

        return [
            (
                int(idx),
                sessions[idx],
                dates[idx] if dates is not None else None,
                float(similarities[idx]),
            )
            for idx in sorted_indices
        ]


# ---------------------------------------------------------------------------
# SessionRanker
# ---------------------------------------------------------------------------

class SessionRanker:
    """Fine-grained generative session ranker powered by MemSifter (vLLM).

    Args:
        model_path:       Path to the MemSifter checkpoint.
        device:           Target CUDA device (e.g. ``"cuda:1"``).  The integer
                          part is extracted and ``CUDA_VISIBLE_DEVICES`` is
                          temporarily set to that value while the vLLM engine
                          initialises, so that the engine runs on the correct
                          physical GPU.
        max_model_len:    Maximum context length for vLLM (tokens).
        max_output_tokens: Maximum tokens to generate per call.
        temperature:      Sampling temperature.
        gpu_memory_utilization: Fraction of GPU memory vLLM may use.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:1",
        max_model_len: int = 32768,
        max_output_tokens: int = 4096,
        temperature: float = 0.6,
        gpu_memory_utilization: float = 0.85,
    ) -> None:
        from vllm import LLM, SamplingParams  # imported here to avoid mandatory dep

        gpu_id = device.split(":")[-1] if ":" in device else "0"

        # Temporarily point vLLM workers at the target GPU.
        prev_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        try:
            self.llm = LLM(
                model=model_path,
                tensor_parallel_size=1,
                max_model_len=max_model_len,
                dtype="bfloat16",
                trust_remote_code=True,
                gpu_memory_utilization=gpu_memory_utilization,
                enable_chunked_prefill=True,
            )
        finally:
            # Always restore the original value.
            if prev_devices is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = prev_devices

        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_output_tokens,
        )
        logger.info(f"SessionRanker loaded on cuda:{gpu_id}: {model_path}")

    def rerank(
        self,
        question: str,
        pre_ranked_sessions: List[Tuple[int, List, Optional[str], float]],
        top_k: int = 5,
    ) -> List[Tuple[int, List, Optional[str]]]:
        """Re-rank pre-filtered sessions with the MemSifter model.

        Args:
            question:            The user query string.
            pre_ranked_sessions: Output of :meth:`SessionEmbedder.get_top_sessions`.
            top_k:               Number of sessions to return.

        Returns:
            List of ``(original_idx, session_turns, date_or_None)`` tuples
            sorted by MemSifter's predicted relevance (most relevant first).
        """
        orig_indices = [i for i, _, _, _ in pre_ranked_sessions]
        session_list = [s for _, s, _, _ in pre_ranked_sessions]
        date_list = [d for _, _, d, _ in pre_ranked_sessions]
        use_dates = any(d is not None for d in date_list)

        history = construct_history_text(
            session_list,
            session_dates=date_list if use_dates else None,
        )
        user_prompt = _RANKER_USER_PROMPT.format(history=history, current=question)
        messages = [{"role": "user", "content": user_prompt}]

        outputs = self.llm.chat(
            messages=[messages],
            sampling_params=self.sampling_params,
        )
        response_text = outputs[0].outputs[0].text

        # robust_session_ranking_parse returns 0-based local indices.
        local_ranked = robust_session_ranking_parse(response_text)

        # Build the final order: ranked first, then unranked (similarity order).
        seen: set = set()
        final_local: List[int] = []
        for li in local_ranked:
            if 0 <= li < len(orig_indices) and li not in seen:
                seen.add(li)
                final_local.append(li)
        for li in range(len(orig_indices)):
            if li not in seen:
                final_local.append(li)

        return [
            (orig_indices[li], session_list[li], date_list[li])
            for li in final_local[:top_k]
        ]


# ---------------------------------------------------------------------------
# LLMChat
# ---------------------------------------------------------------------------

class LLMChat:
    """Call an OpenAI-compatible chat API to produce a final answer.

    Args:
        api_key:           API key for the LLM service.
        base_url:          Base URL of the OpenAI-compatible endpoint.
        model_name:        Model name to request.
        max_output_tokens: Maximum tokens to generate.
        temperature:       Sampling temperature.
        prompt_config_path: Path to the YAML prompt template.
                            Defaults to ``configs/chat_default_prompt.yaml``.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        max_output_tokens: int = 1024,
        temperature: float = 0.6,
        prompt_config_path: str = "configs/chat_default_prompt.yaml",
    ) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature

        with open(prompt_config_path, "r") as f:
            cfg = yaml.safe_load(f)
        pt = cfg["prompt_template"]
        self.system_prompt: str = pt["system_prompt_text"].strip()
        self.user_prompt_tmpl: str = pt["user_prompt_text"]

    def answer(
        self,
        question: str,
        ranked_sessions: List[Tuple[int, List, Optional[str]]],
    ) -> str:
        """Generate an answer to *question* conditioned on *ranked_sessions*.

        Args:
            question:        The user's question string.
            ranked_sessions: Output of :meth:`SessionRanker.rerank`.

        Returns:
            The answer string extracted from between ``<answer>...</answer>``
            tags, or the full model response if those tags are absent.
        """
        sessions = [s for _, s, _ in ranked_sessions]
        dates = [d for _, _, d in ranked_sessions]
        use_dates = any(d is not None for d in dates)

        history_text = construct_history_text(
            sessions,
            session_dates=dates if use_dates else None,
        )
        user_prompt = self.user_prompt_tmpl.format(
            history_chats=history_text,
            question=question,
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
        )
        text: str = response.choices[0].message.content

        if "<answer>" in text and "</answer>" in text:
            start = text.find("<answer>") + len("<answer>")
            end = text.find("</answer>", start)
            return text[start:end].strip()
        return text.strip()
