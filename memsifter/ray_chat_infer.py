import argparse
import os
import random
import sys
import time
from typing import Dict, Any, List

import ray
import yaml
from easydict import EasyDict as edict
from loguru import logger
from openai import OpenAI
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
sys.path.append("./")
from utils.eval_utils import dedup_indexes
from utils.ray_gen_utils import start_local_ray, parse_haystack_sessions
from utils.eval_generation_utils import eval_generation_data
from utils.session_process import construct_history_text, construct_session_text

def load_chat_prompt(dataset_name: str):
    if dataset_name in [
        "2WikiMultihopQA",
        "HotpotQA",
        "MegaScience",
        "MuSiQue",
        "OneGen-TrainDataset-MultiHopQA",
        "QA-Expert-Multi-Hop-V1.0",
        "TaskCraft",
        "Voyager1.0",
        "WebDancer",
        "WebShaper",
        "WebWalkerQA-Silver",
        "WikiTables",
    ]:
        default_path = "configs/deepsearch_default_prompt.yaml"
    else:
        default_path = "configs/chat_default_prompt.yaml"
    # using chat_default_prompt.yaml as default
    if os.path.exists(os.path.join("configs", f"chat_{dataset_name}_prompt.yaml")):
        prompt_config_path = os.path.join("configs", f"{dataset_name}_prompt.yaml")
    else:
        prompt_config_path = default_path
    with open(prompt_config_path, "r") as f:
        prompt_config = edict(yaml.safe_load(f))

    return dict(prompt_config)


def construct_retrieved_history_with_limited_context(
        sessions,
        max_prompt_tokens,
        tokenizer,
        session_dates=None,
        sample_strategy="random",
        session_sim_scores=None,
):
    # Initialize selected session indices list and token count
    selected_session_indices = []
    total_num_tokens = 0

    # Create a search order list (stores indices 0, 1, 2...)
    search_order = list(range(len(sessions)))
    assert len(sessions) == len(search_order), f"sessions length {len(sessions)} not equal to search_order length {len(search_order)}"

    if sample_strategy == "random":
        random.shuffle(search_order)
    elif sample_strategy == "similarity":
        if session_sim_scores is None:
            raise ValueError("session_sim_scores must be provided when sample_strategy is 'similarity'")

        # --- Fix section start ---
        # Sort indices based on session_sim_scores
        # x represents index (int), session_sim_scores[x] gets the corresponding score
        search_order.sort(key=lambda x: session_sim_scores[x], reverse=True)
        # --- Fix section end ---

    else:
        raise ValueError(f"sample_strategy {sample_strategy} not supported")

    # Traverse in determined order (search_order stores original indices sid)
    for sid in search_order:
        session_turns = sessions[sid]  # Use original index to get content
        session_text = construct_session_text(session_turns)

        if session_dates is not None:
            session_date = session_dates[sid]
            session_text = f"<session42> <date>{session_date}</date>\n{session_text}</session>\n"
        else:
            session_text = f"<session42>\n{session_text}</session>\n"

        num_tokens = len(tokenizer.tokenize(session_text))

        # If adding this session won't exceed token limit, add its index
        if total_num_tokens + num_tokens <= max_prompt_tokens:
            selected_session_indices.append(sid)
            total_num_tokens += num_tokens
        else:
            # For similarity strategy, usually from high to low scores, if one doesn't fit, smaller ones later might fit?
            # But usually to ensure relevance, break here is reasonable; if to fill context, can continue
            break

    # Sort selected session indices in original order
    selected_session_indices.sort()

    return selected_session_indices


class ChatInferActor:
    def __init__(self, prompt_config: Dict[str, Any], api_key: str, base_url: str, model_name: str, method: str,
                 max_input_tokens: int, model_path: str, generation_params: Dict[str, Any]):
        self.prompt_config = prompt_config
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # check model exists - 修改：只记录日志，不抛出异常
        # try:
        #     self.client.models.retrieve(model_name)
        # except Exception as e:
        #     logger.error(f"Model {model_name} does not exist. Please check the model name: {e}")
        #     # 不抛出异常，允许程序继续运行

        self.model_name = model_name
        self.method = method
        self.max_input_tokens = max_input_tokens
        self.generation_params = generation_params

    def filter_history_chats(self, row: Dict[str, Any]) -> str:

        # Case 1: Direct retrieved memory (String)
        if "retrieved_memory" in row:
            text = row["retrieved_memory"]

        haystack_sessions = row["haystack_sessions"]
        haystack_session_ids = row["haystack_session_ids"]
        haystack_dates = row.get("haystack_dates", None)
        answer_session_ids = row["answer_session_ids"]
        answer_session_idx = []
        for sid, (session, session_id) in enumerate(zip(haystack_sessions, haystack_session_ids)):
            if session_id in answer_session_ids:
                answer_session_idx.append(sid)
        question = row["question"]
        question_token_num = len(self.tokenizer.tokenize(question))
        haystack_session_ids = row.get("haystack_session_ids", [])

        # Case 2: Similarity based sorting (High -> Low)
        if self.method in ["bge-m3"]:
            similarities = row['similarities']
            selected_session_indices = construct_retrieved_history_with_limited_context(
                haystack_sessions,
                self.max_input_tokens - question_token_num,
                self.tokenizer,
                haystack_dates,
                "similarity",
                similarities,
            )
            selected_sessions = [haystack_sessions[sid] for sid in selected_session_indices]
            if haystack_dates is not None:
                selected_dates = [haystack_dates[sid] for sid in selected_session_indices]
            else:
                selected_dates = None
            history = construct_history_text(selected_sessions, selected_dates)
            return history

        # Case 3: Generative Ranking
        elif self.method in ["ep2-DAPO-Qwen3-4B-Thinking"] or "DAPO-GenRank" in self.method:
            similarities = row['similarities']
            assert len(similarities) == len(haystack_sessions), f"similarities length {len(similarities)} not equal to haystack_sessions length {len(haystack_sessions)}"
            rank_scores = similarities
            haystack_session_gen_rankings = [i for i in row['haystack_session_gen_rankings'] if
                                             i < len(haystack_session_ids)]
            pred_ranking_indexes = dedup_indexes(haystack_session_gen_rankings)
            rank_score = len(pred_ranking_indexes)
            # Map rankings back to session text
            for index in pred_ranking_indexes:
                if index < len(haystack_sessions):
                    rank_scores[index] = rank_score * 10
                    rank_score -= 1

            selected_session_indices = construct_retrieved_history_with_limited_context(
                haystack_sessions,
                self.max_input_tokens - question_token_num,
                self.tokenizer,
                haystack_dates,
                "similarity",
                rank_scores,
            )
            selected_sessions = [haystack_sessions[sid] for sid in selected_session_indices]
            if haystack_dates is not None:
                selected_dates = [haystack_dates[sid] for sid in selected_session_indices]
            else:
                selected_dates = None
            history = construct_history_text(selected_sessions, selected_dates)
            return history


        # Case 4: Default (Original Order)
        else:
            logger.warning(f"Method {self.method} not supported.")
            raise ValueError(f"Method {self.method} not supported.")


    def retry_request(self, messages: List[Dict[str, str]], max_retries: int = 5) -> Any:
        """
        Retry API request with exponential backoff strategy
        
        Args:
            messages: List of messages to send
            max_retries: Maximum number of retries
            
        Returns:
            API response object
            
        Raises:
            Exception: If all retries fail
        """
        for attempt in range(max_retries):
            try:
                generation_params = {
                    "model": self.model_name,
                    "messages": messages,
                    **self.generation_params,
                }
                response = self.client.chat.completions.create(**generation_params)
                return response
            except Exception as e:
                if attempt == max_retries - 1:
                    # Last retry failed, raise exception
                    raise e
                else:
                    # Calculate exponential backoff wait time (2^(attempt) seconds)
                    wait_time = 2 ** attempt
                    logger.warning(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying after {wait_time} seconds...")
                    time.sleep(wait_time)

    def __call__(self, row: Dict[str, Any]) -> Dict[str, Any]:
        # Get history chat records
        if "history_chats" in row:
            history_chats = row["history_chats"]
        else:
            history_chats = self.filter_history_chats(row)

        question = row["question"]
        # format user prompt
        user_prompt = self.prompt_config["prompt_template"]["user_prompt_text"].format(
            history_chats=history_chats, question=question
        )

        # call openai api with retry
        try:
            messages = [
                {"role": "system", "content": self.prompt_config["prompt_template"]["system_prompt_text"]},
                {"role": "user", "content": user_prompt},
            ]
            
            response = self.retry_request(messages)
            generated_text = response.choices[0].message.content  # Add generated_text field
            if "<answer>" in generated_text:
                try:
                    generated_text_start = generated_text.find("<answer>") + len("<answer>")
                    generated_text_end = generated_text.find("</answer>")
                    generated_text = generated_text[generated_text_start:generated_text_end]
                except:
                    logger.info(f"<answer> tag not closed, original text: {generated_text}")
            row["generated_text"] = generated_text
            row["api_status"] = "success"
        except Exception as e:
            logger.error(f"API call failed: {e}")
            row["generated_text"] = ""  # Set this field even on failure
            row["api_status"] = f"error: {e}"

        return row


def chat_inference(args: Dict[str, Any]):
    """
    Main function for chat inference
    """
    input_dir = args["input_dir"]
    output_dir = args["output_dir"]
    dataset_name = args["dataset_name"]
    api_key = os.environ["API_KEY"]
    base_url = os.environ["BASE_URL"]
    model_name = args["model_name"]
    method = args["method"]
    num_actors = args["num_actors"]
    dataset_split = args["dataset_split"]
    debug = args.get("debug", False)

    # Retrieve max_input_tokens
    max_input_tokens = args.get("max_input_tokens", 8192)
    # Build input and output file paths
    if method in ["bge-m3"]:
        input_file = f"{input_dir}/{method}/{dataset_name}_{dataset_split}_embed.parquet"
    elif method in ["hipporag", "nemori", "a_mem"]:
        input_file = f"{input_dir}/{method}/{dataset_name}_{dataset_split}_retrieved_memory.parquet"
    else:
        input_file = f"{input_dir}/{method}/{dataset_name}_{dataset_split}_ranking.parquet"

    output_file = f"{output_dir}/{model_name}/{method}/{dataset_name}_{dataset_split}_chat.parquet"
    os.makedirs(f"{output_dir}/{model_name}/{method}", exist_ok=True)

    # Load prompt configuration
    prompt_config = load_chat_prompt(dataset_name)
    logger.info(f"Prompt configuration loaded")

    # Load input data
    logger.info(f"Loading input file: {input_file}")
    # Check file type and use corresponding read method
    if input_file.endswith('.jsonl'):
        ds = ray.data.read_json(input_file, lines=True)
    elif input_file.endswith('.parquet'):
        ds = ray.data.read_parquet(input_file)
    else:
        raise ValueError(f"Unsupported file format: {input_file}")

    logger.info(f"Loaded {ds.count()} entries")

    if debug:
        ds = ds.limit(10)
        logger.info(f"Debug mode enabled, processing only {ds.count()} entries")

    ds = ds.map(parse_haystack_sessions)
    # Use ChatInferActor to process data
    ds = ds.map(
        fn=ChatInferActor,
        fn_constructor_kwargs={
            "prompt_config": prompt_config,
            "api_key": api_key,
            "base_url": base_url,
            "model_name": model_name,
            "method": method,
            "max_input_tokens": max_input_tokens,  # Added param here
            "model_path": "../models/Qwen3-4B-Instruct",
            "generation_params": {
                "temperature": args.get("temperature", 0.6),
                "max_tokens": args.get("max_output_tokens", 4096),
            },
        },
        concurrency=num_actors,
        num_cpus=0.5,
    )

    # Save results
    logger.info(f"Saving results to: {output_file}")
    if output_file.endswith('.jsonl'):
        ds.to_pandas().to_json(output_file, lines=True, orient="records", force_ascii=False)
    elif output_file.endswith('.parquet'):
        ds.to_pandas().to_parquet(output_file, index=False, compression="zstd")
    else:
        # Default to JSONL format
        ds.to_pandas().to_json(output_file, lines=True, orient="records", force_ascii=False)

    logger.info(f"Save results to {output_file}")

    # Get results for evaluation (optional)
    df = ds.to_pandas()
    if dataset_name in ["zh4o", "perltqa_zh"]:
        is_chinese = True
    else:
        is_chinese = False
    eval_res = eval_generation_data(df, debug, True, is_chinese)
    logger.info(f"Finished evaluation for {df.count()} rows")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Chat inference using Ray concurrent API calls")
    parser.add_argument("--input_dir", required=True, help="Input directory containing embedding or ranking files")
    parser.add_argument("--output_dir", required=True, help="Output directory to save chat results")
    parser.add_argument("--dataset_name", default="default", help="Dataset name for loading corresponding prompt configuration")
    parser.add_argument("--model_name", default="gpt-3.5-turbo", help="Model name")
    parser.add_argument("--method", default="default", help="Retrieval method")
    parser.add_argument("--num_actors", type=int, default=4, help="Number of Ray Actors")
    parser.add_argument('--debug', action="store_true", help='Enable debug mode')
    parser.add_argument('--local', action="store_true", help='Use local ray mode')
    parser.add_argument("--concurrency", type=int, default=8, help="Number of concurrent calls")
    parser.add_argument("--temperature", type=float, default=0.6, help="Model temperature parameter")
    parser.add_argument("--max_output_tokens", type=int, default=1024, help="Maximum output tokens")
    parser.add_argument("--max_input_tokens", type=int, default=8192, help="Maximum input tokens")
    parser.add_argument("--dataset_split", default="test", help="Dataset split, e.g., train, test")


    args = parser.parse_args()

    # Convert arguments to dictionary
    args_dict = vars(args)

    # If in local mode, start local ray
    if args_dict["local"]:
        start_local_ray()

    # Call chat_inference function
    chat_inference(args_dict)


if __name__ == "__main__":
    main()