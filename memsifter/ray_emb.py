import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
sys.path.append("./")
import os

from typing import Dict, Any, List
from loguru import logger
import ray
from utils.embed_utils import compute_mdhash_id
from utils.embed_utils import EmbeddingStore  # Replace with actual module path
from utils.session_process import construct_session_text
from utils.embed_utils import compute_cosine_similarity, STEmbConfig, STEmbedActor
from utils.ray_gen_utils import parse_haystack_sessions, dump_haystack_sessions


# collect text to be embedded, including haystack sessions and questions
class CollectMissingTextsActor:
    def __init__(self, embed_store: EmbeddingStore):
        self.embed_store = embed_store

    # filter out texts that are not embedded in the embedding store
    def filter_texts(self, texts: List[str]) -> List[Dict[str, Any]]:
        missing_result = self.embed_store.get_missing_string_hash_ids(texts)
        missing_texts = []
        for hid, t_dict in missing_result.items():
            missing_texts.append({
                "text": t_dict["content"],
                "hash_id": hid,
            })

        return missing_texts

    def __call__(self, entry: Dict[str, Any]) -> List[Dict[str, Any]]:
        texts = []
        question = entry["question"]
        haystack_sessions = entry["haystack_sessions"]

        # Collect question text
        texts.append(question)

        # Collect session text
        for sid, session_turns in enumerate(haystack_sessions):
            session_text = construct_session_text(session_turns)
            texts.append(session_text)

        missing_texts = self.filter_texts(texts)

        return missing_texts


class CalculateSimilarityActor:
    def __init__(self, emb_store: EmbeddingStore):
        self.emb_store = emb_store

    def __call__(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        question = entry["question"]
        question_hash_id = compute_mdhash_id(question, prefix=self.emb_store.namespace + "-")
        haystack_sessions = entry["haystack_sessions"]
        session_texts = []
        for sid, session_turns in enumerate(haystack_sessions):
            session_text = construct_session_text(session_turns)
            session_texts.append(session_text)
        session_hash_ids = [compute_mdhash_id(t, prefix=self.emb_store.namespace + "-") for t in session_texts]
        question_embedding = self.emb_store.get_embedding(question_hash_id)
        session_embeddings = self.emb_store.get_embeddings(session_hash_ids)
        similarities = compute_cosine_similarity(
            query_emb=question_embedding,
            candidate_embs=session_embeddings
        )
        entry["similarities"] = [float(s) for s in similarities]
        return entry


def main(args,):
    emb_config = STEmbConfig()
    emb_config.model_name = f"../models/{args['embedding_model_name']}"
    emb_config.max_seq_len = args["max_seq_len"]
    emb_config.batch_size = args["batch_size"]
    emb_config.embedding_batch_size = args["emb_batch_size"]
    emb_concurrency = args["emb_concurrency"]
    emb_model_name = args["embedding_model_name"]
    use_gpu = args["use_gpu"]

    dataset_name = args["dataset_name"]
    data_dir = args["data_dir"]
    output_dir = args["output_dir"]
    embed_store_path = args["embed_store_path"]

    db_filename = os.path.join(embed_store_path, f"{dataset_name}")

    embedding_store = EmbeddingStore(
        embedding_model=emb_config.model_name,
        db_filename=db_filename,
        batch_size=emb_config.embedding_batch_size,
        namespace=f"{dataset_name}"
    )

    input_file_name = f"{data_dir}/{dataset_name}/{args['dataset_split']}.parquet"
    output_file_name = f"{output_dir}/{emb_model_name}/{dataset_name}_{args['dataset_split']}_embed.parquet"
    entry_ds = ray.data.read_parquet(input_file_name).map(parse_haystack_sessions)

    logger.info(f"Loaded {entry_ds.count()} lines from {input_file_name}")
    missing_text_ds = entry_ds.flat_map(
        fn=CollectMissingTextsActor,
        fn_constructor_kwargs={
            "embed_store": embedding_store,
        },
        concurrency=emb_concurrency,
    )
    logger.info(f"Found {missing_text_ds.count()} missing texts")
    if use_gpu:
        num_gpus = 1
        num_cpus = None
    else:
        num_gpus = None
        num_cpus = 4
    if missing_text_ds.count() > 0:
        embedded_ds = missing_text_ds.map_batches(
            fn = STEmbedActor,
            fn_constructor_kwargs={
                "emb_config": emb_config,
                "embedding_model_name": emb_config.model_name,
                "use_gpu": use_gpu,
            },
            batch_size=emb_config.embedding_batch_size,
            concurrency=(int(emb_concurrency / 4), emb_concurrency),
            num_gpus=num_gpus,
            num_cpus=num_cpus,
        )
        embedded_pd = embedded_ds.to_pandas()
        # write to embedding store
        embedding_store.insert_embeddings(
            hash_ids=embedded_pd["hash_id"].tolist(),
            texts=embedded_pd["text"].tolist(),
            embeddings=embedded_pd["embedding"].tolist(),
        )
        logger.info(f"Inserted {len(embedded_pd)} embeddings into {db_filename}")
    else:
        logger.info(f"All texts are embedded in {db_filename}")

    # calculate cosine similarity
    similarity_ds = entry_ds.map(
        fn=CalculateSimilarityActor(embedding_store),
        concurrency=emb_concurrency,
    )
    # save to parquet
    similarity_ds = similarity_ds.map(dump_haystack_sessions)
    similarity_ds.to_pandas().to_parquet(output_file_name, index=False, compression="zstd")
    logger.info(f"Written {similarity_ds.count()} lines to {output_file_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--dataset_name", type=str, default="perltqa_en")
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--output_dir", type=str, default="../data/results")
    parser.add_argument("--embed_store_path", type=str, default="../data/embedding_store")
    parser.add_argument("--embedding_model_name", type=str, default="bge-m3")
    parser.add_argument("--emb_concurrency", type=int, default=8)
    parser.add_argument("--emb_batch_size", type=int, default=256)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--use_gpu", action="store_true")
    args = parser.parse_args()
    main(vars(args))



