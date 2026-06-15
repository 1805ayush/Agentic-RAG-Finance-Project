import os
import json
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

CHROMA_DB_PATH  = os.getenv("CHROMA_DB_PATH",  "./indices/chroma_db")
DATA_PATH       = os.getenv("DATA_PATH",        "./data/finance_corpus.jsonl")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL",  "sentence-transformers/all-MiniLM-L6-v2")
BATCH_SIZE      = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))
COLLECTION_NAME = "finance_passages"

def build_chroma_index(data_path: str = DATA_PATH,db_path:   str = CHROMA_DB_PATH,):
    print(f"\nBuilding ChromaDB index from {data_path} …")

    with open(data_path) as f:
        records = [json.loads(line) for line in f]

    print(f"  Records to embed : {len(records)}",flush=True)
    print(f"  Embedding model  : {EMBEDDING_MODEL}",flush=True)

    Path(db_path).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=db_path)

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    for i in tqdm(range(0, len(records), BATCH_SIZE), desc="  Embedding"):
        batch = records[i : i + BATCH_SIZE]
        collection.add(
            ids       = [r["passage_id"] for r in batch],
            documents = [r["passage_text"] for r in batch],
            metadatas = [
                {
                    "source":      r.get("source", ""),
                    "query_type":  r.get("query_type", ""),
                    "sentiment":   r.get("sentiment", "unknown"),
                    "ticker":      r.get("ticker", ""),
                    "url":         r.get("url", ""),
                    "answer":      r.get("answer") or "",
                    "is_selected": int(r.get("is_selected", False)),
                }
                for r in batch
            ],
        )

    count = collection.count()
    print(f"\nChromaDB index built — {count} passages at {db_path}",flush=True)
    return collection

def get_chroma_collection(db_path: str = CHROMA_DB_PATH):
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    client = chromadb.PersistentClient(path=db_path)
    return client.get_collection(name=COLLECTION_NAME, embedding_function=ef)

def semantic_query(collection, query: str, n_results: int = 5, filters: dict | None = None):
    kwargs = dict(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    if filters:
        kwargs["where"] = filters

    results = collection.query(**kwargs)

    output = []
    for doc, meta, dist in zip(results["documents"][0],results["metadatas"][0],results["distances"][0]):
        output.append({
            "text":             doc,
            "score":            round(1 - dist, 4),
            "source":           meta.get("url") or meta.get("source", ""),
            "query_type":       meta.get("query_type", ""),
            "sentiment":        meta.get("sentiment", ""),
            "ticker":           meta.get("ticker", ""),
            "retrieval_method": "semantic_vector",
        })
    return output

if __name__ == "__main__":
    build_chroma_index()
