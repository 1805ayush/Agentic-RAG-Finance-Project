import os
import re
import json
import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

BM25_INDEX_PATH = os.getenv("BM25_INDEX_PATH", "./indices/bm25.pkl")
DATA_PATH       = os.getenv("DATA_PATH",        "./data/finance_corpus.jsonl")

_FINANCE_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "company", "companies", "stock", "stocks", "shares", "market", "markets",
    "said", "says", "according", "also", "its", "their", "this", "that",
}

def finance_tokenize(text: str) -> list[str]:
    """
    Domain-aware tokenizer:
      - Preserves tickers (AAPL, BRK.B) uppercase
      - Preserves dollar amounts ($1.2B) and percentages (3.5%)
      - Lowercases everything else and removes stopwords
    """
    raw = re.findall(
        r"[A-Z]{2,5}(?:\.[AB])?|\d[\d.,]*%?|\$[\d.,]+[BMKTbmkt]?|\w+",
        text,
    )
    tokens = []
    for tok in raw:
        if re.match(r"^[A-Z]{2,5}(?:\.[AB])?$", tok):
            tokens.append(tok)
        elif re.match(r"^\$?[\d.,]+[BMKTbmkt%]?$", tok):
            tokens.append(tok)
        else:
            lower = tok.lower()
            if lower not in _FINANCE_STOPWORDS and len(lower) > 1:
                tokens.append(lower)
    return tokens

def build_bm25_index(data_path: str = DATA_PATH, index_path: str = BM25_INDEX_PATH):
    print(f"\nBuilding BM25 index from {data_path} …")

    with open(data_path) as f:
        records = [json.loads(line) for line in f]

    print(f"  Tokenizing {len(records)} passages …")
    corpus_tokens = [finance_tokenize(r["passage_text"]) for r in tqdm(records, desc="  Tokenizing")]

    print("  Fitting BM25Okapi …")
    bm25 = BM25Okapi(corpus_tokens)

    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "wb") as f:
        pickle.dump({"bm25": bm25, "records": records}, f)

    print(f"\nBM25 index saved → {index_path}  ({len(records)} docs)")

def load_bm25_index(index_path: str = BM25_INDEX_PATH):
    with open(index_path, "rb") as f:
        payload = pickle.load(f)
    return payload["bm25"], payload["records"]

def bm25_query(bm25, records, query: str, n_results: int = 5):
    tokens = finance_tokenize(query)
    scores = bm25.get_scores(tokens)

    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]

    output = []
    for idx in top:
        if scores[idx] == 0:
            break
        r = records[idx]
        output.append({
            "text":             r["passage_text"],
            "score":            round(float(scores[idx]), 4),
            "source":           r.get("url", ""),
            "query_type":       r.get("query_type", ""),
            "sentiment":        r.get("sentiment", ""),
            "ticker":           r.get("ticker", ""),
            "retrieval_method": "bm25_keyword",
        })
    return output


if __name__ == "__main__":
    build_bm25_index()


