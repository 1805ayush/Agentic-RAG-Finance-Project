import json
import os
import re
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

DATA_PATH   = os.getenv("DATA_PATH",   "./data/finance_corpus.jsonl")
FIQA_SIZE   = int(os.getenv("FIQA_SIZE",   "3000"))
PHRASE_SIZE = int(os.getenv("PHRASE_SIZE", "2000"))

_TICKER_RE = re.compile(r"\b([A-Z]{2,5}(?:\.[AB])?)\b")

_STOPWORDS = {
    "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN",
    "HER", "WAS", "ONE", "OUR", "OUT", "WHO", "GET", "HAS", "HIM",
    "HIS", "HOW", "ITS", "NEW", "NOW", "OLD", "SEE", "TWO", "WAY",
    "CEO", "CFO", "COO", "IPO", "GDP", "CPI", "FED", "SEC", "ETF",
    "USA", "USD", "EUR", "GBP", "YEN", "OIL", "GAS", "EPS", "ROE",
}

def extract_ticker(text: str) -> str:
    for match in _TICKER_RE.finditer(text):
        candidate = match.group(1)
        if candidate not in _STOPWORDS:
            return candidate
    return ""

def infer_query_type(text: str, source: str, sentiment: str) -> str:
    if source == "phrasebank":
        return "sentiment"
    t = text.lower()
    if any(w in t for w in ["what is", "explain", "how does", "define", "why"]):
        return "opinion"
    if any(w in t for w in ["price", "rate", "percent", "%", "$", "earnings", "revenue"]):
        return "market_data"
    return "factual"

def load_fiqa(n: int) -> list[dict]:
    print(f"\n[1/2] Loading FiQA corpus (up to {n} passages)…")
    try:
        ds = load_dataset("BeIR/fiqa", "corpus", split="corpus", trust_remote_code=True)
    except Exception as e:
        print(f"Could not load BeIR/fiqa corpus: {e}")
        ds = load_dataset("BeIR/fiqa-generated-queries", split=f"train[:{n}]", trust_remote_code=True)

    records = []
    for i, row in enumerate(tqdm(ds, total=min(n, len(ds)), desc="  FiQA passages")):
        if i >= n:
            break
        text = (row.get("text") or row.get("passage_text") or "").strip()
        if not text:
            continue

        records.append({
            "passage_id":   f"fiqa_{row.get('_id', i)}",
            "passage_text": text,
            "query":        "",
            "query_id":     str(row.get("_id", i)),
            "query_type":   infer_query_type(text, "fiqa", "unknown"),
            "answer":       None,
            "is_selected":  False,
            "url":          row.get("metadata", {}).get("url", "") if isinstance(row.get("metadata"), dict) else "",
            "source":       "fiqa",
            "sentiment":    "unknown",
            "ticker":       extract_ticker(text),
        })

    print(f"Loaded {len(records)} FiQA passages")
    return records

def load_phrasebank(n: int) -> list[dict]:
    print(f"\n[2/2] Loading Financial PhraseBank (up to {n} sentences)…")
    _LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

    try:
        ds = load_dataset(
            "descartes100/enhanced-financial-phrasebank",
            split=f"train[:{n}]",
        )
    except Exception as e:
        print(f"  ⚠  Could not load phrasebank mirror: {e}")
        return []

    records = []
    for i, row in enumerate(tqdm(ds, desc="  PhraseBank sentences")):
        # This mirror wraps the data in a 'train' column
        inner = row.get("train", row)
        text  = (inner.get("sentence") or "").strip()
        if not text:
            continue

        label_val = inner.get("label", 1)
        sentiment = _LABEL_MAP.get(int(label_val), "neutral")

        records.append({
            "passage_id":   f"pb_{i:05d}",
            "passage_text": text,
            "query":        f"What is the sentiment of: {text[:80]}",
            "query_id":     f"pb_{i:05d}",
            "query_type":   "sentiment",
            "answer":       sentiment,
            "is_selected":  True,
            "url":          "",
            "source":       "phrasebank",
            "sentiment":    sentiment,
            "ticker":       extract_ticker(text),
        })

    print(f"  Loaded {len(records)} PhraseBank sentences")
    return records

def save_corpus(records: list[dict], output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"\n  Saved {len(records)} records → {output_path}")

def print_stats(records: list[dict]):
    from collections import Counter
    sources    = Counter(r["source"]     for r in records)
    qtypes     = Counter(r["query_type"] for r in records)
    sentiments = Counter(r["sentiment"]  for r in records)
    tickers    = [r["ticker"] for r in records if r["ticker"]]

    print("\n── Finance Corpus Stats ───────────────────────────")
    print(f"  Total records    : {len(records)}")
    print(f"  By source        : {dict(sources)}")
    print(f"  By query_type    : {dict(qtypes)}")
    print(f"  By sentiment     : {dict(sentiments)}")
    print(f"  Records w/ ticker: {len(tickers)}  (e.g. {list(set(tickers))[:8]})")
    print("────────────────────────────────────────────────────\n")

def load_records(path: str = DATA_PATH) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]

def load_and_save_finance(output_path: str = DATA_PATH,fiqa_size:   int = FIQA_SIZE,phrase_size: int = PHRASE_SIZE) -> list[dict]:
    fiqa_records   = load_fiqa(fiqa_size)
    phrase_records = load_phrasebank(phrase_size)
    all_records = fiqa_records + phrase_records
    save_corpus(all_records, output_path)
    print_stats(all_records)
    return all_records


if __name__ == "__main__":
    load_and_save_finance()
    print("finance corpus ready.")







