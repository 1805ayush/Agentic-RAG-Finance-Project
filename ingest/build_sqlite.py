import os
import re
import json
import sqlite3
from pathlib import Path

from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "./indices/finance_rag.db")
DATA_PATH      = os.getenv("DATA_PATH",       "./data/finance_corpus.jsonl")

SCHEMA = """
CREATE TABLE IF NOT EXISTS passages (
    passage_id   TEXT PRIMARY KEY,
    passage_text TEXT NOT NULL,
    query        TEXT,
    query_id     TEXT,
    query_type   TEXT,
    answer       TEXT,
    is_selected  INTEGER DEFAULT 0,
    url          TEXT,
    source       TEXT,
    sentiment    TEXT,
    ticker       TEXT
);

CREATE TABLE IF NOT EXISTS corpus_stats (
    dimension TEXT,
    value     TEXT,
    count     INTEGER,
    PRIMARY KEY (dimension, value)
);

CREATE INDEX IF NOT EXISTS idx_ticker      ON passages (ticker);
CREATE INDEX IF NOT EXISTS idx_sentiment   ON passages (sentiment);
CREATE INDEX IF NOT EXISTS idx_query_type  ON passages (query_type);
CREATE INDEX IF NOT EXISTS idx_source      ON passages (source);
"""

def build_sqlite_db(data_path: str = DATA_PATH, db_path: str = SQLITE_DB_PATH):
    print(f"\nBuilding SQLite DB from {data_path} …")

    with open(data_path) as f:
        records = [json.loads(line) for line in f]

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur  = conn.cursor()
    cur.executescript(SCHEMA)

    rows = [
        (
            r["passage_id"], r["passage_text"],
            r.get("query", ""), r.get("query_id", ""),
            r.get("query_type", ""), r.get("answer"),
            int(r.get("is_selected", False)),
            r.get("url", ""), r.get("source", ""),
            r.get("sentiment", "unknown"), r.get("ticker", ""),
        )
        for r in tqdm(records, desc="  Inserting")
    ]

    cur.executemany("""
        INSERT OR REPLACE INTO passages
        (passage_id, passage_text, query, query_id, query_type,
         answer, is_selected, url, source, sentiment, ticker)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, rows)

    cur.execute("DELETE FROM corpus_stats")
    for dim in ("query_type", "sentiment", "source", "ticker"):
        cur.execute(f"""
            INSERT INTO corpus_stats (dimension, value, count)
            SELECT ?, {dim}, COUNT(*) FROM passages
            WHERE {dim} != '' GROUP BY {dim}
        """, (dim,))

    conn.commit()
    conn.close()
    print(f"\nSQLite DB built → {db_path}  ({len(rows)} passages)")

def get_connection(db_path: str = SQLITE_DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def _format(rows: list[dict]) -> list[dict]:
    return [
        {
            "text":             r["passage_text"],
            "score":            float(r.get("is_selected", 0)),
            "source":           r.get("url", ""),
            "query_type":       r.get("query_type", ""),
            "sentiment":        r.get("sentiment", ""),
            "ticker":           r.get("ticker", ""),
            "retrieval_method": "sql",
        }
        for r in rows
    ]

def sql_query(natural_query: str, db_path: str = SQLITE_DB_PATH, n_results: int = 5):
    conn = get_connection(db_path)
    cur  = conn.cursor()
    q    = natural_query.lower().strip()

    # Aggregation / count
    if any(w in q for w in ["how many", "count", "total", "breakdown", "distribution"]):
        cur.execute("SELECT dimension, value, count FROM corpus_stats ORDER BY dimension, count DESC")
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        summary = {}
        for row in rows:
            summary.setdefault(row["dimension"], {})[row["value"]] = row["count"]
        return [{
            "text":             json.dumps(summary, indent=2),
            "score":            1.0,
            "source":           "sqlite://corpus_stats",
            "retrieval_method": "sql",
        }]

    # Ticker filter
    m = re.search(r"\b([A-Z]{2,5})\b", natural_query)
    if m:
        cur.execute(
            "SELECT * FROM passages WHERE ticker = ? ORDER BY is_selected DESC LIMIT ?",
            (m.group(1), n_results),
        )
        rows = [dict(r) for r in cur.fetchall()]
        if rows:
            conn.close()
            return _format(rows)

    # Sentiment filter
    for s in ("positive", "negative", "neutral"):
        if s in q:
            cur.execute("SELECT * FROM passages WHERE sentiment = ? LIMIT ?", (s, n_results))
            rows = [dict(r) for r in cur.fetchall()]
            conn.close()
            return _format(rows)

    # Query-type filter
    for qt in ("opinion", "factual", "sentiment", "market_data"):
        if qt in q:
            cur.execute("SELECT * FROM passages WHERE query_type = ? LIMIT ?", (qt, n_results))
            rows = [dict(r) for r in cur.fetchall()]
            conn.close()
            return _format(rows)

    # Fallback: LIKE text search
    keywords = " ".join(q.split()[:6])
    cur.execute("SELECT * FROM passages WHERE passage_text LIKE ? LIMIT ?",
                (f"%{keywords}%", n_results))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return _format(rows)

if __name__ == "__main__":
    build_sqlite_db()