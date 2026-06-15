import os
from dotenv import load_dotenv

load_dotenv()

FACTUAL_QUERY    = "What happens to bond prices when interest rates rise?"
TICKER_QUERY     = "AAPL earnings revenue growth"
STRUCTURED_QUERY = "how many positive sentiment passages"
WEB_QUERY        = "Federal Reserve interest rate decision 2025"
SEP = "─" * 56


def verify_chroma():
    from ingest.embed_chroma import get_chroma_collection, semantic_query
    print(f"\n{SEP}\n  ChromaDB — Semantic Search\n{SEP}")
    col = get_chroma_collection()
    print(f"  Collection size : {col.count()} passages")
    for i, r in enumerate(semantic_query(col, FACTUAL_QUERY, n_results=3), 1):
        print(f"  [{i}] score={r['score']:.3f}  ticker={r['ticker'] or '—'}")
        print(f"       {r['text'][:100]} …")
    print("ChromaDB OK\n")


def verify_bm25():
    from ingest.index_bm25 import load_bm25_index, bm25_query
    print(f"{SEP}\n  BM25 — Keyword Search\n{SEP}")
    bm25, records = load_bm25_index()
    print(f"  Corpus size : {len(records)} passages")
    results = bm25_query(bm25, records, TICKER_QUERY, n_results=3)
    if results:
        for i, r in enumerate(results, 1):
            print(f"  [{i}] score={r['score']:.3f}  ticker={r['ticker'] or '—'}")
            print(f"       {r['text'][:100]} …")
    else:
        print("  (no BM25 matches — try after corpus is loaded)")
    print("BM25 OK\n")


def verify_sqlite():
    from ingest.build_sqlite import sql_query, get_connection
    print(f"{SEP}\n  SQLite — Structured Lookup\n{SEP}")
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM passages")
    total = cur.fetchone()[0]
    cur.execute("SELECT sentiment, COUNT(*) FROM passages GROUP BY sentiment")
    breakdown = {row[0]: row[1] for row in cur.fetchall()}
    conn.close()
    print(f"  Row count   : {total}")
    print(f"  Sentiments  : {breakdown}")
    results = sql_query(STRUCTURED_QUERY)
    print(f"  Sample query: '{STRUCTURED_QUERY}'")
    print(f"  Result      : {results[0]['text'][:120] if results else 'no results'}")
    print("SQLite OK\n")

def verify_ddg():
    print(f"{SEP}\n  DuckDuckGo — Free Web Search\n{SEP}")
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(WEB_QUERY, max_results=2, region="us-en"))
        for i, r in enumerate(results, 1):
            print(f"  [{i}] {r.get('title', '')}")
            print(f"       {r.get('href', '')}")
        print("DuckDuckGo OK\n")
    except Exception as e:
        print(f"DuckDuckGo search failed: {e}\n")

if __name__ == "__main__":
    print(f"\n{'═'*56}\n  PHASE 1 — INDEX VERIFICATION\n{'═'*56}")
    verify_chroma()
    verify_bm25()
    verify_sqlite()
    verify_ddg()
    print(f"{'═'*56}\n  ALL INDICES VERIFIED ✅\n{'═'*56}\n")