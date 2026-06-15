import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

DATA_PATH       = os.getenv("DATA_PATH",       "./data/finance_corpus.jsonl")
CHROMA_DB_PATH  = os.getenv("CHROMA_DB_PATH",  "./indices/chroma_db")
BM25_INDEX_PATH = os.getenv("BM25_INDEX_PATH", "./indices/bm25.pkl")
SQLITE_DB_PATH  = os.getenv("SQLITE_DB_PATH",  "./indices/msmarco.db")
FIQA_SIZE       = int(os.getenv("FIQA_SIZE",   "3000"))
PHRASE_SIZE     = int(os.getenv("PHRASE_SIZE", "2000"))
SKIP_CHROMA     = os.getenv("SKIP_CHROMA",     "0") == "1"


def banner(n: int, title: str):
    print(f"\n{'═'*56}\n  STEP {n} — {title}\n{'═'*56}")


def timed(fn, *args, **kwargs):
    t0 = time.time()
    r = fn(*args, **kwargs)
    print(f"  ⏱  {time.time() - t0:.1f}s")
    return r


if __name__ == "__main__":
    t_start = time.time()

    banner(1, "Load finance corpus (FiQA + PhraseBank)")
    from ingest.load_finance_data import load_and_save_finance
    timed(load_and_save_finance, DATA_PATH, FIQA_SIZE, PHRASE_SIZE)

    if SKIP_CHROMA:
        print("\n  SKIP_CHROMA=1 — skipping embedding")
    else:
        banner(2, "Build ChromaDB semantic index")
        print("  (slow step — ~5 min on CPU)")
        from ingest.embed_chroma import build_chroma_index
        timed(build_chroma_index, DATA_PATH, CHROMA_DB_PATH)

    banner(3, "Build BM25 keyword index")
    from ingest.index_bm25 import build_bm25_index
    timed(build_bm25_index, DATA_PATH, BM25_INDEX_PATH)

    banner(4, "Build SQLite structured store")
    from ingest.build_sqlite import build_sqlite_db
    timed(build_sqlite_db, DATA_PATH, SQLITE_DB_PATH)

    banner(5, "Verify all indices")
    from ingest.verify_indices import verify_chroma, verify_bm25, verify_sqlite, verify_tavily
    verify_chroma()
    verify_bm25()
    verify_sqlite()
    verify_tavily()

    total = time.time() - t_start
    print(f"\n{'═'*56}")
    print(f"PHASE 1 COMPLETE  ({total:.0f}s)")
    print(f"{'═'*56}\n")
    print(f"  Indices:\n    {CHROMA_DB_PATH}/\n    {BM25_INDEX_PATH}\n    {SQLITE_DB_PATH}\n")
