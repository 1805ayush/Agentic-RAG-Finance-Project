"""
Phase 2: Retrieval Tools
─────────────────────────
Four LangChain-compatible tools the agent can call.
"""

import os
from functools import lru_cache
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv()

N_RESULTS = 5


# ── Lazy loaders ─────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_chroma():
    """Load ChromaDB collection once and reuse it."""
    from ingest.embed_chroma import get_chroma_collection
    return get_chroma_collection()


@lru_cache(maxsize=1)
def _get_bm25():
    """Load BM25 index + records list once and reuse them."""
    from ingest.index_bm25 import load_bm25_index
    return load_bm25_index()


# ── Tool 1: Semantic Vector Search ───────────────────────────────────

@tool
def semantic_search(query: str) -> str:
    """
    Search for passages using semantic similarity (ChromaDB + embeddings).

    WHEN TO USE:
    - Conceptual or abstract questions about finance
      ("What is quantitative easing?", "Explain margin calls")
    - Questions about financial mechanisms, theory, or definitions
    - When the query is open-ended or exploratory
    - When wording flexibility matters more than exact terms

    Returns top relevant passages with similarity scores.
    """
    from ingest.embed_chroma import semantic_query
    try:
        results = semantic_query(_get_chroma(), query, n_results=N_RESULTS)
        return _format_results(results, "SEMANTIC SEARCH")
    except Exception as e:
        return f"[semantic_search error: {e}]"


# ── Tool 2: BM25 Keyword Search ──────────────────────────────────────

@tool
def bm25_keyword_search(query: str) -> str:
    """
    Search for passages using BM25 keyword/lexical matching.

    WHEN TO USE:
    - Queries with ticker symbols (AAPL, NVDA, TSLA, BRK.B)
    - Specific company or institution names (Goldman Sachs, Federal Reserve)
    - Technical financial terms or acronyms (EBITDA, FOMC, ROIC, IPO, P/E)
    - When exact word matches matter more than semantic meaning
    - Short factoid queries with proper nouns

    Returns top passages ranked by BM25 relevance score.
    """
    from ingest.index_bm25 import bm25_query
    try:
        bm25, records = _get_bm25()
        results = bm25_query(bm25, records, query, n_results=N_RESULTS)
        return _format_results(results, "BM25 KEYWORD SEARCH")
    except Exception as e:
        return f"[bm25_keyword_search error: {e}]"


# ── Tool 3: SQL Structured Lookup ────────────────────────────────────

@tool
def sql_lookup(query: str) -> str:
    """
    Query the structured SQLite database of financial passages.

    WHEN TO USE:
    - Counts, aggregations, or statistics
      ("How many positive-sentiment passages?")
    - Filtering by ticker, sentiment, or source
      ("Show me negative passages about TSLA")
    - Sentiment-specific lookups
    - Any "how many X" or "show me all Y" structured query

    Returns structured data from the SQLite metadata store.
    """
    from ingest.build_sqlite import sql_query
    try:
        results = sql_query(query, n_results=N_RESULTS)
        return _format_results(results, "SQL STRUCTURED LOOKUP")
    except Exception as e:
        return f"[sql_lookup error: {e}]"


# ── Tool 4: Live Web Search (DuckDuckGo, free) ───────────────────────

@tool
def web_search(query: str) -> str:
    """
    Search the live web using DuckDuckGo for current financial information.

    WHEN TO USE:
    - Recent market events, earnings releases, or breaking news (post-2023)
    - Current stock prices, interest rates, or economic indicators
    - When corpus results appear outdated or insufficient
    - Real-time financial data that the static corpus cannot have

    Returns live web results with titles, URLs, and excerpts.
    """
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=N_RESULTS, region="us-en"))

        if not raw:
            return "[WEB SEARCH]: No results found."

        results = [
            {
                "text":             r.get("body", "") or r.get("snippet", ""),
                "score":            0.0,
                "source":           r.get("href", "") or r.get("url", ""),
                "retrieval_method": "web_search",
            }
            for r in raw
        ]
        return _format_results(results, "WEB SEARCH (DuckDuckGo)")
    except Exception as e:
        return f"[web_search error: {e}]"


# ── All tools (imported by the agent in Phase 3) ─────────────────────

ALL_TOOLS = [semantic_search, bm25_keyword_search, sql_lookup, web_search]


# ── Shared formatter ─────────────────────────────────────────────────

def _format_results(results: list[dict], label: str) -> str:
    if not results:
        return f"[{label}]: No results found."

    lines = [f"[{label}] — {len(results)} results\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"Result {i} (score={r.get('score', 'N/A')}):")
        text = r['text']
        lines.append(f"  Text: {text[:300]}{'...' if len(text) > 300 else ''}")
        if r.get("source"):
            lines.append(f"  Source: {r['source']}")
        if r.get("ticker"):
            lines.append(f"  Ticker: {r['ticker']}")
        if r.get("sentiment") and r["sentiment"] != "unknown":
            lines.append(f"  Sentiment: {r['sentiment']}")
        lines.append("")

    return "\n".join(lines)