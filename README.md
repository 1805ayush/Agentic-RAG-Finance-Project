# 🔍 Agentic RAG with Dynamic Tool Routing

A finance Q&A system where an LLM agent dynamically reasons about which retrieval strategy to use, instead of blindly querying a single vector store.

**🌐 [Live Demo](https://agentic-rag-finance.streamlit.app/)** · **📂 [GitHub](https://github.com/1805ayush/Agentic-RAG-Finance-Project)**

---

## 🎯 What it does

Given a financial question, a Groq-hosted Llama 3.3 70B agent picks from four retrievers and synthesizes a cited answer:

| Tool | When the agent picks it | Backend |
|---|---|---|
| 🔮 **Semantic Search** | Conceptual / explanatory questions ("What is quantitative easing?") | ChromaDB + sentence-transformers |
| 🔑 **BM25 Keyword** | Ticker symbols, company names, technical acronyms ("AAPL P/E ratio") | rank_bm25 with finance-aware tokenizer |
| 🗄 **SQL Lookup** | Counts, aggregations, sentiment filters ("How many negative passages?") | SQLite with structured metadata |
| 🌐 **Web Search** | Live data, recent events, current prices ("Current Fed funds rate") | DuckDuckGo (`ddgs`) |

The routing decision is dynamic — driven by a tuned system prompt + tool docstrings — not hardcoded.

---

## 📊 Evaluation Results

Evaluated on a hand-curated **50-query test set** across 5 query categories using **RAGAS**:

| Metric | Score |
|---|---|
| **Routing accuracy** | **68.0%** |
| **Faithfulness** (RAGAS, against agent-retrieved context) | **0.710** |
| **Answer relevancy** (RAGAS) | **0.922** |

### Per-category routing accuracy

| Category | Expected tool | Accuracy |
|---|---|---|
| Opinion / explanatory | semantic_search | **100%** |
| Aggregation / counts | sql_lookup | **100%** |
| Market data / tickers | bm25_keyword_search | 60% |
| Sentiment | semantic_search | 50% |
| Factual / definitional | bm25_keyword_search | 30% * |

\* *The agent routes definitional questions ("What is EBITDA?") to semantic search instead of BM25. Both produce correct answers; this is an oracle-vs-reality mismatch rather than a routing failure.*

---

## 🏗 Architecture

```mermaid
flowchart TD
    Q([🔍 User Query]):::query --> Agent

    Agent[<b>🧠 LangGraph ReAct Agent</b><br/><i>Groq Llama 3.3 70B</i>]:::agent

    Agent -- "conceptual<br/>questions" --> Sem
    Agent -- "tickers<br/>acronyms" --> BM
    Agent -- "counts<br/>aggregations" --> SQL
    Agent -- "live / recent<br/>data" --> Web

    Sem[<b>🔮 Semantic Search</b><br/><sub>ChromaDB + MiniLM-L6-v2</sub>]:::tool
    BM[<b>🔑 BM25 Keyword</b><br/><sub>rank_bm25 + finance tokenizer</sub>]:::tool
    SQL[<b>🗄 SQL Lookup</b><br/><sub>SQLite + corpus_stats</sub>]:::tool
    Web[<b>🌐 Web Search</b><br/><sub>DuckDuckGo (ddgs)</sub>]:::tool

    Sem --> Loop
    BM --> Loop
    SQL --> Loop
    Web --> Loop

    Loop{Enough<br/>context?}:::decision
    Loop -- "no" --> Agent
    Loop -- "yes" --> Answer([💬 Grounded Answer<br/>+ Source Citations]):::answer

    classDef query    fill:#2a1a3a,stroke:#c084fc,stroke-width:2px,color:#e9d5ff
    classDef agent    fill:#3a2a1a,stroke:#fb923c,stroke-width:2px,color:#fde6c1
    classDef tool     fill:#1a3a5c,stroke:#60a5fa,stroke-width:2px,color:#e6edf3
    classDef decision fill:#3a1a1a,stroke:#f87171,stroke-width:2px,color:#fecaca
    classDef answer   fill:#1a3a2a,stroke:#4ade80,stroke-width:2px,color:#bbf7d0
```

The agent runs a **reason → tool → reason → ...** loop in LangGraph, calling tools in sequence (or sometimes in parallel) until it has enough context, then synthesizes a grounded, cited response.
---

## 📚 Dataset

Combined two free finance datasets into a unified corpus of **~5,000 passages**:

- **FiQA** (BeIR/fiqa) — 2,997 financial Q&A passages from Stack Exchange and earnings calls
- **Financial PhraseBank** (descartes100/enhanced-financial-phrasebank) — 2,000 sentences with positive/negative/neutral sentiment labels

Each passage is enriched with extracted ticker symbols, query types (factual / opinion / market_data / sentiment), and source attribution.

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph (ReAct pattern) |
| LLM | Groq Llama 3.3 70B (free tier, 30 RPM) |
| Semantic retrieval | ChromaDB + `sentence-transformers/all-MiniLM-L6-v2` |
| Lexical retrieval | rank_bm25 with custom finance tokenizer |
| Structured retrieval | SQLite |
| Web retrieval | DuckDuckGo (`ddgs`, free, no API key) |
| Evaluation | RAGAS (faithfulness + answer relevancy) |
| UI | Streamlit |
| Deployment | Streamlit Community Cloud (free) |

**Zero ongoing cost** — every API used is free tier.

---

## 🚀 Run Locally

### 1. Clone and install

```bash
git clone https://github.com/1805ayush/Agentic-RAG-Finance-Project.git
cd Agentic-RAG-Finance-Project

# Create environment
conda create -n agentic-rag python=3.11 -y
conda activate agentic-rag

pip install -r requirements.txt
```

### 2. Set up API keys

```bash
cp .env.example .env
# Edit .env and add:
#   GROQ_API_KEY=gsk_... (get free at console.groq.com)
```

### 3. Build the indices

```bash
python -m ingest.run_all
```

This downloads FiQA + Financial PhraseBank, embeds passages into ChromaDB, builds the BM25 index, and populates the SQLite metadata store. Takes ~5–10 minutes (most of it is embedding on CPU).

### 4. Run the app

```bash
streamlit run app.py
```

Open http://localhost:8501

### 5. (Optional) Run evaluation

```bash
EVAL_SAMPLE_SIZE=30 python -m eval.evaluate
```

Generates `eval/results.json` with routing accuracy + RAGAS scores.

---

## 📁 Project Structure

```
agentic-rag-finance/
│
├── 📱 app.py                          # Streamlit UI — main entry point
│
├── 🤖 agent/
│   ├── __init__.py
│   └── graph.py                       # LangGraph ReAct agent
│                                       #   • System prompt + routing rules
│                                       #   • Reasoning loop + state machine
│                                       #   • Public run_agent() API
│
├── 🔧 tools/
│   ├── __init__.py
│   └── retrieval_tools.py             # 4 @tool-decorated retrievers
│                                       #   • semantic_search   (ChromaDB)
│                                       #   • bm25_keyword_search  (rank_bm25)
│                                       #   • sql_lookup        (SQLite)
│                                       #   • web_search        (DuckDuckGo)
│
├── 📥 ingest/                          # Phase 1: build all indices
│   ├── __init__.py
│   ├── load_finance_data.py           # Pull FiQA + Financial PhraseBank
│   ├── embed_chroma.py                # Embed passages → ChromaDB
│   ├── index_bm25.py                  # Build BM25 (finance-aware tokenizer)
│   ├── build_sqlite.py                # Build SQLite metadata + stats
│   ├── verify_indices.py              # Smoke-test all 4 backends
│   └── run_all.py                     # Master runner — all 4 steps in order
│
├── 📊 eval/                            # Phase 4: evaluation pipeline
│   ├── __init__.py
│   ├── evaluate.py                    # RAGAS + routing accuracy + reporter
│   ├── test_queries.json              # Hand-curated 50-query test set
│   └── results.json                   # Cached eval results (sidebar metrics)
│
├── 💾 indices/                         # Generated by `python -m ingest.run_all`
│   ├── chroma_db/                     # ChromaDB persistent storage
│   │   ├── chroma.sqlite3
│   │   └── <uuid>/                    # HNSW index binaries
│   ├── bm25.pkl                       # Pickled BM25 model + records
│   └── finance_rag.db                 # SQLite with metadata + corpus_stats
│
├── 📦 data/
│   └── finance_corpus.jsonl           # Unified corpus (~5K passages)
│
├── ⚙️  .streamlit/
│   ├── config.toml                    # Dark theme + UI customization
│   └── secrets.toml.example           # Template for deployment secrets
│
├── 🔐 .env.example                     # Template for local API keys
├── 🐍 .python-version                  # Python 3.11 pin (Streamlit Cloud)
├── 📋 requirements.txt                 # Pip dependencies
├── 🚫 .gitignore                       # Excludes .env, secrets, caches
└── 📖 README.md                        # You are here
```

## 🧪 Try These Queries

Each triggers a different tool — useful for demoing the routing:

| Query | Triggers |
|---|---|
| "Explain quantitative easing simply" | 🔮 Semantic |
| "What does EBITDA stand for?" | 🔑 BM25 |
| "How many positive sentiment passages?" | 🗄 SQL |
| "Current Federal Reserve interest rate?" | 🌐 Web |
| "Why do bond prices fall when interest rates rise?" | 🔮 Semantic (with reasoning depth) |

---

## ⚠️ Known Limitations

Honest engineering trade-offs in this build:

1. **Definitional queries route to semantic instead of BM25** — Both produce correct answers, but the routing accuracy metric penalizes the choice. Reflects an oracle-vs-reality mismatch.
2. **Faithfulness scoring depends on retrieved context** — Earlier eval runs that scored against arbitrary gold passages (not what the agent actually retrieved) produced misleadingly low faithfulness scores (~0.18). Fixed by capturing the agent's actual tool outputs as RAGAS context, raising faithfulness to 0.71.
3. **Groq free tier rate limits** — 30 RPM caps eval throughput to ~50 queries every 25 minutes. Throttling logic in `eval/evaluate.py` handles this gracefully.
4. **Cold-start latency** — First query on Streamlit Cloud takes 30+ seconds while the embedding model downloads (~90MB). Subsequent queries are sub-second.
5. **Small corpus** — 5,000 passages is sufficient for routing demonstration but too small for evaluating multi-hop reasoning chains.

---

## 🎓 What I Learned

- **LangGraph state machines** are surprisingly compact for this pattern — the full ReAct loop is ~10 lines.
- **Tool docstrings drive routing more than the system prompt** — LangChain literally passes docstrings to the LLM, so prompt-engineering them is high-leverage.
- **RAGAS faithfulness is brittle** — it needs the actual retrieved context, not a "gold" passage you picked yourself. Subtle but critical for honest metrics.
- **Free tiers compose well** — Groq + DuckDuckGo + local embeddings + Streamlit Cloud = $0/month for the entire stack.

---

## 📄 License

MIT