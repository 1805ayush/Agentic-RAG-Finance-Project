import os
import json
import time

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="Agentic RAG — Finance",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Custom CSS ───────────────────────────────────────────────────────
# Dark terminal-ish theme. Each tool gets its own colored badge.

st.markdown("""
<style>
    .stApp { background: #0f1117; }
    .main  { background: #0f1117; }

    /* Tool badges — one color per retriever */
    .tool-badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        margin: 2px 4px 2px 0;
        font-family: monospace;
    }
    .badge-semantic { background:#1a3a5c; color:#60a5fa; border:1px solid #1e4a7a; }
    .badge-bm25     { background:#1a3a2a; color:#4ade80; border:1px solid #1a4a2a; }
    .badge-sql      { background:#3a2a1a; color:#fb923c; border:1px solid #4a3a1a; }
    .badge-web      { background:#2a1a3a; color:#c084fc; border:1px solid #3a1a4a; }

    .reasoning-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px 16px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12.5px;
        color: #8b949e;
        white-space: pre-wrap;
    }

    .answer-box {
        background: #161b22;
        border-left: 3px solid #60a5fa;
        padding: 16px 20px;
        border-radius: 0 8px 8px 0;
        color: #e6edf3;
        line-height: 1.65;
    }
</style>
""", unsafe_allow_html=True)

# ── Tool badge helper ────────────────────────────────────────────────

TOOL_STYLES = {
    "semantic_search":     ("badge-semantic", "🔮 Semantic"),
    "bm25_keyword_search": ("badge-bm25",     "🔑 BM25"),
    "sql_lookup":          ("badge-sql",      "🗄 SQL"),
    "web_search":          ("badge-web",      "🌐 Web"),
}

def tool_badge(tool_name: str) -> str:
    """Render a tool name as a colored HTML pill."""
    cls, label = TOOL_STYLES.get(tool_name, ("badge-semantic", tool_name))
    return f'<span class="tool-badge {cls}">{label}</span>'

# ── Sidebar ──────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙ Config")

    model_choice = st.selectbox(
        "Groq model",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        index=0,
        help="70B = better reasoning. 8B = faster, lower rate limit.",
    )
    os.environ["GROQ_MODEL"] = model_choice

    show_trace = st.toggle("Show reasoning trace", value=True)
    show_citations = st.toggle("Show source citations", value=True)

    st.markdown("---")
    st.markdown("### 🔧 Tools")
    for tool_name, (cls, label) in TOOL_STYLES.items():
        st.markdown(f'<span class="tool-badge {cls}">{label}</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Evaluation")

    # Show last eval results if present
    eval_path = "./eval/results.json"
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            report = json.load(f)
        st.metric("Routing accuracy", f"{report['routing_accuracy']['overall']:.1%}")
        st.metric("Faithfulness",     f"{report['ragas_metrics']['faithfulness']:.3f}")
        st.metric("Answer relevancy", f"{report['ragas_metrics']['answer_relevancy']:.3f}")
        st.caption(f"Based on {report['n_samples']} samples")
    else:
        st.caption("Run `python -m eval.evaluate` to see metrics here.")

    st.markdown("---")
    st.caption("LangGraph · ChromaDB · BM25 · SQLite · Groq Llama 3.3")
# ── Header ───────────────────────────────────────────────────────────

st.markdown("# 🔍 Agentic RAG — Finance")
st.markdown("*Dynamic tool routing · Grounded answers · Source citations*")
st.markdown("---")

# ── Example query buttons ────────────────────────────────────────────

EXAMPLES = [
    "What causes bond prices to fall when interest rates rise?",
    "What is the AAPL P/E ratio outlook?",
    "How many positive sentiment passages are in the corpus?",
    "Current Federal Reserve interest rate decision?",
    "Explain quantitative easing simply",
]

st.markdown("**Try an example:**")
cols = st.columns(len(EXAMPLES))
for i, (col, ex) in enumerate(zip(cols, EXAMPLES)):
    label = ex if len(ex) < 32 else ex[:32].rstrip() + "…"
    if col.button(label, key=f"ex_{i}", use_container_width=True):
        st.session_state["prefill"] = ex

# ── Chat history state ───────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state["history"] = []


def render_turn(turn: dict):
    """Render one completed agent turn (user query + answer + trace)."""

    # Tools used as colored badges
    badges = " ".join(tool_badge(t) for t in turn.get("tools_used", []))
    if badges:
        st.markdown(f"**Tools used:** {badges}", unsafe_allow_html=True)

    # Expandable reasoning trace
    if show_trace and turn.get("trace"):
        with st.expander("🧠 Reasoning trace", expanded=False):
            st.markdown(
                f'<div class="reasoning-box">{turn["trace"]}</div>',
                unsafe_allow_html=True,
            )

    # The actual answer
    st.markdown(
        f'<div class="answer-box">{turn["answer"]}</div>',
        unsafe_allow_html=True,
    )

    # Source URLs (if any)
    if show_citations and turn.get("sources"):
        with st.expander(f"📚 Sources ({len(turn['sources'])})", expanded=False):
            for src in turn["sources"]:
                st.markdown(f"- {src}")

    # Latency
    if turn.get("latency"):
        st.caption(f"⏱ {turn['latency']:.1f}s")


# Render past turns
for turn in st.session_state["history"]:
    with st.chat_message("user"):
        st.write(turn["query"])
    with st.chat_message("assistant", avatar="🔍"):
        render_turn(turn)

# ── Chat input + processing ──────────────────────────────────────────

prefill = st.session_state.pop("prefill", "")
query = st.chat_input("Ask a finance question…") or prefill

if query:
    # Echo the user's query immediately
    with st.chat_message("user"):
        st.write(query)

    # Process and render the agent's response
    with st.chat_message("assistant", avatar="🔍"):
        status = st.empty()
        status.markdown("*⚙ Routing query to retrieval tools…*")

        t0 = time.time()
        trace_lines = []
        tools_used = []
        sources = []

        try:
            from agent.graph import build_graph
            from langchain_core.messages import HumanMessage

            app = build_graph()
            initial_state = {
                "messages": [HumanMessage(content=query)],
                "tool_calls_made": [],
            }

            final_state = None
            for step in app.stream(initial_state, stream_mode="values"):
                final_state = step
                last = step["messages"][-1]

                # Tool call decisions
                if hasattr(last, "tool_calls") and last.tool_calls:
                    for tc in last.tool_calls:
                        tools_used.append(tc["name"])
                        trace_lines.append(
                            f"→ Calling {tc['name']}("
                            f"{tc['args'].get('query', '')[:80]})"
                        )
                    # Live status update — colored badge while waiting
                    badges = " ".join(tool_badge(tc["name"]) for tc in last.tool_calls)
                    status.markdown(
                        f'<em>Retrieving from {badges}…</em>',
                        unsafe_allow_html=True,
                    )

                # Tool result messages
                elif type(last).__name__ == "ToolMessage":
                    snippet = (last.content or "")[:160].replace("\n", " ")
                    trace_lines.append(f"  ↳ Got: {snippet}…")
                    # Extract URL-looking sources for citation panel
                    for line in (last.content or "").split("\n"):
                        if "Source: http" in line:
                            url = line.split("Source:", 1)[1].strip()
                            if url and url not in sources:
                                sources.append(url)

            status.empty()
            answer = final_state["messages"][-1].content
            latency = time.time() - t0

            turn = {
                "query":      query,
                "answer":     answer,
                "tools_used": tools_used,
                "trace":      "\n".join(trace_lines),
                "sources":    sources,
                "latency":    latency,
            }
            render_turn(turn)
            st.session_state["history"].append(turn)

        except Exception as e:
            status.empty()
            st.error(f"❌ Error: {e}")
            st.info("Tip: ensure `python -m ingest.run_all` has completed and your "
                    "`.env` has a valid `GROQ_API_KEY`.")
            
