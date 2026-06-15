import os
from typing import Annotated, TypedDict, Literal
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage,HumanMessage,SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from tools.retrieval_tools import ALL_TOOLS
load_dotenv()
GROQ_MODEL = os.getenv("GROQ_MODEL")


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]
    tool_calls_made:list[str]

# ── System prompt: where routing intelligence lives ──────────────────

SYSTEM_PROMPT = """You are an intelligent finance question-answering agent with four retrieval tools.
Your job is to answer financial questions accurately by retrieving relevant information first.

## Tool Selection Guide

Use **semantic_search** when:
- The question is conceptual or theoretical ("What is quantitative easing?")
- The query is open-ended or asks for explanations
- Meaning matters more than exact wording
- The question is about financial mechanisms or theory

Use **bm25_keyword_search** when:
- The query contains ticker symbols (AAPL, NVDA, TSLA, BRK.B)
- Specific company or institution names appear
- Technical financial terms or acronyms (EBITDA, FOMC, ROIC, IPO, P/E)
- Exact keyword matches are important

Use **sql_lookup** when:
- The question requires counts, aggregations, or statistics
- Filtering by sentiment (positive/negative/neutral) is needed
- Filtering by ticker, query type, or source
- "How many..." or "Show me all..." structured queries

Use **web_search** when:
- The topic involves recent events (post-2023 — Fed decisions, earnings, etc.)
- Live data is needed (current prices, today's rates, breaking news)
- Corpus results appear outdated or insufficient

## Rules
1. ALWAYS retrieve before answering — never answer from memory alone.
2. Choose the SINGLE BEST tool for the query. Do NOT call multiple tools in parallel by default.
3. Only call additional tools if the first tool's results are clearly insufficient.
4. Prefer ONE tool call when a single retrieval method matches the query type.
5. After retrieval, synthesize a grounded answer with citations.
6. If tools return no useful results, say so honestly rather than calling more tools.

## Decision Heuristic
Before calling tools, ask yourself:
- Does the query mention a specific ticker, company name, or technical term? → bm25_keyword_search
- Does the query ask "how many", "count", or filter by sentiment/category? → sql_lookup
- Does the query reference recent events, current prices, or live data? → web_search
- Otherwise (conceptual, definitional, explanatory questions) → semantic_search

Pick ONE. Only escalate to additional tools if the first result is empty or off-topic.

## Answer Format
- Clear, direct answer to the user's question
- Inline citations like [Source: URL] or [Source: FiQA / PhraseBank corpus]
- Brief note at the end on which retrieval methods were used
"""

def get_llm():
    return ChatGroq(model = GROQ_MODEL,api_key = os.getenv("GROQ_API_KEY"),
                    temperature = 0,max_retries=2).bind_tools(ALL_TOOLS)

def reasoning_node(state: AgentState)->dict:
    llm = get_llm()
    messages = [SystemMessage(content=SYSTEM_PROMPT)]+state["messages"]
    response = llm.invoke(messages)
    return {"messages":[response]}

def should_continue(state: AgentState)-> Literal["tools","end"]:
    last = state["messages"][-1]
    if hasattr(last,"tool_calls") and last.tool_calls:
        return "tools"
    return "end"

def build_graph():
    tool_node = ToolNode(ALL_TOOLS)
    graph = StateGraph(AgentState)

    graph.add_node("reason",reasoning_node)
    graph.add_node("tools",tool_node)

    graph.set_entry_point("reason")

    graph.add_conditional_edges("reason",should_continue,{"tools":"tools","end":END})

    graph.add_edge("tools","reason")
    return graph.compile()

def run_agent(query:str,verbose: bool = True)->dict:
    app = build_graph()
    initial_state = {
        "messages":[HumanMessage(content = query)],
        "tool_calls_made":[]
    }
    if verbose:
        print(f"Query: {query}")
        print("─" * 60)

    final_state = None
    for step in app.stream(initial_state,stream_mode="values"):
        final_state = step
        if verbose:
            last_msg = step["messages"][-1]
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                tools = [tc["name"] for tc in last_msg.tool_calls]
                print(f"  🔧 Calling tools: {tools}")
            elif type(last_msg).__name__ == "ToolMessage":
                print(f"Tool result received")
            else:
                print(f"Final answer generated")
    
    tools_used = []
    for msg in final_state["messages"]:
        if hasattr(msg,"tool_calls") and msg.tool_calls:
            tools_used.extend(tc["name"] for tc in msg.tool_calls)

    answer = final_state["messages"][-1].content

    if verbose:
        print("\n" + "=" * 60)
        print(f"ANSWER:\n{answer}")
        print(f"\nTools used: {tools_used}")
        print("=" * 60 + "\n")

    return {
        "query": query,
        "answer": answer,
        "tools_used": tools_used,
        "messages": final_state["messages"],
    }

if __name__ == "__main__":
    run_agent("What causes bond prices to fall when interest rates rise?")


