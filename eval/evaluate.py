import os
import json
import time
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm 
from dotenv import load_dotenv

load_dotenv()

EVAL_OUTPUT_PATH = "./eval/results.json"
EVAL_SAMPLE_SIZE = int(os.getenv("EVAL_SAMPLE_SIZE", "50"))  
THROTTLE_SECONDS = 2.5

ORACLE_ROUTING = {
    "sentiment":   "semantic_search",            # PhraseBank → sentiment lookups
    "factual":     "bm25_keyword_search",   # short factoid Q's
    "opinion":     "semantic_search",       # open-ended FiQA discussions
    "market_data": "bm25_keyword_search",   # price/ratio/numeric → keywords
    "aggregation":  "sql_lookup", 
}

def load_eval_samples(data_path: str, n: int) -> list[dict]:
    """
    Use a curated hand-written test set for routing accuracy,
    plus PhraseBank records for faithfulness scoring.
    """
    # Load hand-written queries
    test_path = "./eval/test_queries.json"
    with open(test_path) as f:
        test_queries = json.load(f)

    # Load corpus for matching gold passages
    with open(data_path) as f:
        records = [json.loads(line) for line in f]

    # Quick lookup of corpus passages by query_type
    by_type = defaultdict(list)
    for r in records:
        by_type[r["query_type"]].append(r)

    # Build samples: each hand-written query paired with a relevant passage
    per_type = max(1, n // 4)   # 4 query types
    samples = []
    for qt in ["factual", "market_data", "opinion", "sentiment", "aggregation"]:
        queries = test_queries.get(qt, [])[:per_type]
        bucket  = by_type.get(qt, by_type.get("factual", []))[:len(queries)]
        if not bucket:
            # aggregation has no corpus passages — use any factual passage as filler
            bucket = by_type["factual"][:len(queries)]
        for query, record in zip(queries, bucket):
            samples.append({
                "query":        query,
                "query_id":     f"test_{qt}_{len(samples)}",
                "query_type":   qt,
                "gold_answer":  record.get("answer") or record.get("sentiment", ""),
                "gold_passage": record["passage_text"],
            })

    print(f"  Loaded {len(samples)} hand-curated samples across 4 query types")
    for qt in ["factual", "market_data", "opinion", "sentiment"]:
        used = sum(1 for s in samples if s["query_type"] == qt)
        print(f"    {qt:12s} : {used}")
    return samples

def run_eval_batch(samples: list[dict]) -> list[dict]:
    from agent.graph import run_agent

    results = []
    for sample in tqdm(samples, desc="  Running agent"):
        try:
            output = run_agent(sample["query"], verbose=False)

            # Extract actual retrieved content from tool messages
            retrieved_context = []
            for msg in output["messages"]:
                if type(msg).__name__ == "ToolMessage":
                    retrieved_context.append(msg.content[:2000])

            results.append({
                **sample,
                "predicted_answer":  output["answer"],
                "tools_used":        output["tools_used"],
                "first_tool":        output["tools_used"][0] if output["tools_used"] else None,
                "retrieved_context": retrieved_context,
            })
        except Exception as e:
            results.append({
                **sample,
                "predicted_answer":  f"[ERROR: {str(e)[:200]}]",
                "tools_used":        [],
                "first_tool":        None,
                "retrieved_context": [],
            })

        time.sleep(THROTTLE_SECONDS)

    return results

def compute_routing_accuracy(results: list[dict]) -> dict:
    """% of queries where the agent's first tool matches the oracle."""
    correct = 0
    total = 0
    per_type = defaultdict(lambda: {"correct": 0, "total": 0})

    for r in results:
        qt = r["query_type"]
        expected = ORACLE_ROUTING.get(qt)
        actual = r["first_tool"]

        per_type[qt]["total"] += 1
        total += 1
        if actual == expected:
            per_type[qt]["correct"] += 1
            correct += 1

    return {
        "overall": round(correct / total, 3) if total else 0.0,
        "per_type": {
            qt: round(v["correct"] / v["total"], 3) if v["total"] else 0.0
            for qt, v in per_type.items()
        },
    }

def compute_ragas_metrics(results: list[dict]) -> dict:
    """RAGAS with safer score extraction + per-sample fallback."""
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_groq import ChatGroq
        from langchain_huggingface import HuggingFaceEmbeddings
        from datasets import Dataset
        from ragas.run_config import RunConfig      
        import numpy as np

        llm = LangchainLLMWrapper(ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
            max_tokens=2048,         # prevent LLMDidNotFinish errors
            timeout=60,              # longer timeout for slow samples
        ))

        emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        ))

        ragas_data = {
            "question":     [r["query"] for r in results],
            "answer":       [r["predicted_answer"] for r in results],
            "contexts":     [r.get("retrieved_context") or [r["gold_passage"]] for r in results],
            "ground_truth": [str(r["gold_answer"]) for r in results],
        }
        dataset = Dataset.from_dict(ragas_data)

        scores = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=llm,
            embeddings=emb,
            raise_exceptions=False,
            run_config=RunConfig(
                max_workers=1,          # serial, not parallel
                timeout=120,
                max_retries=3,
            ),
        )

        # Robust extraction: handle both float and per-sample list returns
        def _agg(key):
            val = scores[key]
            if isinstance(val, (list, tuple)):
                # filter NaN / None then mean
                clean = [v for v in val if isinstance(v, (int, float)) and not (v != v)]
                return float(np.mean(clean)) if clean else 0.0
            try:
                return float(val)
            except (TypeError, ValueError):
                # Try one more conversion via numpy in case it's an array
                arr = np.array(val).flatten()
                arr = arr[~np.isnan(arr)] if arr.dtype.kind == 'f' else arr
                return float(arr.mean()) if len(arr) else 0.0

        return {
            "faithfulness":     round(_agg("faithfulness"), 3),
            "answer_relevancy": round(_agg("answer_relevancy"), 3),
            "method":           "ragas",
        }
    except Exception as e:
        print(f"\n  ⚠  RAGAS evaluation failed: {str(e)[:200]}")
        print("     Falling back to word-overlap heuristic.")
        return _heuristic_metrics(results)
    
def _heuristic_metrics(results: list[dict])->dict:
    faith_scores =[]
    rel_scores = []
    for r in results:
        answer  = (r["predicted_answer"] or "").lower()
        passage = (r["gold_passage"] or "").lower()
        query   = (r["query"] or "").lower()

        ans_words = set(answer.split())

        # Faithfulness ≈ overlap with retrieved context
        faith = len(ans_words & set(passage.split())) / max(len(ans_words), 1)
        faith_scores.append(min(faith, 1.0))

        # Relevance ≈ overlap with the original query
        rel = len(ans_words & set(query.split())) / max(len(set(query.split())), 1)
        rel_scores.append(min(rel, 1.0))

    return {
        "faithfulness":     round(sum(faith_scores) / max(len(faith_scores), 1), 3),
        "answer_relevancy": round(sum(rel_scores) / max(len(rel_scores), 1), 3),
        "method":           "word_overlap_heuristic",
    }   

def _tool_distribution(results: list[dict]) -> dict:
    """Count tool usage across all results."""
    dist = defaultdict(int)
    for r in results:
        for t in r["tools_used"]:
            dist[t] += 1
    return dict(dist)

def run_evaluation():
    data_path = os.getenv("DATA_PATH", "./data/finance_corpus.jsonl")

    print(f"\n{'═'*60}")
    print(f"  PHASE 4 — EVALUATION  ({EVAL_SAMPLE_SIZE} samples)")
    print(f"{'═'*60}")

    print("\n[1/4] Loading eval samples …")
    samples = load_eval_samples(data_path, EVAL_SAMPLE_SIZE)

    print(f"\n[2/4] Running agent ({THROTTLE_SECONDS}s throttle per call) …")
    print(f"      Estimated time: {len(samples) * THROTTLE_SECONDS / 60:.1f} min")
    t0 = time.time()
    results = run_eval_batch(samples)
    print(f"      Completed in {(time.time()-t0)/60:.1f} min")

    print("\n[3/4] Computing routing accuracy …")
    routing = compute_routing_accuracy(results)

    print("\n[4/4] Computing RAGAS faithfulness + relevancy …")
    ragas_metrics = compute_ragas_metrics(results)

    # ── Build final report ────────────────────────────────────────
    report = {
        "n_samples":               len(results),
        "routing_accuracy":        routing,
        "ragas_metrics":           ragas_metrics,
        "tool_usage_distribution": _tool_distribution(results),
        "oracle_mapping":          ORACLE_ROUTING,
        "per_sample_results":      results,
    }

    Path(EVAL_OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    # ── Print summary ──────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  EVALUATION RESULTS")
    print(f"{'═'*60}")
    print(f"  Samples evaluated   : {len(results)}")
    print(f"  Routing accuracy    : {routing['overall']:.1%}")
    print(f"  Faithfulness        : {ragas_metrics['faithfulness']:.3f}")
    print(f"  Answer relevancy    : {ragas_metrics['answer_relevancy']:.3f}")
    print(f"  Method              : {ragas_metrics.get('method', 'ragas')}")
    print(f"\n  Per-type routing accuracy:")
    for qt, acc in routing["per_type"].items():
        oracle = ORACLE_ROUTING.get(qt, "?")
        print(f"    {qt:12s} → {oracle:25s} : {acc:.1%}")
    print(f"\n  Tool usage distribution:")
    for tool, count in sorted(_tool_distribution(results).items(), key=lambda x: -x[1]):
        print(f"    {tool:25s} : {count} calls")
    print(f"\n  Full results → {EVAL_OUTPUT_PATH}")
    print(f"{'═'*60}\n")

    return report


if __name__ == "__main__":
    run_evaluation()
