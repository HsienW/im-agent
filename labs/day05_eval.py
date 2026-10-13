
import asyncio
import json
import os
import time
from pathlib import Path
from uuid import uuid4

from langchain_core.messages import HumanMessage

from app.agents.graph import build_graph
from app.eval.scoring import evaluate_case, inspect_run
from app.rag.embeddings import create_embeddings
from app.rag.ingestion import load_knowledge_chunks
from app.rag.retriever import build_vector_store


ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT_DIR / "data" / "eval" / "golden_cases.json"
REPORT_PATH = ROOT_DIR / "reports" / "day05_eval.json"

CASE_TIMEOUT_SECONDS = 60


def estimate_chat_cost(token_usage: dict) -> float | None:
    """Optional chat-token cost estimate in USD."""

    if not token_usage["available"]:
        return None

    input_rate = os.getenv(
        "EVAL_CHAT_INPUT_USD_PER_MILLION"
    )
    output_rate = os.getenv(
        "EVAL_CHAT_OUTPUT_USD_PER_MILLION"
    )

    if not input_rate or not output_rate:
        return None

    return round(
        (
            token_usage["input"] * float(input_rate)
            + token_usage["output"] * float(output_rate)
        ) / 1_000_000,
        8,
    )


async def main():
    cases = json.loads(
        DATASET_PATH.read_text(encoding="utf-8-sig")
    )

    print("[Dataset]", len(cases), "cases")

    chunks = load_knowledge_chunks(
        ROOT_DIR / "data" / "knowledge"
    )

    store = await asyncio.to_thread(
        build_vector_store,
        chunks,
        create_embeddings(),
    )

    graph = build_graph(knowledge_store=store)

    results = []

    for case in cases:
        case_id = case["id"]
        thread_id = f"eval-{uuid4().hex}"
        started_at = time.perf_counter()

        print(f"\n[RUN] {case_id}")

        try:
            result = await asyncio.wait_for(
                graph.ainvoke(
                    {
                        "messages": [
                            HumanMessage(
                                content=case["question"]
                            )
                        ]
                    },
                    config={
                        "configurable": {
                            "thread_id": thread_id,
                        },
                        "recursion_limit": 12,
                    },
                ),
                timeout=CASE_TIMEOUT_SECONDS,
            )

            observed = inspect_run(result["messages"])
            score = evaluate_case(case, observed)
            error_type = None

        except Exception as exc:
            # Do not expose provider credentials or raw exceptions.
            error_type = type(exc).__name__

            observed = {
                "answer": "",
                "called_tools": [],
                "retrieved_sources": [],
                "trace": [],
                "token_usage": {
                    "available": False,
                    "input": None,
                    "output": None,
                },
            }

            score = evaluate_case(case, observed)

        duration_ms = round(
            (time.perf_counter() - started_at) * 1000
        )

        cost = estimate_chat_cost(observed["token_usage"])

        record = {
            "case_id": case_id,
            "thread_id": thread_id,
            "passed": score["passed"],
            "checks": score["checks"],
            "missing_terms": score["missing_terms"],
            "found_forbidden": score["found_forbidden"],
            "called_tools": observed["called_tools"],
            "retrieved_sources": observed["retrieved_sources"],
            "answer": observed["answer"],
            "trace": observed["trace"],
            "duration_ms": duration_ms,
            "token_usage": observed["token_usage"],
            "estimated_chat_cost_usd": cost,
            "error_type": error_type,
        }

        results.append(record)

        print(
            "[RESULT]",
            "PASS" if record["passed"] else "FAIL",
            f"{duration_ms}ms",
        )

        if not record["passed"]:
            print("[CHECKS]", record["checks"])
            if error_type:
                print("[ERROR TYPE]", error_type)

    passed = sum(record["passed"] for record in results)

    report = {
        "summary": {
            "total": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "pass_rate": (
                round(passed / len(results), 3)
                if results else 0
            ),
            "total_duration_ms": sum(
                item["duration_ms"]
                for item in results
            ),
        },
        "cases": results,
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== Day 5 Eval Summary ===")
    print(json.dumps(
        report["summary"],
        ensure_ascii=False,
        indent=2,
    ))

    print("[Report]", REPORT_PATH)

    if passed != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
