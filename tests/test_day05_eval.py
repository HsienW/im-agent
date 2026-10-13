
import json
from pathlib import Path

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)

from app.eval.scoring import evaluate_case, inspect_run


CASE = {
    "id": "rag-group-history",
    "expected_tool": "search_knowledge",
    "expected_source": "im-permissions:001",
    "required_terms": ["歷史訊息", "權限"],
    "forbidden_phrases": [
        "即喪失歷史訊息存取權限"
    ],
}


def make_messages(answer=None):
    if answer is None:
        answer = (
            "歷史訊息是否可見取決於權限與保留政策。"
            "來源：im-permissions:001"
        )

    return [
        HumanMessage(content="查詢群組歷史訊息權限"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_knowledge",
                    "args": {"query": "歷史訊息權限"},
                    "id": "call-1",
                }
            ],
            usage_metadata={
                "input_tokens": 20,
                "output_tokens": 4,
                "total_tokens": 24,
            },
        ),
        ToolMessage(
            name="search_knowledge",
            tool_call_id="call-1",
            content=json.dumps(
                [
                    {
                        "source": "im-permissions.md",
                        "chunk_id": "im-permissions:001",
                        "content": "歷史訊息依權限與保留政策判斷。",
                    }
                ],
                ensure_ascii=False,
            ),
        ),
        AIMessage(
            content=answer,
            usage_metadata={
                "input_tokens": 35,
                "output_tokens": 15,
                "total_tokens": 50,
            },
        ),
    ]


def test_inspect_run_records_execution_evidence():
    observed = inspect_run(make_messages())

    assert observed["called_tools"] == ["search_knowledge"]
    assert "im-permissions:001" in observed["retrieved_sources"]

    assert observed["token_usage"] == {
        "available": True,
        "input": 55,
        "output": 19,
    }

    assert [step["type"] for step in observed["trace"]] == [
        "human",
        "assistant",
        "tool",
        "assistant",
    ]


def test_valid_grounded_answer_passes():
    observed = inspect_run(make_messages())

    score = evaluate_case(CASE, observed)

    assert score["passed"] is True
    assert all(score["checks"].values())


def test_missing_citation_fails():
    observed = inspect_run(
        make_messages(
            "歷史訊息是否可見取決於權限與保留政策。"
        )
    )

    score = evaluate_case(CASE, observed)

    assert score["passed"] is False
    assert score["checks"]["source_cited"] is False


def test_forbidden_claim_fails():
    observed = inspect_run(
        make_messages(
            "退出群組後即喪失歷史訊息存取權限。"
            "來源：im-permissions:001"
        )
    )

    score = evaluate_case(CASE, observed)

    assert score["passed"] is False
    assert score["checks"]["forbidden_phrases"] is False


def test_golden_dataset_contract():
    root = Path(__file__).resolve().parents[1]

    cases = json.loads(
        (
            root / "data" / "eval" / "golden_cases.json"
        ).read_text(encoding="utf-8")
    )

    assert len(cases) == 5

    identifiers = [case["id"] for case in cases]

    assert len(identifiers) == len(set(identifiers))

    required_fields = {
        "id",
        "question",
        "expected_tool",
        "expected_source",
        "required_terms",
        "forbidden_phrases",
    }

    for case in cases:
        assert required_fields <= case.keys()
        assert case["question"].strip()
        assert case["expected_source"]
