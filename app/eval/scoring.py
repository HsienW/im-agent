import json

from langchain_core.messages import AIMessage, ToolMessage


def inspect_run(messages: list) -> dict:
    """Extract execution evidence from LangGraph messages."""

    called_tools = []
    retrieved_sources = set()
    trace = []

    input_tokens = 0
    output_tokens = 0
    usage_available = False
    answer = ""

    for message in messages:
        if isinstance(message, AIMessage):
            names = [
                call["name"]
                for call in message.tool_calls
            ]

            called_tools.extend(names)

            trace.append({
                "type": "assistant",
                "tool_calls": names,
                "tool_args": [
                    {
                        "name": call["name"],
                        "args": call.get("args", {}),
                    }
                    for call in message.tool_calls
                ],
            })

            if (
                not names
                and isinstance(message.content, str)
                and message.content.strip()
            ):
                answer = message.content

            usage = message.usage_metadata

            if not usage:
                token_usage = message.response_metadata.get(
                    "token_usage", {}
                )

                if token_usage:
                    usage = {
                        "input_tokens": token_usage.get(
                            "prompt_tokens", 0
                        ),
                        "output_tokens": token_usage.get(
                            "completion_tokens", 0
                        ),
                    }

            if usage:
                usage_available = True

                input_tokens += (
                    usage.get("input_tokens", 0) or 0
                )

                output_tokens += (
                    usage.get("output_tokens", 0) or 0
                )

        elif isinstance(message, ToolMessage):
            trace.append({
                "type": "tool",
                "name": message.name,
            })

            try:
                records = json.loads(message.content)
            except (TypeError, ValueError):
                continue

            if not isinstance(records, list):
                continue

            for record in records:
                if not isinstance(record, dict):
                    continue

                for field in ("id", "chunk_id"):
                    value = record.get(field)

                    if isinstance(value, str):
                        retrieved_sources.add(value)

        else:
            trace.append({
                "type": message.type,
            })

    return {
        "answer": answer,
        "called_tools": called_tools,
        "retrieved_sources": sorted(retrieved_sources),
        "trace": trace,
        "token_usage": {
            "available": usage_available,
            "input": (
                input_tokens if usage_available else None
            ),
            "output": (
                output_tokens if usage_available else None
            ),
        },
    }


def evaluate_case(case: dict, observed: dict) -> dict:
    """Evaluate tool, source, citation, and answer contracts."""

    answer = observed["answer"]
    answer_folded = answer.casefold()

    expected_source = case["expected_source"]

    missing_terms = [
        term
        for term in case["required_terms"]
        if term.casefold() not in answer_folded
    ]

    found_forbidden = [
        phrase
        for phrase in case["forbidden_phrases"]
        if phrase.casefold() in answer_folded
    ]

    checks = {
        "non_empty_answer": bool(answer.strip()),

        "correct_tool": (
            case["expected_tool"]
            in observed["called_tools"]
        ),

        "source_retrieved": (
            expected_source
            in observed["retrieved_sources"]
        ),

        "source_cited": expected_source in answer,

        "required_terms": not missing_terms,

        "forbidden_phrases": not found_forbidden,
    }

    return {
        "passed": all(checks.values()),
        "checks": checks,
        "missing_terms": missing_terms,
        "found_forbidden": found_forbidden,
    }
