
import json

import pytest

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)

from app.agents.graph import build_graph
from app.rag.ingestion import load_knowledge_chunks
from app.rag.retriever import (
    build_vector_store,
    retrieve_knowledge,
)


@pytest.fixture
def knowledge_store(tmp_path):
    """Build an offline knowledge store for testing."""

    (tmp_path / "im-policy.md").write_text(
        "# IM Group Permissions\n"
        "Users who leave a group cannot access new messages.\n"
        "Historical access depends on the retention policy.\n",
        encoding="utf-8",
    )

    chunks = load_knowledge_chunks(tmp_path)

    embeddings = DeterministicFakeEmbedding(size=64)

    return build_vector_store(chunks, embeddings)


def test_chunk_preserves_source(tmp_path):
    """Chunks must retain traceable source metadata."""

    (tmp_path / "policy.md").write_text(
        "# Approval Policy\n"
        "Creating tasks requires human approval.\n",
        encoding="utf-8",
    )

    chunks = load_knowledge_chunks(tmp_path)

    assert len(chunks) >= 1
    assert chunks[0].metadata["source"] == "policy.md"
    assert chunks[0].metadata["chunk_id"] == "policy:001"


def test_retrieval_returns_source(knowledge_store):
    """Retrieval must return source and chunk identifiers."""

    results = retrieve_knowledge(
        knowledge_store,
        "group historical message access",
        k=1,
    )

    assert len(results) == 1

    result = results[0]

    assert result["source"] == "im-policy.md"
    assert result["chunk_id"] == "im-policy:001"
    assert result["content"]


def test_retrieval_rejects_empty_query(knowledge_store):
    """Empty queries must not reach vector search."""

    with pytest.raises(ValueError):
        retrieve_knowledge(
            knowledge_store,
            "",
        )


class FakeChatModel:
    """Deterministic model for offline LangGraph tests."""

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        if isinstance(messages[-1], ToolMessage):
            return AIMessage(
                content="Answer based on retrieved knowledge."
            )

        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_knowledge",
                    "args": {
                        "query": "group historical message access"
                    },
                    "id": "knowledge-test-call",
                }
            ],
        )


@pytest.mark.asyncio
async def test_graph_executes_knowledge_tool(knowledge_store):
    """LangGraph must execute the knowledge retrieval tool."""

    graph = build_graph(
        chat_model=FakeChatModel(),
        knowledge_store=knowledge_store,
    )

    result = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content="Can users access group history?"
                )
            ]
        },
        {
            "configurable": {
                "thread_id": "test-day03-rag",
            },
            "recursion_limit": 12,
        },
    )

    tool_messages = [
        message
        for message in result["messages"]
        if isinstance(message, ToolMessage)
    ]

    assert len(tool_messages) == 1
    assert tool_messages[0].name == "search_knowledge"

    retrieved = json.loads(tool_messages[0].content)

    assert len(retrieved) == 1
    assert retrieved[0]["source"] == "im-policy.md"
    assert retrieved[0]["chunk_id"] == "im-policy:001"

    assert result["messages"][-1].content == (
        "Answer based on retrieved knowledge."
    )
