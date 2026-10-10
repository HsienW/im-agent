import json

import pytest

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)

from app.agents.graph import build_graph


class FakeChatModel:
    """Deterministic model for offline graph tests."""

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        if isinstance(messages[-1], ToolMessage):
            return AIMessage(
                content="Bob 負責排查登入 API 500 錯誤。"
            )

        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_im_messages",
                    "args": {"keyword": "登入"},
                    "id": "test-call-1",
                }
            ],
        )


@pytest.mark.asyncio
async def test_graph_executes_tool():
    graph = build_graph(chat_model=FakeChatModel())

    config = {
        "configurable": {
            "thread_id": "test-tool",
        }
    }

    result = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(content="搜尋登入問題")
            ]
        },
        config,
    )

    tool_results = [
        message
        for message in result["messages"]
        if isinstance(message, ToolMessage)
    ]

    assert len(tool_results) == 1

    data = json.loads(tool_results[0].content)

    assert data[0]["id"] == "m001"
    assert "Bob" in result["messages"][-1].content


@pytest.mark.asyncio
async def test_checkpoint_preserves_messages():
    graph = build_graph(chat_model=FakeChatModel())

    config = {
        "configurable": {
            "thread_id": "test-memory",
        }
    }

    first = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(content="搜尋登入問題")
            ]
        },
        config,
    )

    second = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(content="剛才是誰負責？")
            ]
        },
        config,
    )

    assert len(second["messages"]) > len(first["messages"])


@pytest.mark.asyncio
async def test_thread_isolation():
    graph = build_graph(chat_model=FakeChatModel())

    await graph.ainvoke(
        {
            "messages": [
                HumanMessage(content="搜尋登入")
            ]
        },
        {
            "configurable": {
                "thread_id": "thread-a",
            }
        },
    )

    state_b = await graph.aget_state(
        {
            "configurable": {
                "thread_id": "thread-b",
            }
        }
    )

    assert not state_b.values.get("messages")
