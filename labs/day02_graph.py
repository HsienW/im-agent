import asyncio

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)

from app.agents.graph import build_graph


async def main():

    graph = build_graph()

    thread_a = {
        "configurable": {
            "thread_id": "im-thread-a",
        },
        "recursion_limit": 12,
    }

    thread_b = {
        "configurable": {
            "thread_id": "im-thread-b",
        },
        "recursion_limit": 12,
    }

    # Round 1
    print("\n=== Round 1: Tool Calling ===")

    first = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "請使用工具搜尋『登入』相關 IM 訊息，"
                        "整理已確認的待辦事項及負責人。"
                    )
                )
            ]
        },
        config=thread_a,
    )

    for message in first["messages"]:

        if isinstance(message, AIMessage):
            if message.tool_calls:
                print(
                    "[Tool Calls]",
                    message.tool_calls,
                )

        elif isinstance(message, ToolMessage):
            print(
                "[Tool Result]",
                message.content,
            )

    if not any(
        isinstance(message, ToolMessage)
        for message in first["messages"]
    ):
        raise RuntimeError(
            "Expected Tool Calling did not occur"
        )

    print(
        "[Answer]",
        first["messages"][-1].content,
    )

    # Round 2: Same Thread
    print("\n=== Round 2: Same Thread ===")

    second = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "剛才提到的登入問題，"
                        "是由誰負責排查？"
                    )
                )
            ]
        },
        config=thread_a,
    )

    print(
        "[Answer]",
        second["messages"][-1].content,
    )

    print(
        "[Thread A Messages]",
        len(second["messages"]),
    )

    # Thread isolation
    print("\n=== Thread Isolation ===")

    state_b = await graph.aget_state(thread_b)

    print(
        "[Thread B Messages]",
        len(state_b.values.get("messages", [])),
    )

    assert not state_b.values.get("messages"), (
        "Thread B should not contain Thread A history"
    )


if __name__ == "__main__":
    asyncio.run(main())
