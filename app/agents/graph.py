import json

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool

from langgraph.graph import (
    StateGraph,
    MessagesState,
    START,
)
from langgraph.prebuilt import (
    ToolNode,
    tools_condition,
)
from langgraph.checkpoint.memory import InMemorySaver

from app.models.llm import create_chat_model
from app.tools.message_search import search_messages


# 1. Tool Adapter：重用 Day 1 的業務函式
@tool
def search_im_messages(keyword: str) -> str:
    """Search IM messages containing a keyword."""

    results = search_messages(keyword)

    return json.dumps(
        results,
        ensure_ascii=False,
    )


SYSTEM_PROMPT = """
You are an IM collaboration assistant.

When the user asks about IM messages:
- Search relevant messages using search_im_messages.
- Only use facts from retrieved messages.
- Do not invent assignees or deadlines.
- Reference original message IDs.
- If information is insufficient, say so.

After receiving tool results, answer the user.
"""


def build_graph(chat_model=None, checkpointer=None):

    # 2. Model + Tools
    tools = [search_im_messages]

    model = (
        chat_model
        if chat_model is not None
        else create_chat_model()
    )

    model_with_tools = model.bind_tools(tools)

    # 3. Assistant Node
    async def assistant_node(state: MessagesState):

        response = await model_with_tools.ainvoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                *state["messages"],
            ]
        )

        return {
            "messages": [response],
        }

    # 4. StateGraph
    builder = StateGraph(MessagesState)

    builder.add_node("assistant", assistant_node)
    builder.add_node("tools", ToolNode(tools))

    # 5. Entry
    builder.add_edge(START, "assistant")

    # 6. Conditional Edge
    builder.add_conditional_edges(
        "assistant",
        tools_condition,
    )

    # 7. Tool execution returns to Assistant
    builder.add_edge("tools", "assistant")

    # 8. Checkpointer
    saver = (
        checkpointer
        if checkpointer is not None
        else InMemorySaver()
    )

    return builder.compile(
        checkpointer=saver,
    )
