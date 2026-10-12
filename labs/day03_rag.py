import asyncio
import json
import sys
from pathlib import Path

from langchain_core.embeddings import (
    DeterministicFakeEmbedding,
)
from langchain_core.messages import (
    HumanMessage,
    ToolMessage,
)

from app.rag.ingestion import load_knowledge_chunks
from app.rag.embeddings import create_embeddings
from app.rag.retriever import (
    build_vector_store,
    retrieve_knowledge,
)


ROOT_DIR = Path(__file__).resolve().parents[1]

QUESTION = "退出群組後，還能查看以前的歷史訊息嗎？"


async def main():

    offline = "--offline" in sys.argv[1:]

    chunks = load_knowledge_chunks(
        ROOT_DIR / "data" / "knowledge"
    )

    print("[Knowledge Chunks]", len(chunks))

    if offline:
        embeddings = DeterministicFakeEmbedding(
            size=64
        )
        print("[Embedding] FAKE - offline test")
    else:
        embeddings = create_embeddings()
        print("[Embedding] LIVE")

    store = build_vector_store(
        chunks,
        embeddings,
    )

    results = retrieve_knowledge(
        store,
        QUESTION,
        k=3,
    )

    print("\n=== Retrieval Results ===")

    for item in results:
        print("Source:", item["source"])
        print("Chunk:", item["chunk_id"])
        print("Content:", item["content"][:200])
        print("---")

    if offline:
        print("\n[Offline Retrieval Completed]")
        return

    # Step 5 完成後，才啟用以下部分
    from app.agents.graph import build_graph

    graph = build_graph(
        knowledge_store=store,
    )

    response = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "請使用 search_knowledge 工具查詢："
                        + QUESTION
                        + " 請依據知識文件回答，"
                        "附上來源檔名與 Chunk ID。"
                    )
                )
            ]
        },
        {
            "configurable": {
                "thread_id": "day03-knowledge",
            },
            "recursion_limit": 12,
        },
    )

    tool_messages = [
        message
        for message in response["messages"]
        if (
            isinstance(message, ToolMessage)
            and message.name == "search_knowledge"
        )
    ]

    if not tool_messages:
        raise RuntimeError(
            "Knowledge tool was not called"
        )

    verified_sources = set()

    for message in tool_messages:
        for item in json.loads(message.content):
            verified_sources.add(
                (item["source"], item["chunk_id"])
            )

    print("\n=== Verified Retrieved Sources ===")

    for source, chunk_id in sorted(verified_sources):
        print(source, chunk_id)

    print("\n=== Agent Answer ===")
    print(response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
