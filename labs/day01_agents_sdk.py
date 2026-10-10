import asyncio
import json
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    function_tool,
    set_tracing_disabled,
)

from app.tools.message_search import search_messages


load_dotenv()
set_tracing_disabled(True)


class Task(BaseModel):
    title: str
    assignee: str | None = None
    source_ids: list[str]


class TaskBatch(BaseModel):
    tasks: list[Task]


@function_tool
def search_im_messages(keyword: str) -> str:
    """Search IM conversation messages by keyword."""

    results = search_messages(keyword)

    print(f"[Tool] search_im_messages: {keyword}")

    return json.dumps(
        results,
        ensure_ascii=False,
    )


async def main():
    base_url = os.getenv("LLM_BASE_URL")
    api_key = os.getenv("LLM_API_KEY")
    model_name = os.getenv("LLM_MODEL")

    if not all([base_url, api_key, model_name]):
        raise RuntimeError(
            "Missing LLM_BASE_URL, LLM_API_KEY or LLM_MODEL"
        )

    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=30.0,
        max_retries=1,
    )

    agent = Agent(
        name="IM Task Extractor",
        instructions=(
            "You extract actionable tasks from IM messages. "
            "Always call search_im_messages before answering. "
            "Use only messages returned by the tool. "
            "Never invent an assignee or message ID. "
            "If the assignee is unknown, use null. "
            "Return only valid JSON in this format: "
            '{"tasks":[{"title":"...",'
            '"assignee":null,"source_ids":["m001"]}]}'
        ),
        model=OpenAIChatCompletionsModel(
            model=model_name,
            openai_client=client,
        ),
        tools=[search_im_messages],
    )

    try:
        result = await Runner.run(
            agent,
            "搜尋登入相關訊息，整理已確認的待辦事項。",
            max_turns=5,
        )

        tasks = TaskBatch.model_validate_json(
            result.final_output
        )

        # 驗證來源 ID 確實存在於被檢索的資料中
        valid_ids = {
            message["id"]
            for message in search_messages("登入")
        }

        for task in tasks.tasks:
            if not task.source_ids or not set(
                task.source_ids
            ).issubset(valid_ids):
                raise ValueError(
                    "Task contains missing or invalid source IDs"
                )

        print("\n[Validated Tasks]")
        print(tasks.model_dump_json(indent=2))

    except ValidationError as exc:
        print("[Validation Failed]", exc)
        raise


if __name__ == "__main__":
    asyncio.run(main())
