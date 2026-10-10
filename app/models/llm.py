import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")


def create_chat_model() -> ChatOpenAI:
    required = (
        "LLM_MODEL",
        "LLM_BASE_URL",
        "LLM_API_KEY",
    )

    missing = [
        name for name in required
        if not os.getenv(name)
    ]

    if missing:
        raise RuntimeError(
            f"Missing LLM config: {', '.join(missing)}"
        )

    return ChatOpenAI(
        model=os.environ["LLM_MODEL"],
        base_url=os.environ["LLM_BASE_URL"],
        api_key=os.environ["LLM_API_KEY"],
        temperature=0,
        timeout=30,
        max_retries=1,
        use_responses_api=False,
    )
