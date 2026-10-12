import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings


ROOT_DIR = Path(__file__).resolve().parents[2]

load_dotenv(ROOT_DIR / ".env")


def create_embeddings() -> OpenAIEmbeddings:
    required = (
        "LLM_BASE_URL",
        "LLM_API_KEY",
        "EMBEDDING_MODEL",
    )

    missing = [
        name
        for name in required
        if not os.getenv(name)
    ]

    if missing:
        raise RuntimeError(
            f"Missing Embedding config: {', '.join(missing)}"
        )

    return OpenAIEmbeddings(
        model=os.environ["EMBEDDING_MODEL"],
        base_url=os.environ["LLM_BASE_URL"],
        api_key=os.environ["LLM_API_KEY"],
        check_embedding_ctx_length=False,
        chunk_size=5,
        request_timeout=20,
        max_retries=0,
    )
