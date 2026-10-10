import json
from pathlib import Path

DATA_FILE = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "messages.json"
)


def search_messages(keyword: str) -> list[dict]:
    if not keyword.strip():
        raise ValueError("Search keyword cannot be empty")

    messages = json.loads(
        DATA_FILE.read_text(encoding="utf-8-sig")
    )

    return [
        message
        for message in messages
        if keyword.casefold() in message["text"].casefold()
    ][:10]
