
import asyncio
import json

import httpx


API_URL = "http://127.0.0.1:8000/stream"


async def main():
    payload = {
        "thread_id": "day04-stream-001",
        "message": (
            "請搜尋知識庫，說明退出群組後"
            "歷史訊息的存取權限，並引用來源。"
        ),
    }

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(60.0)
    ) as client:

        async with client.stream(
            "POST",
            API_URL,
            json=payload,
        ) as response:

            response.raise_for_status()

            current_event = None
            completed = False

            async for line in response.aiter_lines():

                if not line:
                    current_event = None
                    continue

                if line.startswith("event:"):
                    current_event = line[6:].strip()
                    continue

                if not line.startswith("data:"):
                    continue

                data = json.loads(line[5:].strip())

                if current_event == "start":
                    print(
                        "[START]",
                        data["thread_id"],
                    )

                elif current_event == "token":
                    print(
                        data["text"],
                        end="",
                        flush=True,
                    )

                elif current_event == "done":
                    completed = True
                    print(
                        "\n[DONE]",
                        data["thread_id"],
                    )
                    break

                elif current_event == "error":
                    raise RuntimeError(
                        f"SSE error: {data}"
                    )

            if not completed:
                raise RuntimeError(
                    "Stream ended without a done event"
                )


if __name__ == "__main__":
    asyncio.run(main())
