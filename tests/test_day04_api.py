import asyncio
import json

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk

from app.api.main import create_app


class FakeGraph:
    """Offline graph for HTTP and SSE contract tests."""

    def __init__(self):
        self.thread_ids = []

    async def ainvoke(self, values, config):
        self.thread_ids.append(
            config["configurable"]["thread_id"]
        )

        return {
            "messages": [
                AIMessage(
                    content="歷史訊息需依權限與保留政策判斷。"
                )
            ]
        }

    async def astream(
        self,
        values,
        config,
        stream_mode,
    ):
        assert stream_mode == "messages"

        self.thread_ids.append(
            config["configurable"]["thread_id"]
        )

        yield (
            AIMessageChunk(content="根據文件"),
            {"langgraph_node": "assistant"},
        )

        # Tool node data must not be exposed as SSE tokens.
        yield (
            AIMessageChunk(content="PRIVATE_TOOL_DATA"),
            {"langgraph_node": "tools"},
        )

        yield (
            AIMessageChunk(content="，需檢查權限。"),
            {"langgraph_node": "assistant"},
        )


class SlowGraph(FakeGraph):
    """Simulates slow model execution."""

    async def ainvoke(self, values, config):
        await asyncio.sleep(0.1)

        return await super().ainvoke(
            values,
            config,
        )

    async def astream(
        self,
        values,
        config,
        stream_mode,
    ):
        await asyncio.sleep(0.1)

        async for item in super().astream(
            values,
            config,
            stream_mode,
        ):
            yield item


class FailedGraph(FakeGraph):
    """Simulates an internal provider failure."""

    async def ainvoke(self, values, config):
        raise RuntimeError("private-provider-error")

    async def astream(
        self,
        values,
        config,
        stream_mode,
    ):
        raise RuntimeError("private-provider-error")
        yield  # Keep this method an async generator.


# -------------------------------------------------
# Test 1: Health
# -------------------------------------------------

def test_health():
    with TestClient(
        create_app(graph=FakeGraph())
    ) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# -------------------------------------------------
# Test 2: Normal chat response
# -------------------------------------------------

def test_chat_returns_answer():
    with TestClient(
        create_app(graph=FakeGraph())
    ) as client:
        response = client.post(
            "/chat",
            json={
                "thread_id": "test-chat",
                "message": "退出群組後還能查看歷史訊息嗎？",
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["thread_id"] == "test-chat"
    assert data["answer"] == (
        "歷史訊息需依權限與保留政策判斷。"
    )


# -------------------------------------------------
# Test 3: Thread ID propagation
# -------------------------------------------------

def test_chat_reuses_thread():
    graph = FakeGraph()

    with TestClient(
        create_app(graph=graph)
    ) as client:
        for message in ("第一輪", "第二輪"):
            response = client.post(
                "/chat",
                json={
                    "thread_id": "same-thread",
                    "message": message,
                },
            )

            assert response.status_code == 200

    assert graph.thread_ids == [
        "same-thread",
        "same-thread",
    ]


# -------------------------------------------------
# Test 4: SSE events and tool filtering
# -------------------------------------------------


def test_stream_events_and_tool_filtering():
    with TestClient(
        create_app(graph=FakeGraph())
    ) as client:
        response = client.post(
            "/stream",
            json={
                "thread_id": "test-stream",
                "message": "查詢群組權限",
            },
        )

    assert response.status_code == 200

    assert "text/event-stream" in (
        response.headers["content-type"]
    )

    assert "event: start" in response.text
    assert "event: token" in response.text
    assert "event: done" in response.text
    assert "event: error" not in response.text

    # Decode JSON payloads instead of checking raw SSE text.
    token_texts = []
    current_event = None

    for line in response.text.splitlines():
        if line.startswith("event:"):
            current_event = line.removeprefix(
                "event:"
            ).strip()

        elif line.startswith("data:"):
            if current_event != "token":
                continue

            payload = json.loads(
                line.removeprefix("data:").strip()
            )

            token_texts.append(payload["text"])

        elif not line:
            current_event = None

    # Reconstruct assistant output.
    answer = "".join(token_texts)

    assert answer == "根據文件，需檢查權限。"

    # Tool data must not leak to the client.
    assert "PRIVATE_TOOL_DATA" not in answer
    assert "PRIVATE_TOOL_DATA" not in response.text



# -------------------------------------------------
# Test 5: Request validation
# -------------------------------------------------

def test_rejects_invalid_input():
    with TestClient(
        create_app(graph=FakeGraph())
    ) as client:
        blank_message = client.post(
            "/chat",
            json={
                "message": "   ",
            },
        )

        invalid_thread = client.post(
            "/chat",
            json={
                "thread_id": "invalid/thread",
                "message": "hello",
            },
        )

    assert blank_message.status_code == 422
    assert invalid_thread.status_code == 422


# -------------------------------------------------
# Test 6: Chat timeout
# -------------------------------------------------

def test_chat_timeout():
    with TestClient(
        create_app(
            graph=SlowGraph(),
            timeout_seconds=0.01,
        )
    ) as client:
        response = client.post(
            "/chat",
            json={
                "message": "查詢",
            },
        )

    assert response.status_code == 504

    assert response.json()["error"]["code"] == (
        "AGENT_TIMEOUT"
    )


# -------------------------------------------------
# Test 7: SSE timeout
# -------------------------------------------------

def test_stream_timeout():
    with TestClient(
        create_app(
            graph=SlowGraph(),
            timeout_seconds=0.01,
        )
    ) as client:
        response = client.post(
            "/stream",
            json={
                "message": "查詢",
            },
        )

    assert response.status_code == 200

    assert "event: start" in response.text
    assert "event: error" in response.text
    assert "AGENT_TIMEOUT" in response.text

    assert "event: done" not in response.text


# -------------------------------------------------
# Test 8: Safe error responses
# -------------------------------------------------

def test_failures_hide_provider_errors():
    with TestClient(
        create_app(graph=FailedGraph())
    ) as client:
        chat_response = client.post(
            "/chat",
            json={
                "message": "查詢",
            },
        )

        stream_response = client.post(
            "/stream",
            json={
                "message": "查詢",
            },
        )

    assert chat_response.status_code == 502

    assert chat_response.json()["error"]["code"] == (
        "AGENT_FAILURE"
    )

    assert "event: error" in stream_response.text
    assert "AGENT_FAILURE" in stream_response.text
    assert "event: done" not in stream_response.text

    assert "private-provider-error" not in (
        chat_response.text
    )

    assert "private-provider-error" not in (
        stream_response.text
    )
