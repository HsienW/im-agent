
import asyncio
import logging
from contextlib import aclosing, asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from app.agents.graph import build_graph
from app.api.schemas import ChatRequest, ChatResponse
from app.rag.embeddings import create_embeddings
from app.rag.ingestion import load_knowledge_chunks
from app.rag.retriever import build_vector_store
from collections.abc import AsyncIterator


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

AGENT_TIMEOUT_SECONDS = 45

class UTF8JSONResponse(JSONResponse):
    media_type = "application/json; charset=utf-8"


def build_production_graph():
    """Build the shared agent and knowledge index once at startup."""

    chunks = load_knowledge_chunks(
        PROJECT_ROOT / "data" / "knowledge"
    )

    vector_store = build_vector_store(
        chunks,
        create_embeddings(),
    )

    return build_graph(
        knowledge_store=vector_store,
    )


def graph_config(thread_id: str) -> dict:
    return {
        "configurable": {
            "thread_id": thread_id,
        },
        "recursion_limit": 12,
    }


def api_error(
    status_code: int,
    code: str,
    message: str,
) -> UTF8JSONResponse:
    return UTF8JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
            }
        },
    )


def create_app(
    graph=None,
    timeout_seconds: float = AGENT_TIMEOUT_SECONDS,
) -> FastAPI:

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if graph is not None:
            application.state.graph = graph
        else:
            application.state.graph = await asyncio.to_thread(
                build_production_graph
            )

        yield

    application = FastAPI(
        title="IM Python Agent API",
        version="0.1.0",
        lifespan=lifespan,
    )

    @application.get("/health")
    async def health():
        return {"status": "ok"}

    @application.post(
        "/chat",
        response_model=ChatResponse,
        response_class=UTF8JSONResponse,
    )
    async def chat(payload: ChatRequest, request: Request):
        thread_id = payload.thread_id or uuid4().hex

        try:
            result = await asyncio.wait_for(
                request.app.state.graph.ainvoke(
                    {
                        "messages": [
                            HumanMessage(
                                content=payload.message
                            )
                        ]
                    },
                    config=graph_config(thread_id),
                ),
                timeout=timeout_seconds,
            )

            final_message = result["messages"][-1]

            if (
                not isinstance(final_message, AIMessage)
                or not isinstance(final_message.content, str)
                or not final_message.content.strip()
            ):
                raise ValueError("Missing final assistant answer")

            return ChatResponse(
                thread_id=thread_id,
                answer=final_message.content,
            )

        except TimeoutError:
            return api_error(
                504,
                "AGENT_TIMEOUT",
                "Agent request timed out.",
            )

        except Exception:
            logger.exception("Agent request failed")
            return api_error(
                502,
                "AGENT_FAILURE",
                "Agent execution failed.",
            )


    @application.post(
        "/stream",
        response_class=EventSourceResponse,
    )
    async def stream_chat(
        payload: ChatRequest,
        request: Request,
    ) -> AsyncIterator[ServerSentEvent]:

        thread_id = payload.thread_id or uuid4().hex

        # Yield SSE events directly from the route.
        yield ServerSentEvent(
            event="start",
            data={"thread_id": thread_id},
        )

        emitted_text = False

        try:
            async with asyncio.timeout(timeout_seconds):

                stream = request.app.state.graph.astream(
                    {
                        "messages": [
                            HumanMessage(
                                content=payload.message
                            )
                        ]
                    },
                    config=graph_config(thread_id),
                    stream_mode="messages",
                )

                async with aclosing(stream):
                    async for chunk, metadata in stream:

                        if await request.is_disconnected():
                            return

                        if metadata.get("langgraph_node") != "assistant":
                            continue

                        if not isinstance(chunk, AIMessageChunk):
                            continue

                        # Do not expose tool call arguments.
                        if chunk.tool_call_chunks:
                            continue

                        if not isinstance(chunk.content, str):
                            continue

                        if not chunk.content:
                            continue

                        emitted_text = True

                        yield ServerSentEvent(
                            event="token",
                            data={"text": chunk.content},
                        )

            if not emitted_text:
                raise ValueError("No assistant text was streamed")

            yield ServerSentEvent(
                event="done",
                data={"thread_id": thread_id},
            )

        except asyncio.CancelledError:
            raise

        except TimeoutError:
            yield ServerSentEvent(
                event="error",
                data={
                    "code": "AGENT_TIMEOUT",
                    "message": "Agent request timed out.",
                },
            )

        except Exception:
            logger.exception("Agent stream failed")

            yield ServerSentEvent(
                event="error",
                data={
                    "code": "AGENT_FAILURE",
                    "message": "Agent streaming failed.",
                },
            )


    return application


app = create_app()
