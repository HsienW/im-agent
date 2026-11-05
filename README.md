# IM Agent Python

An IM collaboration assistant agent built with LangGraph and FastAPI, covering tool calling, RAG, SSE streaming, and offline evaluation.

> Learning project: all data is synthetic and storage is in-memory. Not a production system.

## Features

- **Tool-calling agent**: a LangGraph `StateGraph` loop between the assistant and a `ToolNode`, with conversation state keyed by `thread_id`.
- **IM message search**: keyword matching over synthetic IM messages; retries once with a shorter keyword when nothing matches, then states clearly that no result was found.
- **Knowledge base RAG**: Markdown documents are chunked into a vector store, and answers must cite the source filename and chunk ID.
- **Injection and fabrication guards**: retrieved content is treated as untrusted data and never executed as instructions; when evidence is insufficient, the agent says so instead of inventing sources.
- **FastAPI service**: `/chat` returns a complete answer, and `/stream` pushes `start` / `token` / `done` / `error` events over SSE; requests are validated with Pydantic, with a timeout and client-disconnect detection.
- **Offline evaluation**: golden cases check tool usage, source citation, required terms, and forbidden phrases, and record the execution trace and token usage.
- **React frontend**: a Vite + React + TypeScript chat UI that consumes the SSE stream.

## Architecture

```text
React (Vite :5173) ──/api proxy──▶ FastAPI (:8000)
                                    ├─ POST /chat
                                    └─ POST /stream (SSE)
                                           │
                                  LangGraph StateGraph
                                  assistant ⇄ ToolNode
                                    ├─ search_im_messages → data/messages.json
                                    └─ search_knowledge   → vector store ← data/knowledge/*.md
```

## Quick Start

Requires Python 3.13 and an OpenAI-compatible model provider.

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS / Linux: source .venv/bin/activate
pip install -r requirements.lock
cp .env.example .env            # set LLM_MODEL, LLM_BASE_URL, LLM_API_KEY, EMBEDDING_MODEL
uvicorn app.api.main:app --reload --port 8000
```

Frontend:

```bash
cd frontend
pnpm install
pnpm dev
```

## API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/chat` | Request `{ message, thread_id? }`, returns `{ thread_id, answer }` |
| POST | `/stream` | Same request body, streamed over SSE; on failure an `error` event carries `AGENT_TIMEOUT` / `AGENT_FAILURE` |

## Tests and Evaluation

```bash
python -m pytest              # uses fake models, no API key needed
python -m labs.day05_eval     # runs golden cases against the real model, writes reports/day05_eval.json
```

### Deep research metrics (ported from chat-gun)

`app/eval/research_metrics.py` re-scores runs recorded by chat-gun's TypeScript Deep Researcher
evaluation (`research-eval-v1`): citation verification, no-source refusal, clarify routing,
fetched-source usability, rerank latency and nearest-rank P50/P90 latency. The recorded live runs
under `data/research-eval/` are checked for exact parity with the TypeScript scores:

```bash
python -m app.eval.research_metrics data/research-eval/research-eval-v1-2026-11-05-r1/run-rule.json --check
```

## Known Limitations

- The vector store and conversation checkpoints are in-memory and are lost on restart.
- IM message search is keyword matching, not semantic retrieval.
- No authentication or rate limiting; the agent is read-only and performs no write operations.
- The evaluation set is small and synthetic.
