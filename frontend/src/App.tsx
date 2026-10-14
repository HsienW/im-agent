
import { useRef, useState, type FormEvent } from "react";
import { streamAgent } from "./lib/agent-api";

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  status: "streaming" | "done" | "interrupted";
};

const STORAGE_KEY = "im-agent-thread-id";

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [threadId, setThreadId] = useState<string | null>(
      () => sessionStorage.getItem(STORAGE_KEY),
  );
  const [status, setStatus] = useState<
      "idle" | "streaming" | "interrupted"
  >("idle");
  const [error, setError] = useState<string | null>(null);

  const controllerRef = useRef<AbortController | null>(null);
  const requestVersion = useRef(0);

  async function sendMessage(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const question = input.trim();
    if (!question || status !== "idle") return;

    const version = ++requestVersion.current;
    const controller = new AbortController();
    const assistantId = crypto.randomUUID();

    controllerRef.current = controller;
    setInput("");
    setError(null);
    setStatus("streaming");

    setMessages((previous) => [
      ...previous,
      {
        id: crypto.randomUUID(),
        role: "user",
        content: question,
        status: "done",
      },
      {
        id: assistantId,
        role: "assistant",
        content: "",
        status: "streaming",
      },
    ]);

    try {
      await streamAgent(
          {
            message: question,
            ...(threadId ? { thread_id: threadId } : {}),
          },
          {
            onStart: (id) => {
              if (version !== requestVersion.current) return;

              setThreadId(id);
              sessionStorage.setItem(STORAGE_KEY, id);
            },

            onToken: (text) => {
              if (version !== requestVersion.current) return;

              setMessages((previous) =>
                  previous.map((message) =>
                      message.id === assistantId
                          ? {
                            ...message,
                            content: message.content + text,
                          }
                          : message,
                  ),
              );
            },

            onDone: () => {
              if (version !== requestVersion.current) return;

              setMessages((previous) =>
                  previous.map((message) =>
                      message.id === assistantId
                          ? { ...message, status: "done" }
                          : message,
                  ),
              );
            },
          },
          controller.signal,
      );

      if (version === requestVersion.current) {
        setStatus("idle");
      }
    } catch (cause) {
      if (version !== requestVersion.current) return;

      setStatus("interrupted");

      setMessages((previous) =>
          previous.map((message) =>
              message.id === assistantId
                  ? { ...message, status: "interrupted" }
                  : message,
          ),
      );

      setError(
          controller.signal.aborted
              ? "生成已停止。請建立新對話後繼續。"
              : cause instanceof Error
                  ? cause.message
                  : "Agent request failed.",
      );
    } finally {
      if (version === requestVersion.current) {
        controllerRef.current = null;
      }
    }
  }

  function newConversation() {
    requestVersion.current += 1;
    controllerRef.current?.abort();
    controllerRef.current = null;

    sessionStorage.removeItem(STORAGE_KEY);

    setThreadId(null);
    setMessages([]);
    setInput("");
    setStatus("idle");
    setError(null);
  }

  return (
      <main className="chat-container">
        <header className="chat-header">
          <div>
            <h1>IM Python Agent</h1>
            <p>React + FastAPI + LangGraph + Qwen</p>
          </div>

          <button type="button" onClick={newConversation}>
            + 新對話
          </button>
        </header>

        <div className="thread-bar">
        <span>
          Thread：{threadId ?? "尚未建立"}
        </span>
          <span>
          {status === "streaming"
              ? "● Streaming"
              : status === "interrupted"
                  ? "● Interrupted"
                  : "● Ready"}
        </span>
        </div>

        <section className="message-list" aria-live="polite">
          {messages.length === 0 && (
              <div className="empty-state">
                <h2>Ask your IM Agent</h2>
                <p>
                  可以搜尋 IM 訊息、查詢知識庫，
                  或詢問內部業務規則。
                </p>
              </div>
          )}

          {messages.map((message) => (
              <article
                  key={message.id}
                  className={`message ${message.role}`}
              >
                <strong>
                  {message.role === "user" ? "You" : "Agent"}
                </strong>

                <p>
                  {message.content ||
                      (message.status === "streaming"
                          ? "正在思考..."
                          : "沒有完整回答")}
                </p>

                {message.status === "interrupted" && (
                    <small>本次回合未確認完成</small>
                )}
              </article>
          ))}
        </section>

        {error && (
            <div className="error-box" role="alert">
              {error}
            </div>
        )}

        <form className="chat-form" onSubmit={sendMessage}>
        <textarea
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder="請輸入問題..."
            maxLength={2000}
            rows={3}
            disabled={status !== "idle"}
            onKeyDown={(event) => {
              if (
                  event.key === "Enter" &&
                  (event.ctrlKey || event.metaKey)
              ) {
                event.preventDefault();
                event.currentTarget.form?.requestSubmit();
              }
            }}
        />

          <div className="chat-actions">
            <span>Ctrl + Enter 發送</span>

            {status === "streaming" ? (
                <button
                    type="button"
                    onClick={() => controllerRef.current?.abort()}
                >
                  停止生成
                </button>
            ) : (
                <button
                    type="submit"
                    disabled={
                        status !== "idle" || !input.trim()
                    }
                >
                  發送訊息
                </button>
            )}
          </div>
        </form>
      </main>
  );
}
