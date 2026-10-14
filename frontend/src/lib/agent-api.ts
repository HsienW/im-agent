
import { AgentSseParser } from "./sse";

export interface ChatRequest {
    message: string;
    thread_id?: string;
}

export interface StreamCallbacks {
    onStart: (threadId: string) => void;
    onToken: (text: string) => void;
    onDone: (threadId: string) => void;
}

/**
 * Stream an Agent response from the FastAPI SSE endpoint.
 *
 * A successful HTTP response does not imply Agent completion.
 * Completion requires a valid SSE "done" event.
 */
export async function streamAgent(
    payload: ChatRequest,
    callbacks: StreamCallbacks,
    signal?: AbortSignal,
): Promise<string> {
    const response = await fetch("/api/stream", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
        signal,
    });

    if (!response.ok) {
        throw new Error(`Agent HTTP error: ${response.status}`);
    }

    const contentType =
        response.headers.get("content-type") ?? "";

    if (!contentType.includes("text/event-stream")) {
        throw new Error("Expected SSE response");
    }

    if (!response.body) {
        throw new Error(
            "Streaming response body is unavailable",
        );
    }

    let startedThreadId: string | null = null;
    let completed = false;

    const parser = new AgentSseParser((event) => {
        if (completed) {
            throw new Error(
                "SSE event received after completion",
            );
        }

        switch (event.event) {
            case "start": {
                if (startedThreadId !== null) {
                    throw new Error("Duplicate SSE start");
                }

                const receivedId = event.data.thread_id;

                if (
                    payload.thread_id &&
                    payload.thread_id !== receivedId
                ) {
                    throw new Error("SSE thread ID mismatch");
                }

                startedThreadId = receivedId;
                callbacks.onStart(receivedId);
                break;
            }

            case "token": {
                if (!startedThreadId) {
                    throw new Error(
                        "SSE token received before start",
                    );
                }

                callbacks.onToken(event.data.text);
                break;
            }

            case "done": {
                if (!startedThreadId) {
                    throw new Error(
                        "SSE done received before start",
                    );
                }

                if (event.data.thread_id !== startedThreadId) {
                    throw new Error(
                        "SSE completion thread ID mismatch",
                    );
                }

                completed = true;
                callbacks.onDone(startedThreadId);
                break;
            }

            case "error": {
                throw new Error(
                    `${event.data.code}: ${event.data.message}`,
                );
            }
        }
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");

    try {
        while (true) {
            const { done, value } = await reader.read();

            if (done) {
                break;
            }

            // Preserve incomplete UTF-8 characters across chunks.
            parser.push(
                decoder.decode(value, {
                    stream: true,
                }),
            );
        }

        // Flush any remaining UTF-8 bytes.
        parser.push(decoder.decode());

        // Reject incomplete SSE frames.
        parser.finish();

        if (!completed || !startedThreadId) {
            throw new Error(
                "Stream ended without SSE done event",
            );
        }

        return startedThreadId;
    } finally {
        if (!completed) {
            await reader.cancel().catch(() => undefined);
        }

        reader.releaseLock();
    }
}
