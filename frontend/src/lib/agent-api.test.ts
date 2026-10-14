import {
    afterEach,
    describe,
    expect,
    it,
    vi,
} from "vitest";

import { streamAgent } from "./agent-api";

function mockSseResponse(
    content: string,
    splitAt?: number,
): Response {
    const bytes = new TextEncoder().encode(content);

    const pieces = splitAt === undefined
        ? [bytes]
        : [
            bytes.slice(0, splitAt),
            bytes.slice(splitAt),
        ];

    const stream = new ReadableStream<Uint8Array>({
        start(controller) {
            for (const piece of pieces) {
                controller.enqueue(piece);
            }

            controller.close();
        },
    });

    return new Response(stream, {
        status: 200,
        headers: {
            "Content-Type": "text/event-stream; charset=utf-8",
        },
    });
}

afterEach(() => {
    vi.unstubAllGlobals();
});

describe("streamAgent", () => {
    it("decodes streaming UTF-8 and completes", async () => {
        const content =
            'event: start\ndata: {"thread_id":"t-1"}\n\n' +
            'event: token\ndata: {"text":"你好🌏"}\n\n' +
            'event: done\ndata: {"thread_id":"t-1"}\n\n';

        const prefix = new TextEncoder().encode(
            'event: start\ndata: {"thread_id":"t-1"}\n\n' +
            'event: token\ndata: {"text":"',
        );

        const response = mockSseResponse(
            content,
            prefix.length + 1,
        );

        vi.stubGlobal(
            "fetch",
            vi.fn().mockResolvedValue(response),
        );

        const onStart = vi.fn();
        const onToken = vi.fn();
        const onDone = vi.fn();

        const threadId = await streamAgent(
            { message: "測試" },
            { onStart, onToken, onDone },
        );

        expect(threadId).toBe("t-1");
        expect(onStart).toHaveBeenCalledWith("t-1");
        expect(onToken).toHaveBeenCalledWith("你好🌏");
        expect(onDone).toHaveBeenCalledWith("t-1");
    });

    it("rejects streams without a done event", async () => {
        vi.stubGlobal(
            "fetch",
            vi.fn().mockResolvedValue(
                mockSseResponse(
                    'event: start\ndata: {"thread_id":"t-1"}\n\n',
                ),
            ),
        );

        await expect(
            streamAgent(
                { message: "測試" },
                {
                    onStart: vi.fn(),
                    onToken: vi.fn(),
                    onDone: vi.fn(),
                },
            ),
        ).rejects.toThrow("without SSE done");
    });

    it("propagates server error events", async () => {
        vi.stubGlobal(
            "fetch",
            vi.fn().mockResolvedValue(
                mockSseResponse(
                    'event: start\ndata: {"thread_id":"t-1"}\n\n' +
                    'event: error\ndata: {"code":"AGENT_FAILURE","message":"Agent streaming failed."}\n\n',
                ),
            ),
        );

        await expect(
            streamAgent(
                { message: "測試" },
                {
                    onStart: vi.fn(),
                    onToken: vi.fn(),
                    onDone: vi.fn(),
                },
            ),
        ).rejects.toThrow("AGENT_FAILURE");
    });

    it("rejects a mismatched thread ID", async () => {
        vi.stubGlobal(
            "fetch",
            vi.fn().mockResolvedValue(
                mockSseResponse(
                    'event: start\ndata: {"thread_id":"wrong-thread"}\n\n',
                ),
            ),
        );

        await expect(
            streamAgent(
                {
                    message: "測試",
                    thread_id: "expected-thread",
                },
                {
                    onStart: vi.fn(),
                    onToken: vi.fn(),
                    onDone: vi.fn(),
                },
            ),
        ).rejects.toThrow("thread ID mismatch");
    });
});
