
export type AgentSseEvent =
    | {
    event: "start";
    data: { thread_id: string };
}
    | {
    event: "token";
    data: { text: string };
}
    | {
    event: "done";
    data: { thread_id: string };
}
    | {
    event: "error";
    data: {
        code: string;
        message: string;
    };
};

function isRecord(
    value: unknown,
): value is Record<string, unknown> {
    return (
        typeof value === "object" &&
        value !== null &&
        !Array.isArray(value)
    );
}

function parseEvent(
    frame: string,
): AgentSseEvent | null {
    let eventName = "";
    const dataLines: string[] = [];

    for (const line of frame.split(/\r?\n/)) {
        if (line.startsWith(":")) {
            continue;
        }

        const separatorIndex = line.indexOf(":");

        const field =
            separatorIndex < 0
                ? line
                : line.slice(0, separatorIndex);

        const rawValue =
            separatorIndex < 0
                ? ""
                : line.slice(separatorIndex + 1);

        const value = rawValue.startsWith(" ")
            ? rawValue.slice(1)
            : rawValue;

        if (field === "event") {
            eventName = value;
        }

        if (field === "data") {
            dataLines.push(value);
        }
    }

    // Ignore heartbeat and unsupported events.
    if (
        dataLines.length === 0 ||
        !["start", "token", "done", "error"].includes(
            eventName,
        )
    ) {
        return null;
    }

    const payload: unknown = JSON.parse(
        dataLines.join("\n"),
    );

    if (!isRecord(payload)) {
        throw new Error("Invalid SSE payload");
    }

    if (eventName === "token") {
        if (typeof payload.text !== "string") {
            throw new Error("Invalid SSE token");
        }

        return {
            event: "token",
            data: {
                text: payload.text,
            },
        };
    }

    if (eventName === "error") {
        if (
            typeof payload.code !== "string" ||
            typeof payload.message !== "string"
        ) {
            throw new Error("Invalid SSE error");
        }

        return {
            event: "error",
            data: {
                code: payload.code,
                message: payload.message,
            },
        };
    }

    if (
        typeof payload.thread_id !== "string" ||
        !payload.thread_id
    ) {
        throw new Error("Invalid SSE thread ID");
    }

    return {
        event: eventName as "start" | "done",
        data: {
            thread_id: payload.thread_id,
        },
    };
}

export class AgentSseParser {
    private buffer = "";

    private readonly onEvent: (
        event: AgentSseEvent,
    ) => void;

    constructor(
        onEvent: (event: AgentSseEvent) => void,
    ) {
        this.onEvent = onEvent;
    }

    push(text: string): void {
        this.buffer += text;

        while (true) {
            // SSE events are separated by a blank line.
            const boundary = /\r?\n\r?\n/.exec(
                this.buffer,
            );

            if (!boundary) {
                break;
            }

            const frame = this.buffer.slice(
                0,
                boundary.index,
            );

            this.buffer = this.buffer.slice(
                boundary.index + boundary[0].length,
            );

            const event = parseEvent(frame);

            if (event) {
                this.onEvent(event);
            }
        }

        // Prevent unbounded memory usage.
        if (this.buffer.length > 1024 * 1024) {
            throw new Error(
                "SSE event exceeds size limit",
            );
        }
    }

    finish(): void {
        if (this.buffer.trim()) {
            throw new Error(
                "Incomplete SSE event",
            );
        }
    }
}
