import { describe, expect, it } from "vitest";
import { AgentSseParser, type AgentSseEvent } from "./sse";

describe("AgentSseParser", () => {
    it("parses events split across network chunks", () => {
        const events: AgentSseEvent[] = [];

        const parser = new AgentSseParser((event) => {
            events.push(event);
        });

        parser.push("event: start\r\n");
        parser.push('data: {"thread_id":"t-1"}\r\n\r\n');

        parser.push(
            'event: token\ndata: {"text":"\\u6839\\u64da"}\n\n',
        );

        parser.push(
            'event: done\ndata: {"thread_id":"t-1"}\n\n',
        );

        parser.finish();

        expect(events.map((event) => event.event)).toEqual([
            "start",
            "token",
            "done",
        ]);

        expect(events[1]).toEqual({
            event: "token",
            data: { text: "根據" },
        });
    });

    it("ignores heartbeat comments", () => {
        const events: AgentSseEvent[] = [];

        const parser = new AgentSseParser((event) => {
            events.push(event);
        });

        parser.push(": heartbeat\n\n");
        parser.finish();

        expect(events).toEqual([]);
    });

    it("rejects malformed JSON events", () => {
        const parser = new AgentSseParser(() => undefined);

        expect(() => {
            parser.push(
                "event: token\ndata: {invalid}\n\n",
            );
        }).toThrow();
    });
});
