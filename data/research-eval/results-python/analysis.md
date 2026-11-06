# Python vs TypeScript research agent on research-eval-v1

Same 26 cases (dataset sha256 `d27b1a73…`), same LLM (qwen-plus), same search (Tavily basic, 8
results per query), same no_source stubs, same DoH resolver, same scorer (`app/eval/research_metrics.py`,
parity-checked against chat-gun `scoring.ts`). Both columns are citation-guard arms. One run per
column; live web results change between runs.

- TypeScript: chat-gun Deep Researcher, `research-eval-v1-2026-11-05-r1/run-guard.json` (2026-11-05).
- Python: `app/research` (LangGraph), runs r1–r3 on 2026-11-06. **r3 is the current code**
  (commit `2e018fb`); r1 and r2 each exposed one defect, fixed before the next run (see below).

| Metric | TS guard (11-05) | Py r1 | Py r2 | **Py r3** |
|---|---|---|---|---|
| Completed | 26/26 | 26/26 | 26/26 | 26/26 |
| Citation verification pass | 14/14 | 16/16 | 12/14 | **11/11** |
| Citation precision | 37/37 | 43/43 | 36/40 | **21/21** |
| Answerable routed to direct (no search) | 2/16 | 0/16 | 4/16 | 4/16 |
| Answerable with zero usable sources | 2 | 0 | 4 | 5 |
| Fetched usable ratio | 45/70 | 50/80 | 43/60 | 29/58 |
| No-source refusal, keyword detector | 6/6 | 6/6 | 6/6 | **6/6** |
| No-source, manual: no unsupported claims | 3/6 | 4/6 | 4/6 | **4/6** |
| Clarify | 4/4 | 4/4 | 4/4 | **4/4** |
| Latency P50 / P90 — all | 23.8s / 42.8s | 25.1s / 55.7s | 15.4s / 33.6s | **14.2s / 48.5s** |
| Latency P50 / P90 — answerable | 29.5s / 43.1s | 41.4s / 58.9s | 23.6s / 36.2s | **29.0s / 52.0s** |

For reference, the TypeScript prompt-only arm (`run-rule.json`, no deterministic guard) passed
citation verification on 3/14 answers. No Python prompt-only arm was run.

Latency is wall-clock per graph invocation on a local developer machine, cases run one at a time,
through a local proxy; it is not a production latency. Answerable P50 moved between 23.6s and 41.4s
across three Python runs of the same code path, so latency differences between the columns are
within run-to-run variation.

## Defects found by the live runs

1. **r1 — planner fallback on an optional field.** qwen returned `"freshness": ""`; pydantic rejected
   the whole plan, so 4/26 cases fell back to one query (the question). chat-gun's `coercePlan`
   ignores an invalid freshness. Fix `9bbd73b`: optional fields degrade to defaults; `answerMode`
   stays strict.
2. **r2 — invented citations in direct answers.** With valid plans the planner sent 4/16 answerable
   cases to direct. Two of those answers contained `[1]`/`[2]` with no sources, and the guard only
   covered research answers. Fix `2e018fb`: direct answers get a no-citation prompt and also pass
   through the guard (with no sources, every marker or URL is removed).

Both fixes were made after looking at results on this dataset. The planner prompt, the rule ranking
and the guard rules were not changed between runs.

## What the numbers do and do not say

- Citation verification is scored **after** the guard, so 100% mostly shows the deterministic step
  works; it does not measure answer quality. r3 has fewer citing answers (11) than TS (14) because 4
  answerable cases were answered directly and 1 research case (ans-02) got no usable source.
- The no-source keyword detector passed 6/6 in every run, but manual inspection found 2/6 answers
  per Python run asserting unsupported facts (expected publication dates, rebalance schedules),
  against 3/6 for the TS guard run. The guard does not catch these; they are claims, not citations.
- Manual inspection was done by the same agent that built the Python agent, not an independent reviewer.
- Fetch failures in this environment include DoH answers outside public ranges (e.g. Teredo `2001::/32`
  for some hosts), which the egress check rejects by design, and 12 s fetch timeouts through the proxy.
