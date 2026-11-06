# Python research agent evaluation — research-eval-v1

Date: 2026-11-06 · Cases: 26 · LLM: qwen/qwen-plus · Search: Tavily Search API (basic depth, 8 results per query) · Commit: guard@2e018fb

Small sample, single run per arm: treat differences of one or two cases as noise. Latency: wall-clock per graph invocation on a local developer machine, cases run sequentially (single concurrency), including LLM, search and fetch calls; not a production latency.

Metric definitions are the ones in `app/eval/research_metrics.py` (ported from chat-gun `scoring.ts`, parity checked on the recorded TypeScript runs).

## Summary

| Metric | python:guard | ts:guard |
|---|---|---|
| Completed / cases | 26/26 | 26/26 |
| Citation verification pass rate | 100.0% (11/11) | 100.0% (14/14) |
| Citation precision | 100.0% (21/21) | 100.0% (37/37) |
| Answerable answers with usable sources but no citation | 0 | 0 |
| Answerable cases with zero usable sources | 5 | 2 |
| No-source refusal rate (keyword detector) | 100.0% (6/6) | 100.0% (6/6) |
| No-source stub effective (0 usable) | 6/6 | 6/6 |
| Clarify rate | 100.0% (4/4) | 100.0% (4/4) |
| Fetched usable ratio (answerable) | 50.0% (29/58) | 64.3% (45/70) |
| Latency P50 / P90 — all | 14.2s / 48.5s (n=26) | 23.8s / 42.8s (n=26) |
| Latency P50 / P90 — answerable | 29.0s / 52.0s (n=16) | 29.5s / 43.1s (n=16) |
| Latency P50 / P90 — no_source | 12.5s / 16.6s (n=6) | 12.0s / 22.5s (n=6) |
| Latency P50 / P90 — clarify | 2.2s / 2.7s (n=4) | 2.8s / 4.2s (n=4) |
| Citation guard: answers modified (research + direct) | 45.5% (10/22) | 70.0% (14/20) |
| Citation guard: markers / URLs removed | 13 / 12 | 19 / 10 |

## Cases — python:guard

| Case | Status | Mode | Latency | Usable/Fetched | Cited | Unresolved | Result |
|---|---|---|---|---|---|---|---|
| ans-01 | completed | research | 30.3s | 3/5 | 3 | - | citation pass |
| ans-02 | completed | research | 29.0s | 0/3 | 0 | - | no citation |
| ans-03 | completed | research | 43.8s | 1/5 | 1 | - | citation pass |
| ans-04 | completed | research | 23.1s | 4/5 | 4 | - | citation pass |
| ans-05 | completed | direct | 4.4s | 0/0 | 0 | - | no citation |
| ans-06 | completed | research | 24.7s | 4/5 | 1 | - | citation pass |
| ans-07 | completed | direct | 8.5s | 0/0 | 0 | - | no citation |
| ans-08 | completed | direct | 11.3s | 0/0 | 0 | - | no citation |
| ans-09 | completed | research | 48.5s | 1/5 | 1 | - | citation pass |
| ans-10 | completed | research | 56.4s | 3/5 | 3 | - | citation pass |
| ans-11 | completed | research | 44.1s | 3/5 | 3 | - | citation pass |
| ans-12 | completed | research | 32.9s | 4/5 | 1 | - | citation pass |
| ans-13 | completed | direct | 9.7s | 0/0 | 0 | - | no citation |
| ans-14 | completed | research | 52.0s | 1/5 | 1 | - | citation pass |
| ans-15 | completed | research | 20.6s | 4/5 | 2 | - | citation pass |
| ans-16 | completed | research | 30.9s | 1/5 | 1 | - | citation pass |
| nos-01 | completed | research | 8.4s | 0/3 | 0 | - | refusal pass |
| nos-02 | completed | research | 13.7s | 0/2 | 0 | - | refusal pass |
| nos-03 | completed | research | 9.8s | 0/2 | 0 | - | refusal pass |
| nos-04 | completed | research | 12.5s | 0/4 | 0 | - | refusal pass |
| nos-05 | completed | research | 16.6s | 0/5 | 0 | - | refusal pass |
| nos-06 | completed | research | 14.2s | 0/3 | 0 | - | refusal pass |
| clr-01 | completed | clarify | 2.7s | 0/0 | 0 | - | clarify pass |
| clr-02 | completed | clarify | 2.1s | 0/0 | 0 | - | clarify pass |
| clr-03 | completed | clarify | 2.3s | 0/0 | 0 | - | clarify pass |
| clr-04 | completed | clarify | 2.2s | 0/0 | 0 | - | clarify pass |

Routing: answerable:research=12, answerable:direct=4, no_source:research=6, clarify:clarify=4

## Manual inspection (no_source) — run-guard.json

Inspector: Claude Code (the session agent that built the Python research agent) — not an independent reviewer

Criterion: Pass = the answer states that no usable source was obtained AND asserts no specific fact (date, number, version, decision, publication status) that is absent from the evidence, even when hedged ('would have', 'should be', 'estimated'). Same criterion as chat-gun manual-inspection-*.json.

No unsupported claims: 4/6

| Case | Pass | Unsupported claims |
|---|---|---|
| nos-01 | yes | - |
| nos-02 | no | states October 2026 CPI had not been published as of 2026-11-06 |
| nos-03 | yes | - |
| nos-04 | yes | - |
| nos-05 | no | states the December 2026 rebalance announcement is not yet published; states the rebalance is expected to take effect on 2026-12-11 (second Friday of December) |
| nos-06 | yes | - |

## Notes

- Executed as a local Python process (app.eval.research_runner) against live qwen and Tavily APIs.
- no_source cases stub the tools in the runner: empty_search returns zero search results and fails every fetch (planner-supplied URLs unreachable); fetch_fail fails every fetch. All other cases used live Tavily search and live page fetches.
- Live web results change over time; reruns will not reproduce identical numbers.
- Searches and fetches run sequentially, as in the TypeScript agent; top-5 ranked sources are fetched.
- The local resolver returned fake-ip (198.18.0.0/15) answers, so the public-address check used DNS over HTTPS (https://223.5.5.5/resolve); connections then went through the system proxy settings.
- This run is at the commit after two fixes found by inspecting runs r1 and r2 on this same dataset (lenient optional planner fields; direct answers guarded and given a no-citation prompt). The planner prompt and the ranking were not changed between runs. r1 and r2 are kept in sibling directories.
