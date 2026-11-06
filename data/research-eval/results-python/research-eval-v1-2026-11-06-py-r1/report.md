# Python research agent evaluation — research-eval-v1

Date: 2026-11-06 · Cases: 26 · LLM: qwen/qwen-plus · Search: Tavily Search API (basic depth, 8 results per query) · Commit: guard@3483712

Small sample, single run per arm: treat differences of one or two cases as noise. Latency: wall-clock per graph invocation on a local developer machine, cases run sequentially (single concurrency), including LLM, search and fetch calls; not a production latency.

Metric definitions are the ones in `app/eval/research_metrics.py` (ported from chat-gun `scoring.ts`, parity checked on the recorded TypeScript runs).

## Summary

| Metric | python:guard | ts:guard |
|---|---|---|
| Completed / cases | 26/26 | 26/26 |
| Citation verification pass rate | 100.0% (16/16) | 100.0% (14/14) |
| Citation precision | 100.0% (43/43) | 100.0% (37/37) |
| Answerable answers with usable sources but no citation | 0 | 0 |
| Answerable cases with zero usable sources | 0 | 2 |
| No-source refusal rate (keyword detector) | 100.0% (6/6) | 100.0% (6/6) |
| No-source stub effective (0 usable) | 6/6 | 6/6 |
| Clarify rate | 100.0% (4/4) | 100.0% (4/4) |
| Fetched usable ratio (answerable) | 62.5% (50/80) | 64.3% (45/70) |
| Latency P50 / P90 — all | 25.1s / 55.7s (n=26) | 23.8s / 42.8s (n=26) |
| Latency P50 / P90 — answerable | 41.4s / 58.9s (n=16) | 29.5s / 43.1s (n=16) |
| Latency P50 / P90 — no_source | 11.0s / 17.7s (n=6) | 12.0s / 22.5s (n=6) |
| Latency P50 / P90 — clarify | 2.5s / 2.9s (n=4) | 2.8s / 4.2s (n=4) |
| Citation guard: research answers modified | 31.8% (7/22) | 70.0% (14/20) |
| Citation guard: markers / URLs removed | 8 / 8 | 19 / 10 |

## Cases — python:guard

| Case | Status | Mode | Latency | Usable/Fetched | Cited | Unresolved | Result |
|---|---|---|---|---|---|---|---|
| ans-01 | completed | research | 53.5s | 5/5 | 5 | - | citation pass |
| ans-02 | completed | research | 54.6s | 1/5 | 1 | - | citation pass |
| ans-03 | completed | research | 62.1s | 2/5 | 1 | - | citation pass |
| ans-04 | completed | research | 36.9s | 5/5 | 2 | - | citation pass |
| ans-05 | completed | research | 54.2s | 3/5 | 3 | - | citation pass |
| ans-06 | completed | research | 55.7s | 3/5 | 3 | - | citation pass |
| ans-07 | completed | research | 41.4s | 4/5 | 4 | - | citation pass |
| ans-08 | completed | research | 40.4s | 3/5 | 3 | - | citation pass |
| ans-09 | completed | research | 49.5s | 2/5 | 2 | - | citation pass |
| ans-10 | completed | research | 51.9s | 4/5 | 4 | - | citation pass |
| ans-11 | completed | research | 58.9s | 4/5 | 3 | - | citation pass |
| ans-12 | completed | research | 25.1s | 1/5 | 2 | - | citation pass |
| ans-13 | completed | research | 20.5s | 4/5 | 3 | - | citation pass |
| ans-14 | completed | research | 28.5s | 1/5 | 1 | - | citation pass |
| ans-15 | completed | research | 31.9s | 4/5 | 2 | - | citation pass |
| ans-16 | completed | research | 24.8s | 4/5 | 4 | - | citation pass |
| nos-01 | completed | research | 7.4s | 0/3 | 0 | - | refusal pass |
| nos-02 | completed | research | 9.5s | 0/2 | 0 | - | refusal pass |
| nos-03 | completed | research | 11.0s | 0/3 | 0 | - | refusal pass |
| nos-04 | completed | research | 14.4s | 0/5 | 0 | - | refusal pass |
| nos-05 | completed | research | 17.7s | 0/5 | 0 | - | refusal pass |
| nos-06 | completed | research | 17.0s | 0/5 | 0 | - | refusal pass |
| clr-01 | completed | clarify | 2.9s | 0/0 | 0 | - | clarify pass |
| clr-02 | completed | clarify | 2.5s | 0/0 | 0 | - | clarify pass |
| clr-03 | completed | clarify | 2.5s | 0/0 | 0 | - | clarify pass |
| clr-04 | completed | clarify | 2.1s | 0/0 | 0 | - | clarify pass |

Routing: answerable:research=16, no_source:research=6, clarify:clarify=4

## Notes

- Executed as a local Python process (app.eval.research_runner) against live qwen and Tavily APIs.
- no_source cases stub the tools in the runner: empty_search returns zero search results and fails every fetch (planner-supplied URLs unreachable); fetch_fail fails every fetch. All other cases used live Tavily search and live page fetches.
- Live web results change over time; reruns will not reproduce identical numbers.
- Searches and fetches run sequentially, as in the TypeScript agent; top-5 ranked sources are fetched.
- The local resolver returned fake-ip (198.18.0.0/15) answers, so the public-address check used DNS over HTTPS (https://223.5.5.5/resolve); connections then went through the system proxy settings.
- Known defect at this commit: 4/26 planner outputs (ans-05, ans-07, ans-08, ans-13) were rejected by pydantic only because qwen returned "freshness": "", so those cases fell back to a single query (the question). Fixed afterwards; see run r2.
