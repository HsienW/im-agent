# Python research agent evaluation — research-eval-v1

Date: 2026-11-06 · Cases: 26 · LLM: qwen/qwen-plus · Search: Tavily Search API (basic depth, 8 results per query) · Commit: guard@9bbd73b

Small sample, single run per arm: treat differences of one or two cases as noise. Latency: wall-clock per graph invocation on a local developer machine, cases run sequentially (single concurrency), including LLM, search and fetch calls; not a production latency.

Metric definitions are the ones in `app/eval/research_metrics.py` (ported from chat-gun `scoring.ts`, parity checked on the recorded TypeScript runs).

## Summary

| Metric | python:guard | ts:guard |
|---|---|---|
| Completed / cases | 26/26 | 26/26 |
| Citation verification pass rate | 85.7% (12/14) | 100.0% (14/14) |
| Citation precision | 90.0% (36/40) | 100.0% (37/37) |
| Answerable answers with usable sources but no citation | 0 | 0 |
| Answerable cases with zero usable sources | 4 | 2 |
| No-source refusal rate (keyword detector) | 100.0% (6/6) | 100.0% (6/6) |
| No-source stub effective (0 usable) | 6/6 | 6/6 |
| Clarify rate | 100.0% (4/4) | 100.0% (4/4) |
| Fetched usable ratio (answerable) | 71.7% (43/60) | 64.3% (45/70) |
| Latency P50 / P90 — all | 15.4s / 33.6s (n=26) | 23.8s / 42.8s (n=26) |
| Latency P50 / P90 — answerable | 23.6s / 36.2s (n=16) | 29.5s / 43.1s (n=16) |
| Latency P50 / P90 — no_source | 10.8s / 17.1s (n=6) | 12.0s / 22.5s (n=6) |
| Latency P50 / P90 — clarify | 2.2s / 2.7s (n=4) | 2.8s / 4.2s (n=4) |
| Citation guard: answers modified (research + direct) | 38.9% (7/18) | 70.0% (14/20) |
| Citation guard: markers / URLs removed | 2 / 13 | 19 / 10 |

## Cases — python:guard

| Case | Status | Mode | Latency | Usable/Fetched | Cited | Unresolved | Result |
|---|---|---|---|---|---|---|---|
| ans-01 | completed | research | 24.3s | 4/5 | 2 | - | citation pass |
| ans-02 | completed | research | 23.6s | 3/5 | 3 | - | citation pass |
| ans-03 | completed | research | 28.0s | 3/5 | 2 | - | citation pass |
| ans-04 | completed | research | 23.0s | 4/5 | 4 | - | citation pass |
| ans-05 | completed | direct | 5.9s | 0/0 | 2 | [1] [2] | citation fail |
| ans-06 | completed | research | 33.6s | 4/5 | 4 | - | citation pass |
| ans-07 | completed | direct | 7.9s | 0/0 | 2 | [1] [2] | citation fail |
| ans-08 | completed | direct | 4.7s | 0/0 | 0 | - | no citation |
| ans-09 | completed | research | 21.0s | 3/5 | 3 | - | citation pass |
| ans-10 | completed | research | 32.1s | 5/5 | 3 | - | citation pass |
| ans-11 | completed | research | 36.2s | 4/5 | 3 | - | citation pass |
| ans-12 | completed | research | 45.1s | 4/5 | 4 | - | citation pass |
| ans-13 | completed | direct | 12.4s | 0/0 | 0 | - | no citation |
| ans-14 | completed | research | 18.3s | 1/5 | 1 | - | citation pass |
| ans-15 | completed | research | 26.6s | 3/5 | 2 | - | citation pass |
| ans-16 | completed | research | 25.2s | 5/5 | 5 | - | citation pass |
| nos-01 | completed | research | 7.5s | 0/3 | 0 | - | refusal pass |
| nos-02 | completed | research | 10.3s | 0/1 | 0 | - | refusal pass |
| nos-03 | completed | research | 10.8s | 0/2 | 0 | - | refusal pass |
| nos-04 | completed | research | 15.4s | 0/5 | 0 | - | refusal pass |
| nos-05 | completed | research | 17.1s | 0/5 | 0 | - | refusal pass |
| nos-06 | completed | research | 11.5s | 0/5 | 0 | - | refusal pass |
| clr-01 | completed | clarify | 2.7s | 0/0 | 0 | - | clarify pass |
| clr-02 | completed | clarify | 2.2s | 0/0 | 0 | - | clarify pass |
| clr-03 | completed | clarify | 2.1s | 0/0 | 0 | - | clarify pass |
| clr-04 | completed | clarify | 2.3s | 0/0 | 0 | - | clarify pass |

Routing: answerable:research=12, answerable:direct=4, no_source:research=6, clarify:clarify=4

## Notes

- Executed as a local Python process (app.eval.research_runner) against live qwen and Tavily APIs.
- no_source cases stub the tools in the runner: empty_search returns zero search results and fails every fetch (planner-supplied URLs unreachable); fetch_fail fails every fetch. All other cases used live Tavily search and live page fetches.
- Live web results change over time; reruns will not reproduce identical numbers.
- Searches and fetches run sequentially, as in the TypeScript agent; top-5 ranked sources are fetched.
- The local resolver returned fake-ip (198.18.0.0/15) answers, so the public-address check used DNS over HTTPS (https://223.5.5.5/resolve); connections then went through the system proxy settings.
- Known defect at this commit: the citation guard ran only for answerMode=research. The planner routed 4/16 answerable cases to direct (the TypeScript guard run routed 2/16), and 2 of those direct answers (ans-05, ans-07) contained invented [1]/[2] markers with no sources, which failed citation verification. Fixed afterwards (direct prompt plus guard on direct answers); see run r3.
