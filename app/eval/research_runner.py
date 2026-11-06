"""Live research-eval-v1 run of the Python research agent (``app/research``).

    python -m app.eval.research_runner --env-file path/to/.env --doh-url https://223.5.5.5/resolve

Writes ``run-<arm>.json`` (same observation shape as chat-gun's TypeScript runner), scored with
``app.eval.research_metrics``, plus ``provenance.json`` and ``report.md`` into
``data/research-eval/results-python/<label>/``. ``--cases`` runs a subset and writes only to
``--smoke-out``.

Requires ``LLM_MODEL`` / ``LLM_BASE_URL`` / ``LLM_API_KEY`` and ``TAVILY_API_KEY``. no_source
cases stub the tools exactly as the TypeScript runner does: ``empty_search`` returns zero search
results and fails every fetch (so planner-supplied URLs are unreachable too); ``fetch_fail``
fails every fetch. Every other case uses live Tavily search and live page fetches.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

from app.eval.research_metrics import CATEGORIES, score_case, summarize_run
from app.research.agent import build_research_graph, to_observation
from app.research.web import FetchResult, doh_resolver, fetch_page, system_resolver, tavily_search

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "research-eval"
RUNNER_VERSION = "python-research-eval-runner-v1"
STUB_FETCH_ERROR = "Error: web_fetch failed - 503 Service Unavailable (research-eval stub)"
EMPTY_TRACE = {"queries": [], "searchResultCount": 0, "rankedUrls": [], "sources": [], "answer": ""}


def load_dataset(path: Path) -> tuple[dict[str, Any], str]:
    dataset = json.loads(path.read_text(encoding="utf-8"))
    ids = [case["caseId"] for case in dataset["cases"]]
    if len(set(ids)) != len(ids):
        raise ValueError("caseId values must be unique")
    for case in dataset["cases"]:
        if case["category"] not in CATEGORIES:
            raise ValueError(f"unknown category in {case['caseId']}")
        if (case["category"] == "no_source") != (case.get("stub") in ("empty_search", "fetch_fail")):
            raise ValueError(f"no_source cases need a stub and other cases must not have one: {case['caseId']}")
    # Same canonical form as the TypeScript loader (sorted keys, compact JSON) so the hashes match.
    canonical = json.dumps(dataset, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return dataset, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- report


def _pct(rate: dict[str, Any]) -> str:
    if rate["rate"] is None:
        return "n/a (0)"
    return f"{rate['rate'] * 100:.1f}% ({rate['passed']}/{rate['n']})"


def _seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value / 1000:.1f}s"


def _latency(value: dict[str, Any] | None) -> str:
    return f"{_seconds(value['p50Ms'])} / {_seconds(value['p90Ms'])} (n={value['n']})" if value else "n/a"


def _manual(inspection: dict[str, Any]) -> list[str]:
    items = inspection.get("items", [])
    return [
        "",
        f"## Manual inspection (no_source) — {inspection.get('run')}",
        "",
        f"Inspector: {inspection.get('inspector')}",
        "",
        f"Criterion: {inspection.get('criterion')}",
        "",
        f"No unsupported claims: {sum(1 for item in items if item.get('pass'))}/{len(items)}",
        "",
        "| Case | Pass | Unsupported claims |",
        "|---|---|---|",
        *[
            f"| {item['caseId']} | {'yes' if item.get('pass') else 'no'} | {'; '.join(item.get('unsupportedClaims', [])) or '-'} |"
            for item in items
        ],
    ]


def render_report(
    runs: list[dict[str, Any]],
    provenance: dict[str, Any],
    manual: list[dict[str, Any]],
    reference_runs: list[dict[str, Any]] | None = None,
) -> str:
    """Python counterpart of chat-gun ``report.ts``; reference runs appear in the summary only."""
    columns = runs + (reference_runs or [])
    lines = [
        f"# Python research agent evaluation — {provenance['datasetVersion']}",
        "",
        f"Date: {provenance['date']} · Cases: {provenance['caseCount']} · LLM: {provenance['llmProvider']}/"
        f"{provenance['plannerModel']} · Search: {provenance['searchProvider']} · Commit: {provenance['commit']}",
        "",
        "Small sample, single run per arm: treat differences of one or two cases as noise. "
        f"Latency: {provenance['latencyScope']}",
        "",
        "Metric definitions are the ones in `app/eval/research_metrics.py` (ported from chat-gun `scoring.ts`, "
        "parity checked on the recorded TypeScript runs).",
        "",
        "## Summary",
        "",
        f"| Metric | {' | '.join(run['label'] for run in columns)} |",
        f"|---|{'|'.join('---' for _ in columns)}|",
    ]

    def row(label: str, value) -> None:
        lines.append(f"| {label} | {' | '.join(value(run['summary']) for run in columns)} |")

    row("Completed / cases", lambda s: f"{s['completed']}/{s['cases']}")
    row("Citation verification pass rate", lambda s: _pct(s["citation"]["answerPassRate"]))
    row("Citation precision", lambda s: _pct(s["citation"]["citationPrecision"]))
    row("Answerable answers with usable sources but no citation", lambda s: str(s["citation"]["answerableWithoutCitation"]))
    row("Answerable cases with zero usable sources", lambda s: str(s["citation"]["answerableWithZeroUsableSources"]))
    row("No-source refusal rate (keyword detector)", lambda s: _pct(s["refusal"]))
    row("No-source stub effective (0 usable)", lambda s: f"{s['refusal']['stubEffective']}/{s['refusal']['n']}")
    row("Clarify rate", lambda s: _pct(s["clarify"]))
    row("Fetched usable ratio (answerable)", lambda s: _pct(s["fetchedUsable"]))
    for category in ("all", *CATEGORIES):
        row(f"Latency P50 / P90 — {category}", lambda s, c=category: _latency(s["latency"].get(c)))
    if any(run["summary"].get("citationGuard") for run in columns):
        row(
            "Citation guard: research answers modified",
            lambda s: _pct(s["citationGuard"]["answersModified"]) if s.get("citationGuard") else "-",
        )
        row(
            "Citation guard: markers / URLs removed",
            lambda s: f"{s['citationGuard']['removedMarkers']} / {s['citationGuard']['removedUrls']}" if s.get("citationGuard") else "-",
        )

    for run in runs:
        scores = {score["caseId"]: score for score in run["scores"]}
        lines += [
            "",
            f"## Cases — {run['label']}",
            "",
            "| Case | Status | Mode | Latency | Usable/Fetched | Cited | Unresolved | Result |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for obs in run["observations"]:
            score = scores[obs["caseId"]]
            usable = sum(1 for source in obs["sources"] if source["usable"])
            if obs["category"] == "no_source":
                result = "refusal pass" if score["refusal"]["pass"] else "refusal fail"
            elif obs["category"] == "clarify":
                result = "clarify pass" if score["clarifyPass"] else "clarify fail"
            else:
                passed = score["citation"]["pass"]
                result = "no citation" if passed is None else "citation pass" if passed else "citation fail"
            lines.append(
                f"| {obs['caseId']} | {obs['status']} | {obs.get('answerMode') or '-'} | {_seconds(obs['latencyMs'])} | "
                f"{usable}/{len(obs['sources'])} | {score['citation']['citedCount']} | "
                f"{' '.join(score['citation']['unresolved']) or '-'} | {result} |"
            )
        routing = ", ".join(f"{key}={count}" for key, count in run["summary"]["routing"].items())
        lines += ["", f"Routing: {routing}"]
    for inspection in manual:
        lines += _manual(inspection)
    if provenance.get("notes"):
        lines += ["", "## Notes", "", *[f"- {note}" for note in provenance["notes"]]]
    return "\n".join(lines) + "\n"


def write_report(out_dir: Path, reference: list[Path]) -> None:
    """(Re)render report.md from the run files, provenance and manual inspections in ``out_dir``."""
    runs = []
    for path in sorted(out_dir.glob("run-*.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        runs.append({**run, "label": f"python:{run['ranking']}"})
    reference_runs = []
    for path in reference:
        run = json.loads(path.read_text(encoding="utf-8"))
        scores = [score_case(obs) for obs in run["observations"]]
        reference_runs.append(
            {**run, "scores": scores, "summary": summarize_run(run["observations"], scores), "label": f"ts:{run['ranking']}"}
        )
    provenance = json.loads((out_dir / "provenance.json").read_text(encoding="utf-8"))
    manual = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(out_dir.glob("manual-inspection-*.json"))]
    (out_dir / "report.md").write_text(render_report(runs, provenance, manual, reference_runs), encoding="utf-8")


# --------------------------------------------------------------------------- live run


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


async def run_cases(
    cases: list[dict[str, Any]],
    *,
    citation_guard: bool,
    doh_url: str | None,
    timeout_s: float,
) -> list[dict[str, Any]]:
    from app.models.llm import create_chat_model

    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("TAVILY_API_KEY is not set; add it to the .env file passed with --env-file.")
    planner_model = create_chat_model(timeout=60).bind(response_format={"type": "json_object"})
    synthesis_model = create_chat_model(temperature=0.1, timeout=120)
    # Preflight: a dead LLM account would otherwise turn every case into a planner fallback plus a
    # "synthesis failed" answer, which is not a measurement of the agent.
    try:
        await synthesis_model.ainvoke("Reply with OK.")
    except Exception as error:
        raise SystemExit(f"LLM preflight failed, no cases were run: {type(error).__name__}: {str(error)[:200]}") from None
    active_stub: dict[str, str | None] = {"value": None}
    observations: list[dict[str, Any]] = []

    async with httpx.AsyncClient() as search_client, httpx.AsyncClient() as fetch_client, httpx.AsyncClient() as dns_client:
        resolver = doh_resolver(doh_url, dns_client) if doh_url else system_resolver

        async def search(query: str, time_range: str | None):
            if active_stub["value"] == "empty_search":
                return []
            return await tavily_search(search_client, api_key, query, max_results=8, time_range=time_range)

        async def fetch(url: str) -> FetchResult:
            if active_stub["value"]:
                return FetchResult(url, error=STUB_FETCH_ERROR)
            return await fetch_page(fetch_client, url, resolver=resolver)

        graph = build_research_graph(
            planner_model=planner_model,
            synthesis_model=synthesis_model,
            search=search,
            fetch=fetch,
            citation_guard=citation_guard,
        )
        for position, case in enumerate(cases, start=1):
            active_stub["value"] = case.get("stub")
            identity = {"caseId": case["caseId"], "category": case["category"]}
            if case.get("stub"):
                identity["stub"] = case["stub"]
            started = time.perf_counter()
            try:
                state = await asyncio.wait_for(graph.ainvoke({"question": case["question"]}), timeout_s)
                observation = {**identity, "status": "completed", "latencyMs": round((time.perf_counter() - started) * 1000), **to_observation(state)}
            except asyncio.TimeoutError:
                observation = {**identity, "status": "timeout", "error": "RESEARCH_EVAL_CASE_TIMEOUT", "latencyMs": round((time.perf_counter() - started) * 1000), **EMPTY_TRACE}
            except Exception as error:  # record and continue; the run reports failed cases
                observation = {**identity, "status": "failed", "error": f"{type(error).__name__}: {error}"[:500], "latencyMs": round((time.perf_counter() - started) * 1000), **EMPTY_TRACE}
            finally:
                active_stub["value"] = None
            observations.append(observation)
            print(
                f"[research-eval] {position}/{len(cases)} {observation['caseId']} {observation['status']} "
                f"{observation['latencyMs']}ms mode={observation.get('answerMode') or '-'}",
                flush=True,
            )
    return observations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run research-eval-v1 against the Python research agent.")
    parser.add_argument("--dataset", type=Path, default=DATA_DIR / "research-eval-v1.json")
    parser.add_argument("--arm", choices=("guard", "rule"), default="guard", help="guard = citation guard on; rule = prompt only")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--doh-url", default=os.getenv("RESEARCH_EVAL_DOH_URL") or None)
    parser.add_argument("--label", default=None, help="results directory name (default research-eval-v1-<date>-py-r1)")
    parser.add_argument("--cases", default=None, help="comma-separated caseIds for a smoke run")
    parser.add_argument("--smoke-out", type=Path, default=None)
    parser.add_argument("--timeout", type=float, default=180.0, help="per-case timeout in seconds")
    parser.add_argument("--reference", type=Path, action="append", default=[], help="TypeScript run-*.json shown next to the Python run")
    parser.add_argument("--report-only", type=Path, default=None, help="re-render report.md in this results directory")
    args = parser.parse_args(argv)

    if args.report_only:
        write_report(args.report_only, args.reference)
        return 0

    load_dotenv(args.env_file, override=False)
    dataset, sha256 = load_dataset(args.dataset)
    selected = dataset["cases"]
    if args.cases:
        wanted = {case_id.strip() for case_id in args.cases.split(",") if case_id.strip()}
        selected = [case for case in selected if case["caseId"] in wanted]
    started_at = dt.datetime.now(dt.timezone.utc)
    observations = asyncio.run(run_cases(selected, citation_guard=args.arm == "guard", doh_url=args.doh_url, timeout_s=args.timeout))
    scores = [score_case(obs) for obs in observations]
    run = {
        "runnerVersion": RUNNER_VERSION,
        "implementation": "python",
        "datasetVersion": dataset["datasetVersion"],
        "datasetSha256": sha256,
        "ranking": args.arm,
        "startedAt": started_at.isoformat().replace("+00:00", "Z"),
        "finishedAt": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "observations": observations,
        "scores": scores,
        "summary": summarize_run(observations, scores),
    }
    if args.cases:
        out = args.smoke_out or Path(os.getenv("TEMP", "/tmp")) / "python-research-eval-smoke.json"
        out.write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"smoke run written to {out}")
        return 0

    date = started_at.astimezone().date().isoformat()
    out_dir = DATA_DIR / "results-python" / (args.label or f"{dataset['datasetVersion']}-{date}-py-r1")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"run-{args.arm}.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    provenance_path = out_dir / "provenance.json"
    previous = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.exists() else {}
    commits = {**previous.get("commits", {}), args.arm: _git_commit()}
    provenance = {
        "implementation": "Python (im-agent-python app/research, LangGraph StateGraph)",
        "datasetVersion": dataset["datasetVersion"],
        "datasetSha256": sha256,
        "runnerVersion": RUNNER_VERSION,
        "rankings": sorted(path.stem.removeprefix("run-") for path in out_dir.glob("run-*.json")),
        "llmProvider": os.getenv("LLM_PROVIDER", "openai-compatible"),
        "plannerModel": os.getenv("LLM_MODEL", "unknown"),
        "synthesisModel": os.getenv("LLM_MODEL", "unknown"),
        "searchProvider": "Tavily Search API (basic depth, 8 results per query)",
        "caseCount": len(dataset["cases"]),
        "date": date,
        "commit": ", ".join(f"{arm}@{commit}" for arm, commit in sorted(commits.items())),
        "commits": commits,
        "latencyScope": "wall-clock per graph invocation on a local developer machine, cases run sequentially "
        "(single concurrency), including LLM, search and fetch calls; not a production latency.",
        "notes": [
            "Executed as a local Python process (app.eval.research_runner) against live qwen and Tavily APIs.",
            "no_source cases stub the tools in the runner: empty_search returns zero search results and fails every "
            "fetch (planner-supplied URLs unreachable); fetch_fail fails every fetch. All other cases used live "
            "Tavily search and live page fetches.",
            "Live web results change over time; reruns will not reproduce identical numbers.",
            "Searches and fetches run sequentially, as in the TypeScript agent; top-5 ranked sources are fetched.",
            *(
                [
                    f"The local resolver returned fake-ip (198.18.0.0/15) answers, so the public-address check used DNS "
                    f"over HTTPS ({args.doh_url}); connections then went through the system proxy settings."
                ]
                if args.doh_url
                else []
            ),
        ],
    }
    provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(out_dir, args.reference)
    print(f"results written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
