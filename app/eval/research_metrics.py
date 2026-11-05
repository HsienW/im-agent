"""Deep research evaluation metrics, ported from chat-gun's TypeScript scorer.

The TypeScript runner (chat-gun ``backend/src/evaluation/research/scoring.ts``) executes the
Deep Researcher graph and stores every observation in ``run-<ranking>.json``. This module
re-scores those observations independently so the metric definitions can be checked in a
second language:

    python -m app.eval.research_metrics path/to/run-rule.json --check

Definitions
-----------
* Citation verification pass: an answer that cites at least once passes only when every
  ``[n]`` marker resolves to an extracted source marked usable and every raw URL belongs to a
  usable source. Answers without citations are counted separately.
* No-source refusal pass: zero usable sources, the answer states the limitation, and it cites
  no ``[n]`` evidence marker (raw URLs are listed for manual review, not failed).
* Clarify pass: the planner routed the request to ``answerMode=clarify``.
* Latency: wall-clock per graph invocation; nearest-rank percentiles.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

MAX_SOURCE_INDEX = 99
CATEGORIES = ("answerable", "no_source", "clarify")

_MARKER = re.compile(r"\[([0-9\s,，、\-–]+)\]")
_RANGE = re.compile(r"^([0-9]+)\s*[-–]\s*([0-9]+)$")
# Full-width punctuation ends a URL: answers often write "https://host/path（說明）".
_URL = re.compile(r"https?://[^\s<>\"'`）)\]（，。、；：！？「」【】《》]+")
_TRAILING = re.compile(r"[)\]>.,;:!?'\"，。）]+$")
_DEFAULT_PORTS = {"http": 80, "https": 443}

LIMITATION_PATTERNS = [
    re.compile(r"no usable (?:sources?|evidence|results?)", re.I),
    re.compile(r"no (?:relevant |search )?results", re.I),
    re.compile(
        r"(?:could not|couldn['’]t|cannot|can['’]t|unable to|was unable to|were unable to|not able to) "
        r"(?:be )?(?:find|retriev|fetch|access|locat|verif|confirm|provid|report|answer|determin)",
        re.I,
    ),
    re.compile(r"(?:search|fetch|retrieval|tool)[^.\n]{0,40}(?:fail|error|unavailable|returned no)", re.I),
    re.compile(r"\bfailed to\b", re.I),
    re.compile(r"insufficient (?:evidence|sources|information)", re.I),
    re.compile(
        r"無法|无法|未能|沒有(?:找到|可用|取得|檢索|搜尋到)|没有(?:找到|可用|取得|检索|搜索到)"
        r"|未(?:找到|檢索到|检索到|取得)|(?:搜尋|搜索|檢索|检索|抓取|擷取|获取|取得)[^。\n]{0,12}(?:失敗|失败|為空|为空)"
    ),
]


def extract_citation_markers(answer: str) -> list[int]:
    markers: set[int] = set()
    for match in _MARKER.finditer(answer):
        for part in re.split(r"[,，、]", match.group(1)):
            part = part.strip()
            ranged = _RANGE.match(part)
            if ranged:
                start, end = int(ranged.group(1)), int(ranged.group(2))
                if end >= start and end - start <= 20:
                    markers.update(range(start, end + 1))
                continue
            if part.isascii() and part.isdigit():
                markers.add(int(part))
    return sorted(value for value in markers if 1 <= value <= MAX_SOURCE_INDEX)


def normalize_url(raw: str) -> str:
    """Mirror WHATWG URL normalization for the parts the scorer compares."""
    trimmed = _TRAILING.sub("", raw)
    try:
        parts = urlsplit(trimmed)
        if not parts.scheme or not parts.hostname:
            return trimmed
        port = parts.port
    except ValueError:
        return trimmed
    host = parts.hostname.lower()
    if port is not None and port != _DEFAULT_PORTS.get(parts.scheme.lower()):
        host = f"{host}:{port}"
    path = quote(parts.path, safe="/%:@!$&'()*+,;=-._~").rstrip("/")
    query = "?" + quote(parts.query, safe="=&%:@!$/?()*+,;-._~") if parts.query else ""
    return f"{parts.scheme.lower()}://{host}{path}{query}"


def extract_urls(answer: str) -> list[str]:
    seen: dict[str, None] = {}
    for match in _URL.finditer(answer):
        seen.setdefault(normalize_url(match.group(0)), None)
    return list(seen)


def score_citations(answer: str, sources: list[dict[str, Any]]) -> dict[str, Any]:
    markers = extract_citation_markers(answer)
    urls = extract_urls(answer)
    by_index = {source["index"]: source for source in sources}
    usable_urls = {normalize_url(source["url"]) for source in sources if source["usable"]}
    unresolved: list[str] = []
    usable_cited = 0
    for marker in markers:
        if by_index.get(marker, {}).get("usable"):
            usable_cited += 1
        else:
            unresolved.append(f"[{marker}]")
    for url in urls:
        if url in usable_urls:
            usable_cited += 1
        else:
            unresolved.append(url)
    cited = len(markers) + len(urls)
    return {
        "markers": markers,
        "urls": urls,
        "citedCount": cited,
        "usableCitedCount": usable_cited,
        "unresolved": unresolved,
        "pass": None if cited == 0 else not unresolved,
    }


def states_limitation(answer: str) -> bool:
    return any(pattern.search(answer) for pattern in LIMITATION_PATTERNS)


def score_case(observation: dict[str, Any]) -> dict[str, Any]:
    citation = score_citations(observation["answer"], observation["sources"])
    score: dict[str, Any] = {"caseId": observation["caseId"], "category": observation["category"], "citation": citation}
    completed = observation["status"] == "completed"
    if observation["category"] == "no_source":
        usable = sum(1 for source in observation["sources"] if source["usable"])
        limitation = states_limitation(observation["answer"])
        cited_evidence = bool(citation["markers"])
        score["refusal"] = {
            "usableSourceCount": usable,
            "limitationStated": limitation,
            "citedEvidence": cited_evidence,
            "urlsMentioned": citation["urls"],
            "pass": completed and usable == 0 and limitation and not cited_evidence,
        }
    if observation["category"] == "clarify":
        score["clarifyPass"] = completed and observation.get("answerMode") == "clarify"
    return score


def nearest_rank_percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile / 100 * len(ordered)))
    return ordered[rank - 1]


def summarize_latency(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "p50Ms": nearest_rank_percentile(values, 50),
        "p90Ms": nearest_rank_percentile(values, 90),
        # JavaScript Math.round rounds halves up; Python's round() would round half to even.
        "meanMs": math.floor(sum(values) / len(values) + 0.5) if values else None,
    }


def _rate(passed: int, n: int) -> dict[str, Any]:
    return {"passed": passed, "n": n, "rate": None if n == 0 else passed / n}


def summarize_run(observations: list[dict[str, Any]], scores: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {score["caseId"]: score for score in scores}
    completed = [obs for obs in observations if obs["status"] == "completed"]
    answerable = [obs for obs in completed if obs["category"] == "answerable"]
    citing = [by_id[obs["caseId"]] for obs in answerable if by_id[obs["caseId"]]["citation"]["pass"] is not None]
    all_citing = [by_id[obs["caseId"]] for obs in completed if by_id[obs["caseId"]]["citation"]["pass"] is not None]
    no_source = [by_id[obs["caseId"]]["refusal"] for obs in observations if obs["category"] == "no_source"]
    clarify = [obs for obs in observations if obs["category"] == "clarify"]
    fetched = [source for obs in answerable for source in obs["sources"]]

    routing: dict[str, int] = {}
    for obs in completed:
        key = f"{obs['category']}:{obs.get('answerMode') or 'unknown'}"
        routing[key] = routing.get(key, 0) + 1

    latency = {"all": summarize_latency([obs["latencyMs"] for obs in completed])}
    for category in CATEGORIES:
        latency[category] = summarize_latency([obs["latencyMs"] for obs in completed if obs["category"] == category])

    summary: dict[str, Any] = {}
    reranked = [obs for obs in completed if obs.get("rerank") is not None]
    if reranked:
        ok = [obs for obs in reranked if obs["rerank"]["ok"]]
        summary["rerank"] = {
            "calls": len(reranked),
            "failed": len(reranked) - len(ok),
            "latency": summarize_latency([obs["rerank"]["latencyMs"] for obs in reranked]),
            "casesWithChangedSelection": _rate(
                sum(1 for obs in reranked if (obs["rerank"].get("changedSelections") or 0) > 0), len(ok)
            ),
        }
    summary.update(
        {
            "cases": len(observations),
            "completed": len(completed),
            "failedOrTimeout": len(observations) - len(completed),
            "citation": {
                "answerPassRate": _rate(sum(1 for s in citing if s["citation"]["pass"]), len(citing)),
                "citationPrecision": _rate(
                    sum(s["citation"]["usableCitedCount"] for s in citing), sum(s["citation"]["citedCount"] for s in citing)
                ),
                "answerableWithoutCitation": sum(
                    1
                    for obs in answerable
                    if by_id[obs["caseId"]]["citation"]["pass"] is None and any(src["usable"] for src in obs["sources"])
                ),
                "answerableWithZeroUsableSources": sum(
                    1 for obs in answerable if not any(src["usable"] for src in obs["sources"])
                ),
                "allCategoriesAnswerPassRate": _rate(sum(1 for s in all_citing if s["citation"]["pass"]), len(all_citing)),
            },
            "refusal": {
                **_rate(sum(1 for r in no_source if r["pass"]), len(no_source)),
                "stubEffective": sum(1 for r in no_source if r["usableSourceCount"] == 0),
            },
            "clarify": _rate(sum(1 for obs in clarify if by_id[obs["caseId"]]["clarifyPass"]), len(clarify)),
            "fetchedUsable": _rate(sum(1 for src in fetched if src["usable"]), len(fetched)),
            "routing": routing,
            "latency": latency,
        }
    )
    return summary


def rescore(run: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scores = [score_case(obs) for obs in run["observations"]]
    return scores, summarize_run(run["observations"], scores)


def _diff(expected: Any, actual: Any, path: str = "") -> list[str]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        out: list[str] = []
        for key in sorted(set(expected) | set(actual)):
            out += _diff(expected.get(key), actual.get(key), f"{path}.{key}")
        return out
    if isinstance(expected, list) and isinstance(actual, list) and len(expected) == len(actual):
        return [line for i, (e, a) in enumerate(zip(expected, actual)) for line in _diff(e, a, f"{path}[{i}]")]
    if isinstance(expected, float) or isinstance(actual, float):
        if expected is not None and actual is not None and math.isclose(expected, actual, rel_tol=1e-12):
            return []
    return [] if expected == actual else [f"{path}: typescript={expected!r} python={actual!r}"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run", type=Path, help="run-<ranking>.json written by the TypeScript runner")
    parser.add_argument("--check", action="store_true", help="fail if any score or summary differs from the TypeScript result")
    args = parser.parse_args(argv)
    run = json.loads(args.run.read_text(encoding="utf-8"))
    scores, summary = rescore(run)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not args.check:
        return 0
    differences = _diff(run["scores"], scores, "scores") + _diff(run["summary"], summary, "summary")
    for line in differences:
        print(line, file=sys.stderr)
    print(f"parity: {'OK' if not differences else f'{len(differences)} differences'} over {len(scores)} cases", file=sys.stderr)
    return 0 if not differences else 1


if __name__ == "__main__":
    raise SystemExit(main())
