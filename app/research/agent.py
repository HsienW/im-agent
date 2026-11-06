"""LangGraph research agent.

    plan ──research──▶ search ──▶ rank ──▶ fetch ──▶ extract ──▶ verify ──▶ synthesize
      │                  └──no results────────────────────────────▶┘          ▲
      └──direct / clarify──────────────────────────────────────────────────────┘

* plan: the LLM returns one JSON object validated by pydantic; any LLM, JSON or validation
  failure falls back to a single research query (the question itself).
* search: every planned query goes to the search tool (8 results each); planner URLs are added
  as candidates.
* rank: URL dedup and the rule score ported from chat-gun (``ranking.py``).
* fetch / extract: top-5 pages; a source is usable when the fetch had no error and the readable
  text is at least 200 characters.
* synthesize: only usable sources are numbered in the evidence; with the citation guard on, every
  ``[n]`` or URL in the answer that does not point at a usable source is removed afterwards.

Search and fetch are injected as async callables so tests can run the graph offline.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.research.citation_guard import CITATION_GUARD_POLICY, guard_citations
from app.research.ranking import ExtractedSource, RankedSource, excerpt, extract_evidence, rank_sources
from app.research.web import FetchResult, SearchResult

SearchFn = Callable[[str, str | None], Awaitable[list[SearchResult]]]
FetchFn = Callable[[str], Awaitable[FetchResult]]

MAX_QUERIES = 3
MAX_PLANNER_URLS = 5
MAX_FETCHED_SOURCES = 5
FRESHNESS_TO_TIME_RANGE = {"pd": "day", "pw": "week", "pm": "month", "py": "year"}
DEFAULT_CLARIFICATION = "Please clarify the request before I continue."


class PlannerOutput(BaseModel):
    """Structured planner output; unknown keys are ignored, wrong types fail validation."""

    model_config = ConfigDict(extra="ignore")

    question: str = ""
    answerMode: Literal["research", "direct", "clarify"]
    rationale: str = ""
    queries: list[str] = Field(default_factory=list)
    urls: list[str] = Field(default_factory=list)
    freshness: Literal["pd", "pw", "pm", "py"] | None = None
    clarification: str | None = None
    requiredSourceCount: int = 3


class ResearchState(TypedDict, total=False):
    question: str
    plan: dict[str, Any]
    search_results: list[SearchResult]
    search_errors: list[str]
    ranked_sources: list[RankedSource]
    fetched: list[tuple[RankedSource, FetchResult]]
    extracted_sources: list[ExtractedSource]
    verification: dict[str, Any]
    answer: str
    citation_guard: dict[str, Any] | None


PLANNER_POLICY = "\n".join(
    [
        "You are a research pipeline planner for a LangGraph agent.",
        "Return only one JSON object. Do not use markdown.",
        "Set question exactly to the current user request below.",
        "Classify the user request and decide whether it needs a direct answer, web research, or clarification.",
        "Use research for current facts, news, products, prices, law, regulations, company/person changes, technical "
        "docs that may have changed, and any topic requiring external evidence.",
        "Use clarify when the request cannot be researched without more information from the user, for example when "
        "the subject is missing or only referred to by a pronoun; put one short question for the user in "
        "clarification, in the user's language.",
        "For research, write up to 3 focused search queries and list official URLs only when you are sure they exist.",
        "JSON schema:",
        '{"question":"string","answerMode":"direct|research|clarify","rationale":"string","queries":["string"],'
        '"urls":["https://..."],"freshness":"pd|pw|pm|py optional","clarification":"string optional",'
        '"requiredSourceCount":3}',
    ]
)

SYNTHESIS_POLICY = [
    "You are the final synthesis node in a production-style research graph.",
    "Answer the user in the same language as the user unless they requested otherwise.",
    "Use only the provided tool results and evidence for current or researched facts.",
    "If evidence contains tool/API errors or no usable sources, state the limitation clearly and do not fabricate facts.",
    "For web research answers, cite sources inline with [1], [2], etc. using the evidence index numbers.",
    "Keep the answer concise but include exact numbers, dates, and source limitations when relevant.",
]


def _content(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return str(content)


def _json_object(raw: str) -> str | None:
    text = re.sub(r"```(?:json)?", "", raw, flags=re.I).replace("```", "").strip()
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else None


def _unique(values: list[str], limit: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        value = value.strip()
        if value and value.lower() not in seen:
            seen.add(value.lower())
            out.append(value)
        if len(out) >= limit:
            break
    return out


def fallback_plan(question: str, reason: str) -> dict[str, Any]:
    return {
        "question": question,
        "answerMode": "research",
        "rationale": f"Planner fallback: {reason}"[:300],
        "queries": [question],
        "urls": [],
        "freshness": None,
        "clarification": None,
        "requiredSourceCount": 3,
        "plannerFallback": True,
    }


def coerce_plan(raw: str, question: str) -> dict[str, Any]:
    """Validate the planner JSON; any failure falls back to one research query."""
    payload = _json_object(raw)
    if payload is None:
        return fallback_plan(question, "no JSON object in planner output")
    try:
        parsed = PlannerOutput.model_validate_json(payload)
    except ValidationError as error:
        return fallback_plan(question, f"planner output failed validation: {error.error_count()} errors")
    queries = _unique(parsed.queries, MAX_QUERIES)
    if parsed.answerMode == "research" and not queries:
        queries = [question]
    return {
        "question": question,
        "answerMode": parsed.answerMode,
        "rationale": parsed.rationale,
        "queries": queries if parsed.answerMode == "research" else [],
        "urls": _unique([url for url in parsed.urls if url.startswith(("http://", "https://"))], MAX_PLANNER_URLS)
        if parsed.answerMode == "research"
        else [],
        "freshness": parsed.freshness,
        "clarification": parsed.clarification,
        "requiredSourceCount": max(1, min(parsed.requiredSourceCount, 8)),
        "plannerFallback": False,
    }


def format_guarded_evidence(state: ResearchState) -> str:
    """Only usable sources carry an [n] index and URL; unusable ones are listed with their failure only."""
    extracted = state.get("extracted_sources", [])
    usable = [source for source in extracted if source.usable]
    unusable = [source for source in extracted if not source.usable]
    sections = []
    diagnostics = _tool_diagnostics(state)
    if diagnostics:
        sections.append(
            "Tool diagnostics (use for errors and status only; results listed here are not citable unless they "
            f"appear under 'Usable sources'):\n\n{diagnostics}"
        )
    sections.append(
        f"Web research summary: {len(state.get('search_results', []))} search results, "
        f"{len(state.get('fetched', []))} fetched, {len(usable)} usable."
    )
    if usable:
        sections.append(
            "\n\n".join(
                ["Usable sources (the only sources you may cite):"]
                + [
                    f"[{source.index}] {source.title}\nURL: {source.url}\nSummary: {source.summary}\n"
                    f"Key claims: {' | '.join(source.key_claims)}"
                    for source in usable
                ]
            )
        )
    else:
        sections.append("Usable sources: none.")
    if unusable:
        sections.append(
            "\n".join(
                ["Unavailable sources (fetch failed or content too short; do not cite them and do not use their content):"]
                + [f"- {source.title}: {excerpt(source.summary, 200)}" for source in unusable]
            )
        )
    return "\n\n".join(sections)


def format_unguarded_evidence(state: ResearchState) -> str:
    """Prompt-only arm: every fetched source is numbered with its usable flag."""
    sections = []
    diagnostics = _tool_diagnostics(state)
    if diagnostics:
        sections.append(diagnostics)
    for source in state.get("extracted_sources", []):
        sections.append(
            f"[{source.index}] {source.title}\nURL: {source.url}\nUsable: {str(source.usable).lower()}\n"
            f"Summary: {source.summary}\nKey claims: {' | '.join(source.key_claims)}"
        )
    return "\n\n".join(sections)


def _tool_diagnostics(state: ResearchState) -> str:
    lines: list[str] = []
    for error in state.get("search_errors", []):
        lines.append(f"web_search error: {error}")
    results = state.get("search_results", [])
    if state.get("plan", {}).get("answerMode") == "research":
        lines.append(f"web_search returned {len(results)} results.")
    for result in results[:16]:
        lines.append(f"- {result.title} | {result.url} | {excerpt(result.snippet, 200)}")
    for source, page in state.get("fetched", []):
        status = page.error or f"ok, {len(page.content)} characters"
        lines.append(f"web_fetch {source.url}: {status}")
    return "\n".join(lines)


def build_research_graph(
    *,
    planner_model: Any,
    synthesis_model: Any,
    search: SearchFn,
    fetch: FetchFn,
    citation_guard: bool = True,
    today: Callable[[], str] | None = None,
    max_fetched_sources: int = MAX_FETCHED_SOURCES,
):
    """Compile the research graph. Models need ``async ainvoke(prompt)`` returning a message or str."""
    current_date = today or (lambda: dt.date.today().isoformat())

    async def plan(state: ResearchState) -> ResearchState:
        question = state.get("question", "").strip()
        if not question:
            return {
                "plan": {
                    "question": "",
                    "answerMode": "clarify",
                    "rationale": "No user question was provided.",
                    "queries": [],
                    "urls": [],
                    "clarification": "Please provide a question to research.",
                    "requiredSourceCount": 1,
                    "plannerFallback": False,
                }
            }
        prompt = f"{PLANNER_POLICY}\nToday is {current_date()}.\nCurrent user request:\n{question}"
        try:
            raw = _content(await planner_model.ainvoke(prompt))
        except Exception as error:  # LLM unavailable: still research the question as one query
            return {"plan": fallback_plan(question, f"planner LLM failed: {type(error).__name__}")}
        return {"plan": coerce_plan(raw, question)}

    async def search_web(state: ResearchState) -> ResearchState:
        current = state["plan"]
        results: list[SearchResult] = []
        errors: list[str] = []
        time_range = FRESHNESS_TO_TIME_RANGE.get(current.get("freshness") or "")
        for query in current["queries"] or [current["question"]]:
            try:
                results.extend(await search(query, time_range))
            except Exception as error:
                errors.append(f"{query}: {type(error).__name__}: {error}"[:300])
        for url in current.get("urls", []):
            results.append(
                SearchResult(
                    title=url,
                    url=url,
                    snippet="URL supplied by planner or user.",
                    query=current["question"],
                    source_type="provided_url",
                )
            )
        return {"search_results": results, "search_errors": errors}

    async def rank(state: ResearchState) -> ResearchState:
        current = state["plan"]
        limit = max(current.get("requiredSourceCount", 3), max_fetched_sources)
        return {"ranked_sources": rank_sources(state["search_results"], current["question"], limit)}

    async def fetch_sources(state: ResearchState) -> ResearchState:
        fetched = []
        for source in state["ranked_sources"][:max_fetched_sources]:
            fetched.append((source, await fetch(source.url)))
        return {"fetched": fetched}

    async def extract(state: ResearchState) -> ResearchState:
        return {"extracted_sources": extract_evidence(state.get("fetched", []))}

    async def verify(state: ResearchState) -> ResearchState:
        extracted = state.get("extracted_sources", [])
        usable = sum(1 for source in extracted if source.usable)
        warnings = []
        if usable == 0:
            warnings.append(
                "No usable fetched sources are available. The final answer must explain the search/fetch failure "
                "instead of fabricating facts."
            )
        elif usable == 1:
            warnings.append("Only one usable source is available. Treat claims as weakly corroborated.")
        return {
            "verification": {
                "usableSourceCount": usable,
                "rejectedSourceCount": len(extracted) - usable,
                "warnings": warnings,
            }
        }

    async def synthesize(state: ResearchState) -> ResearchState:
        current = state["plan"]
        if current["answerMode"] == "clarify":
            return {"answer": current.get("clarification") or DEFAULT_CLARIFICATION, "citation_guard": None}
        research = current["answerMode"] == "research"
        if research:
            evidence = format_guarded_evidence(state) if citation_guard else format_unguarded_evidence(state)
        else:
            evidence = ""
        policy = SYNTHESIS_POLICY + (CITATION_GUARD_POLICY if citation_guard else []) + [f"Today is {current_date()}."]
        public_plan = {key: value for key, value in current.items() if key != "plannerFallback"}
        prompt = "\n\n".join(
            [
                "\n\n".join(policy),
                f"Plan: {json.dumps(public_plan, ensure_ascii=False, indent=2)}",
                f"Verification: {json.dumps(state.get('verification', {}), ensure_ascii=False, indent=2)}",
                "Evidence:",
                evidence or "No external evidence was collected.",
                "Conversation:",
                current["question"],
            ]
        )
        try:
            answer = _content(await synthesis_model.ainvoke(prompt))
        except Exception as error:
            return {"answer": f"Research synthesis failed: {type(error).__name__}: {error}"[:500], "citation_guard": None}
        if not (citation_guard and research):
            return {"answer": answer, "citation_guard": None}
        guarded, report = guard_citations(answer, state.get("extracted_sources", []))
        return {"answer": guarded, "citation_guard": asdict(report)}

    def after_plan(state: ResearchState) -> str:
        return "search" if state["plan"]["answerMode"] == "research" else "synthesize"

    def after_search(state: ResearchState) -> str:
        return "rank" if state["search_results"] else "verify"

    builder = StateGraph(ResearchState)
    builder.add_node("plan", plan)
    builder.add_node("search", search_web)
    builder.add_node("rank", rank)
    builder.add_node("fetch", fetch_sources)
    builder.add_node("extract", extract)
    builder.add_node("verify", verify)
    builder.add_node("synthesize", synthesize)
    builder.add_edge(START, "plan")
    builder.add_conditional_edges("plan", after_plan, {"search": "search", "synthesize": "synthesize"})
    builder.add_conditional_edges("search", after_search, {"rank": "rank", "verify": "verify"})
    builder.add_edge("rank", "fetch")
    builder.add_edge("fetch", "extract")
    builder.add_edge("extract", "verify")
    builder.add_edge("verify", "synthesize")
    builder.add_edge("synthesize", END)
    return builder.compile()


def to_observation(state: ResearchState) -> dict[str, Any]:
    """The per-case trace in chat-gun's ``run-*.json`` observation shape (minus runner fields)."""
    current = state.get("plan", {})
    extracted = state.get("extracted_sources", [])
    trace: dict[str, Any] = {}
    guard = state.get("citation_guard")
    if guard is not None:
        trace["citationGuard"] = {
            "modified": guard["modified"],
            "removedMarkers": len(guard["removed_markers"]),
            "removedUrls": len(guard["removed_urls"]),
        }
    trace.update(
        {
            "answerMode": current.get("answerMode"),
            "plannerFallback": bool(current.get("plannerFallback")),
            "queries": list(current.get("queries", [])),
            "searchResultCount": len(state.get("search_results", [])),
            "rankedUrls": [source.url for source in state.get("ranked_sources", [])],
            "sources": [
                {
                    "index": source.index,
                    "url": source.url,
                    "usable": source.usable,
                    **({"fetchError": source.fetch_error[:200]} if source.fetch_error else {}),
                }
                for source in extracted
            ],
            "answer": state.get("answer", ""),
        }
    )
    return trace
