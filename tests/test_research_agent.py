import json

import httpx
import pytest
from langchain_core.messages import AIMessage

from app.eval.research_metrics import score_case
from app.research.agent import build_research_graph, coerce_plan, to_observation
from app.research.citation_guard import guard_citations
from app.research.ranking import (
    RankedSource,
    extract_evidence,
    normalize_url_for_dedup,
    rank_sources,
    source_score,
)
from app.research.web import (
    EgressError,
    FetchResult,
    SearchResult,
    check_url,
    doh_resolver,
    fetch_page,
    html_to_text,
    tavily_search,
)

LONG_TEXT = "LangGraph checkpointers persist graph state between steps. " * 10
PUBLIC_IP = "93.184.215.14"


class FakeModel:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.prompts = []

    async def ainvoke(self, prompt):
        self.prompts.append(prompt)
        response = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(response, Exception):
            raise response
        return AIMessage(content=response)


async def public_resolver(host):
    return [PUBLIC_IP]


def plan_json(**overrides):
    plan = {
        "question": "q",
        "answerMode": "research",
        "rationale": "needs sources",
        "queries": ["langgraph checkpointer"],
        "urls": [],
        "requiredSourceCount": 3,
    }
    plan.update(overrides)
    return json.dumps(plan)


def result(url, title="LangGraph checkpointer docs", snippet="checkpointer persistence", **kwargs):
    return SearchResult(title=title, url=url, snippet=snippet, query="q", **kwargs)


def make_search(results_by_query=None, calls=None):
    async def search(query, time_range):
        if calls is not None:
            calls.append((query, time_range))
        return list((results_by_query or {}).get(query, []))

    return search


def make_fetch(pages):
    async def fetch(url):
        page = pages.get(url)
        if page is None:
            return FetchResult(url, error="Error: web_fetch failed - 404 Not Found")
        return FetchResult(url, content=page)

    return fetch


# --------------------------------------------------------------------------- planner


@pytest.mark.parametrize(
    "raw",
    [
        "Sure! I will research that.",
        '{"answerMode": "weather", "queries": ["x"]}',
        '{"answerMode": "research", "queries": "not a list"}',
    ],
)
def test_planner_falls_back_to_single_query(raw):
    plan = coerce_plan(raw, "What is MCP?")
    assert plan["answerMode"] == "research"
    assert plan["queries"] == ["What is MCP?"]
    assert plan["plannerFallback"] is True


def test_planner_validates_and_trims_queries():
    raw = "```json\n" + plan_json(queries=["a", "A ", "b", "c", "d"], urls=["https://x.org", "ftp://y"]) + "\n```"
    plan = coerce_plan(raw, "What is MCP?")
    assert plan["queries"] == ["a", "b", "c"]
    assert plan["urls"] == ["https://x.org"]
    assert plan["question"] == "What is MCP?"
    assert plan["plannerFallback"] is False


def test_planner_ignores_invalid_optional_fields():
    raw = plan_json(freshness="", clarification="", requiredSourceCount="many", queries=["rrf k constant"])
    plan = coerce_plan(raw, "What is RRF?")
    assert plan["plannerFallback"] is False
    assert plan["queries"] == ["rrf k constant"]
    assert plan["freshness"] is None and plan["clarification"] is None
    assert plan["requiredSourceCount"] == 3


@pytest.mark.asyncio
async def test_planner_llm_error_still_researches_question():
    calls = []
    graph = build_research_graph(
        planner_model=FakeModel(RuntimeError("gateway down")),
        synthesis_model=FakeModel("No usable source was obtained."),
        search=make_search(calls=calls),
        fetch=make_fetch({}),
    )
    state = await graph.ainvoke({"question": "What is MCP?"})
    assert state["plan"]["plannerFallback"] is True
    assert calls == [("What is MCP?", None)]


# --------------------------------------------------------------------------- ranking


def test_dedup_normalizes_tracking_params_fragment_and_slash():
    assert normalize_url_for_dedup("HTTPS://Docs.Example.com:443/a/?utm_source=x&b=1#top") == "https://docs.example.com/a/?b=1"
    assert normalize_url_for_dedup("https://example.com/a/") == "https://example.com/a"
    assert normalize_url_for_dedup("https://example.com") == "https://example.com"


def test_rank_dedups_and_orders_by_rule_score():
    question = "What does the LangGraph checkpointer do?"
    results = [
        result("http://blog.example.com/post", title="Unrelated", snippet="nothing"),
        result("https://docs.langchain.com/checkpointer/", age="2026-01-01"),
        result("https://docs.langchain.com/checkpointer?utm_source=feed"),
        result("https://example.com/langgraph", title="LangGraph overview", snippet="graph"),
    ]
    ranked = rank_sources(results, question, limit=5)
    assert [source.url for source in ranked] == [
        "https://docs.langchain.com/checkpointer",
        "https://example.com/langgraph",
        "http://blog.example.com/post",
    ]
    # 2 term hits x 8 + https 4 + age 2 + "docs." 5
    assert source_score(results[1], question) == (27, "2 query term hits, freshness: 2026-01-01")


def test_usable_requires_no_error_and_200_characters():
    ranked = [RankedSource(result(f"https://e.com/{i}"), 0, "") for i in range(3)]
    fetched = [
        (ranked[0], FetchResult("https://e.com/0", content=LONG_TEXT)),
        (ranked[1], FetchResult("https://e.com/1", content="too short")),
        (ranked[2], FetchResult("https://e.com/2", content=LONG_TEXT, error="Error: web_fetch failed - 503")),
    ]
    sources = extract_evidence(fetched)
    assert [(source.index, source.usable) for source in sources] == [(1, True), (2, False), (3, False)]
    assert sources[2].fetch_error.endswith("503")


# --------------------------------------------------------------------------- citation guard


class Src:
    def __init__(self, index, url, usable):
        self.index, self.url, self.usable = index, url, usable


def test_guard_removes_unusable_markers_and_unknown_urls():
    sources = [Src(1, "https://a.com/x", True), Src(2, "https://b.com", False), Src(3, "https://c.com/", True)]
    answer = "Fact [1][2]. More [2-3]. See [docs](https://evil.com/p) and https://c.com.\n- https://b.com\n官網（https://evil.com）。"
    guarded, report = guard_citations(answer, sources)
    assert guarded == "Fact [1]. More [3]. See docs and https://c.com.\n官網。"
    assert report.modified is True
    assert report.removed_markers == [2]
    assert set(report.removed_urls) == {"https://evil.com/p", "https://b.com", "https://evil.com"}


def test_guard_leaves_valid_answers_untouched():
    sources = [Src(1, "https://a.com/x", True)]
    answer = "Fact [1] https://a.com/x/ and year [2026]."
    guarded, report = guard_citations(answer, sources)
    assert guarded == answer
    assert report.modified is False and report.removed_markers == [] and report.removed_urls == []


# --------------------------------------------------------------------------- graph paths


@pytest.mark.asyncio
async def test_research_path_guards_citations_and_scores_as_pass():
    synthesis = FakeModel("Checkpointers persist state [1]. Old docs [2]. Made up [7] https://made.up/x")
    graph = build_research_graph(
        planner_model=FakeModel(plan_json(queries=["langgraph checkpointer"], urls=["https://langchain.com/cp"])),
        synthesis_model=synthesis,
        search=make_search(
            {
                "langgraph checkpointer": [
                    result("https://docs.langchain.com/cp"),
                    result("https://docs.langchain.com/cp/"),
                    result("https://old.example.com/cp"),
                ]
            }
        ),
        fetch=make_fetch({"https://docs.langchain.com/cp": LONG_TEXT}),
    )
    state = await graph.ainvoke({"question": "What does the LangGraph checkpointer do?"})
    trace = to_observation(state)

    assert trace["searchResultCount"] == 4
    assert len(trace["rankedUrls"]) == 3
    assert [source["usable"] for source in trace["sources"]] == [True, False, False]
    assert trace["answer"] == "Checkpointers persist state [1]. Old docs. Made up"
    assert trace["citationGuard"] == {"modified": True, "removedMarkers": 2, "removedUrls": 1}
    prompt = synthesis.prompts[0]
    assert "Usable sources (the only sources you may cite):\n\n[1]" in prompt
    assert "[2] " not in prompt.split("Unavailable sources")[0].split("Usable sources (")[1]

    observation = {"caseId": "ans-x", "category": "answerable", "status": "completed", "latencyMs": 1, **trace}
    assert score_case(observation)["citation"]["pass"] is True


@pytest.mark.asyncio
async def test_no_source_path_skips_fetch_and_reports_no_usable_sources():
    synthesis = FakeModel("無法取得可用來源：搜尋沒有結果。")
    fetched = []

    async def fetch(url):
        fetched.append(url)
        return FetchResult(url, error="should not be called")

    graph = build_research_graph(
        planner_model=FakeModel(plan_json()),
        synthesis_model=synthesis,
        search=make_search({}),
        fetch=fetch,
    )
    state = await graph.ainvoke({"question": "2026 年 10 月中国 CPI 同比涨幅是多少？"})
    trace = to_observation(state)
    assert fetched == []
    assert trace["sources"] == [] and trace["rankedUrls"] == []
    assert "Usable sources: none." in synthesis.prompts[0]
    assert state["verification"]["usableSourceCount"] == 0
    observation = {"caseId": "nos-x", "category": "no_source", "status": "completed", "latencyMs": 1, **trace}
    assert score_case(observation)["refusal"]["pass"] is True


@pytest.mark.asyncio
async def test_fetch_failures_mark_every_source_unusable():
    graph = build_research_graph(
        planner_model=FakeModel(plan_json()),
        synthesis_model=FakeModel("Every fetch failed, so no usable source was obtained [1]."),
        search=make_search({"langgraph checkpointer": [result(f"https://e{i}.com/a") for i in range(7)]}),
        fetch=make_fetch({}),
    )
    trace = to_observation(await graph.ainvoke({"question": "q"}))
    assert len(trace["sources"]) == 5
    assert not any(source["usable"] for source in trace["sources"])
    assert "[1]" not in trace["answer"]


@pytest.mark.asyncio
async def test_clarify_path_never_searches():
    calls = []
    synthesis = FakeModel("should not be called")
    graph = build_research_graph(
        planner_model=FakeModel(plan_json(answerMode="clarify", queries=[], clarification="你指的是哪一隻基金？")),
        synthesis_model=synthesis,
        search=make_search(calls=calls),
        fetch=make_fetch({}),
    )
    state = await graph.ainvoke({"question": "帮我查一下那只基金最近表现怎么样。"})
    assert calls == [] and synthesis.prompts == []
    assert state["answer"] == "你指的是哪一隻基金？"
    assert to_observation(state)["answerMode"] == "clarify"
    assert "citationGuard" not in to_observation(state)


# --------------------------------------------------------------------------- web tools


def test_html_to_text_prefers_main_content_and_drops_chrome():
    html = (
        "<html><head><title>Docs &amp; Guide</title><meta name='description' content='About'></head>"
        "<body><nav>Menu</nav><header>Top</header><main><h1>Title</h1><p>Hello&nbsp;<b>world</b></p>"
        "<script>alert(1)</script><style>p{}</style></main><footer>Footer</footer></body></html>"
    )
    assert html_to_text(html) == ("Title Hello world", "Docs & Guide", "About")


@pytest.mark.asyncio
async def test_tavily_search_maps_results():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "query": "mcp",
                "results": [
                    {"title": "MCP", "url": "https://modelcontextprotocol.io", "content": "spec", "published_date": "2025"},
                    {"title": "no url", "url": ""},
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        results = await tavily_search(client, "test-key", "mcp", time_range="week")
    assert seen["auth"] == "Bearer test-key"
    assert seen["body"]["max_results"] == 8 and seen["body"]["search_depth"] == "basic"
    assert seen["body"]["time_range"] == "week"
    assert results == [SearchResult("MCP", "https://modelcontextprotocol.io", "spec", "mcp", age="2025")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("url", "message"),
    [
        ("ftp://example.com/x", "scheme not allowed"),
        ("https://example.com:8443/x", "port not allowed"),
        ("http://127.0.0.1/admin", "non-public address"),
        ("http://[::1]/", "non-public address"),
        ("http://169.254.169.254/latest/meta-data", "non-public address"),
        ("http://localhost/", "host not allowed"),
        ("https://user:pw@example.com/", "credentials"),
    ],
)
async def test_egress_policy_rejects(url, message):
    with pytest.raises(EgressError, match=message):
        await check_url(url, public_resolver)


@pytest.mark.asyncio
async def test_egress_policy_rejects_private_and_fake_ip_dns_answers():
    for answer in ("10.0.0.5", "192.168.1.1", "198.18.0.7", "100.64.0.1", "::ffff:127.0.0.1"):

        async def resolver(host, answer=answer):
            return [PUBLIC_IP, answer]

        with pytest.raises(EgressError, match="non-public"):
            await check_url("https://example.com/", resolver)
    await check_url("https://example.com/", public_resolver)


@pytest.mark.asyncio
async def test_fetch_page_converts_html_and_checks_redirect_targets():
    def handler(request):
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "/doc"})
        if request.url.path == "/doc":
            return httpx.Response(200, headers={"content-type": "text/html; charset=utf-8"}, text=f"<main><p>{LONG_TEXT}</p></main>")
        if request.url.path == "/to-private":
            return httpx.Response(302, headers={"location": "http://10.0.0.1/secret"})
        if request.url.path == "/pdf":
            return httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF")
        if request.url.path == "/big":
            return httpx.Response(200, headers={"content-type": "text/plain"}, content=b"x" * 5000)
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        ok = await fetch_page(client, "https://example.com/start", resolver=public_resolver)
        blocked = await fetch_page(client, "https://example.com/to-private", resolver=public_resolver)
        pdf = await fetch_page(client, "https://example.com/pdf", resolver=public_resolver)
        big = await fetch_page(client, "https://example.com/big", resolver=public_resolver, max_bytes=1000)
        missing = await fetch_page(client, "https://example.com/nope", resolver=public_resolver)

    assert ok.error is None and ok.content == LONG_TEXT.strip()
    assert blocked.error.startswith("Error: web_fetch blocked") and "10.0.0.1" in blocked.error
    assert "unsupported content type" in pdf.error
    assert "larger than 1000 bytes" in big.error
    assert missing.error == "Error: web_fetch failed - 404 Not Found"


@pytest.mark.asyncio
async def test_doh_resolver_reads_a_and_aaaa_answers():
    def handler(request):
        record = request.url.params["type"]
        answers = [{"type": 1, "data": PUBLIC_IP}] if record == "A" else [{"type": 5, "data": "alias."}]
        return httpx.Response(200, json={"Answer": answers})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        resolve = doh_resolver("https://223.5.5.5/resolve", client)
        assert await resolve("example.com") == [PUBLIC_IP]


# --------------------------------------------------------------------------- eval runner


def test_dataset_hash_matches_typescript_loader():
    from app.eval.research_runner import DATA_DIR, load_dataset

    dataset, sha256 = load_dataset(DATA_DIR / "research-eval-v1.json")
    recorded = json.loads((DATA_DIR / "research-eval-v1-2026-11-05-r1" / "run-guard.json").read_text(encoding="utf-8"))
    assert len(dataset["cases"]) == 26
    assert sha256 == recorded["datasetSha256"]


def test_report_renders_recorded_run(tmp_path):
    from app.eval.research_runner import DATA_DIR, write_report

    recorded = DATA_DIR / "research-eval-v1-2026-11-05-r1" / "run-guard.json"
    (tmp_path / "run-guard.json").write_text(recorded.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "provenance.json").write_text(
        json.dumps(
            {
                "datasetVersion": "research-eval-v1",
                "date": "2026-11-05",
                "caseCount": 26,
                "llmProvider": "qwen",
                "plannerModel": "qwen-plus",
                "searchProvider": "Tavily",
                "commit": "guard@test",
                "latencyScope": "local, sequential",
            }
        ),
        encoding="utf-8",
    )
    write_report(tmp_path, [recorded])
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "| Metric | python:guard | ts:guard |" in report
    assert "| Citation verification pass rate | 100.0% (14/14) | 100.0% (14/14) |" in report
    assert "| Latency P50 / P90 — all | 23.8s / 42.8s (n=26) | 23.8s / 42.8s (n=26) |" in report


@pytest.mark.asyncio
async def test_direct_answers_are_guarded_against_invented_citations():
    calls = []
    synthesis = FakeModel("HTTP 429 means Too Many Requests [1]; wait per Retry-After [2] (https://made.up/rfc).")
    graph = build_research_graph(
        planner_model=FakeModel(plan_json(answerMode="direct", queries=[], freshness="")),
        synthesis_model=synthesis,
        search=make_search(calls=calls),
        fetch=make_fetch({}),
    )
    trace = to_observation(await graph.ainvoke({"question": "What does HTTP 429 mean?"}))
    assert calls == [] and trace["answerMode"] == "direct"
    assert trace["answer"] == "HTTP 429 means Too Many Requests; wait per Retry-After."
    assert trace["citationGuard"] == {"modified": True, "removedMarkers": 2, "removedUrls": 1}
    assert "Do not use [n] citation markers" in synthesis.prompts[0]
    assert "Evidence:" not in synthesis.prompts[0]
