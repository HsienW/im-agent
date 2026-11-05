import json
from pathlib import Path

import pytest

from app.eval.research_metrics import (
    extract_citation_markers,
    extract_urls,
    main,
    nearest_rank_percentile,
    rescore,
    score_case,
    score_citations,
    states_limitation,
    summarize_latency,
)

RUN_DIR = Path(__file__).resolve().parents[1] / "data" / "research-eval"
SOURCES = [
    {"index": 1, "url": "https://example.com/a", "usable": True},
    {"index": 2, "url": "https://example.org/b/", "usable": True},
    {"index": 3, "url": "https://example.net/c", "usable": False, "fetchError": "Error: 404"},
]


def test_markers_lists_ranges_and_years():
    assert extract_citation_markers("A [1][2], B [1, 3], C [4-6], 2026 [2026]") == [1, 2, 3, 4, 5, 6]
    assert extract_citation_markers("依據來源 [2]、[3]，以及 [1，2]") == [1, 2, 3]
    assert extract_citation_markers("full-width digits [１] are not markers") == []


def test_url_ends_at_full_width_punctuation():
    assert extract_urls("官方頁面 https://kubernetes.io/releases/（官方發布日曆）與 https://github.com/k/k/releases，原始") == [
        "https://kubernetes.io/releases",
        "https://github.com/k/k/releases",
    ]


def test_urls_are_normalized_like_whatwg():
    assert extract_urls("See (https://Example.org/b/). Also [x](https://example.com/a#top)") == [
        "https://example.org/b",
        "https://example.com/a",
    ]
    assert extract_urls("https://example.com:443/x and http://example.com:8080/y?q=1") == [
        "https://example.com/x",
        "http://example.com:8080/y?q=1",
    ]


@pytest.mark.parametrize(
    ("answer", "expected_pass", "unresolved"),
    [
        ("Fact [1]. Other fact [2] https://example.org/b", True, []),
        ("Fact [1] and [3]", False, ["[3]"]),
        ("Fact [9]", False, ["[9]"]),
        ("See https://unknown.example/x", False, ["https://unknown.example/x"]),
        ("No citations here.", None, []),
    ],
)
def test_citation_pass_requires_usable_sources(answer, expected_pass, unresolved):
    score = score_citations(answer, SOURCES)
    assert score["pass"] is expected_pass
    assert score["unresolved"] == unresolved


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("No usable sources were retrieved, so I cannot confirm the figure.", True),
        ("The web search returned no results for this query.", True),
        ("目前搜尋結果為空，無法提供最新數據。", True),
        ("由于抓取失败，未能获取国家统计局的数据。", True),
        ("The federal funds rate target range is 4.25% to 4.50%.", False),
    ],
)
def test_limitation_detector(answer, expected):
    assert states_limitation(answer) is expected


def _no_source(answer, sources):
    return {
        "caseId": "nos-01", "category": "no_source", "stub": "empty_search", "status": "completed",
        "latencyMs": 10, "queries": [], "searchResultCount": 0, "rankedUrls": [], "sources": sources, "answer": answer,
    }


def test_refusal_fails_on_evidence_marker_or_leftover_source():
    failed = [{"index": 1, "url": "https://example.com/a", "usable": False}]
    assert score_case(_no_source("No usable sources were found [1].", failed))["refusal"]["pass"] is False
    assert score_case(_no_source("No usable sources were found.", failed))["refusal"]["pass"] is True
    pointer = score_case(_no_source("無法取得資料，請查閱 https://www.cbc.gov.tw", failed))["refusal"]
    assert pointer["pass"] is True and pointer["urlsMentioned"] == ["https://www.cbc.gov.tw"]
    usable = [{"index": 1, "url": "https://example.com/a", "usable": True}]
    assert score_case(_no_source("I could not find sources.", usable))["refusal"]["pass"] is False


def test_nearest_rank_and_js_rounding():
    values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    assert nearest_rank_percentile(values, 50) == 50
    assert nearest_rank_percentile(values, 90) == 90
    assert nearest_rank_percentile([], 50) is None
    # JavaScript Math.round(2.5) is 3; Python's round(2.5) would give 2.
    assert summarize_latency([2, 3])["meanMs"] == 3


RUN_FILES = sorted(RUN_DIR.glob("*/run-*.json"))


@pytest.mark.skipif(not RUN_FILES, reason="no recorded TypeScript run under data/research-eval")
@pytest.mark.parametrize("run_file", RUN_FILES, ids=lambda path: f"{path.parent.name}/{path.name}")
def test_parity_with_recorded_typescript_run(run_file):
    run = json.loads(run_file.read_text(encoding="utf-8"))
    scores, summary = rescore(run)
    assert scores == run["scores"]
    assert summary == run["summary"]
    assert main([str(run_file), "--check"]) == 0
