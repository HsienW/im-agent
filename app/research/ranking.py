"""Rule-based source ranking, ported from chat-gun ``deep-researcher.ts``.

``source_score`` / ``rank_sources`` / ``extract_evidence`` keep the TypeScript scoring weights,
the dedup rule and the usability threshold so the two agents are evaluated on the same rules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from app.research.web import FetchResult, SearchResult

MIN_USABLE_CHARS = 200
_TERM_SPLIT = re.compile(r"[\W_]+")
_AUTHORITY = re.compile(r"\.gov|\.edu|\.org|docs\.|developer\.|official", re.I)
_DEFAULT_PORTS = {"http": 80, "https": 443}


@dataclass(frozen=True)
class RankedSource:
    result: SearchResult
    score: int
    reason: str

    @property
    def url(self) -> str:
        return self.result.url


@dataclass(frozen=True)
class ExtractedSource:
    index: int
    title: str
    url: str
    summary: str
    key_claims: list[str] = field(default_factory=list)
    usable: bool = False
    fetch_error: str | None = None


def normalize_url_for_dedup(raw: str) -> str:
    """Drop the fragment, utm_* and fbclid parameters and a trailing slash."""
    try:
        parts = urlsplit(raw.strip())
        port = parts.port
    except ValueError:
        return raw.strip()
    if not parts.scheme or not parts.hostname:
        return raw.strip()
    scheme = parts.scheme.lower()
    host = parts.hostname.lower()
    if port is not None and port != _DEFAULT_PORTS.get(scheme):
        host = f"{host}:{port}"
    query = parts.query
    if query:
        pairs = parse_qsl(query, keep_blank_values=True)
        kept = [(key, value) for key, value in pairs if not (key.lower().startswith("utm_") or key.lower() == "fbclid")]
        if len(kept) != len(pairs):
            query = urlencode(kept)
    return urlunsplit((scheme, host, parts.path or "/", query, "")).rstrip("/")


def source_score(result: SearchResult, question: str) -> tuple[int, str]:
    text = f"{result.title} {result.snippet}".lower()
    terms = [term for term in _TERM_SPLIT.split(question.lower()) if len(term) >= 3]
    hits = sum(1 for term in terms if term in text)
    score = hits * 8
    if result.url.startswith("https://"):
        score += 4
    if result.source_type == "news":
        score += 3
    if result.age:
        score += 2
    if _AUTHORITY.search(result.url):
        score += 5
    reason = f"{hits} query term hits" + (f", freshness: {result.age}" if result.age else "")
    return score, reason


def rank_sources(results: list[SearchResult], question: str, limit: int) -> list[RankedSource]:
    """Dedup by normalized URL (first occurrence wins), score, stable-sort descending, keep ``limit``."""
    deduped: dict[str, SearchResult] = {}
    for result in results:
        key = normalize_url_for_dedup(result.url)
        if key not in deduped:
            deduped[key] = SearchResult(
                title=result.title,
                url=key,
                snippet=result.snippet,
                query=result.query,
                source_type=result.source_type,
                age=result.age,
            )
    ranked = [RankedSource(result, *source_score(result, question)) for result in deduped.values()]
    ranked.sort(key=lambda source: source.score, reverse=True)
    return ranked[:limit]


def excerpt(text: str, max_chars: int) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    return compact if len(compact) <= max_chars else f"{compact[:max_chars]}..."


def extract_evidence(fetched: list[tuple[RankedSource, FetchResult]]) -> list[ExtractedSource]:
    """A source is usable when the fetch had no error and the text is at least 200 characters."""
    extracted: list[ExtractedSource] = []
    for position, (source, page) in enumerate(fetched, start=1):
        usable = not page.error and len(page.content.strip()) >= MIN_USABLE_CHARS
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", page.content) if len(part.strip()) > 40]
        claims = [excerpt(sentence, 320) for sentence in sentences[:5]] or [excerpt(source.result.snippet, 320)]
        summary = (
            excerpt(page.content, 900)
            if usable
            else page.error or "Fetched content was too short to use as evidence."
        )
        extracted.append(
            ExtractedSource(
                index=position,
                title=source.result.title,
                url=source.url,
                summary=summary,
                key_claims=claims,
                usable=usable,
                fetch_error=page.error,
            )
        )
    return extracted
