"""Deterministic citation guard, ported from chat-gun ``deep-researcher-citation-guard.ts``.

After the model answers, every ``[n]`` marker that does not point at a usable source and every
URL that is not a usable source's URL is removed, and the removals are reported. The guard
validates citations against graph state; it never adds citations or rewrites claims.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import urlsplit, urlunsplit

CITATION_GUARD_POLICY = [
    "Cite only with the [n] numbers listed under 'Usable sources'. Never cite an unavailable source, never invent a "
    "number, and never write a URL that is not listed under 'Usable sources'.",
    "When there are no usable sources, say that no usable source was obtained and why, and stop there: do not state "
    "any specific date, number, version, schedule, decision or publication status from prior knowledge or inference, "
    "even with hedging such as 'would have', 'should be' or 'estimated'.",
]

_URL = re.compile(r"https?://[^\s<>\"'`）)\]（，。、；：！？「」【】《》]+")
_TRAILING = re.compile(r"[).,;:!?'\"]+$")
_MARKDOWN_LINK = re.compile(r"\[([^\]\n]+)\]\((https?://[^)\s]+)\)")
_MARKER_GROUP = re.compile(r"\s?\[([0-9]+(?:\s*(?:[,，、]|[-–])\s*[0-9]+)*)\]")
_RANGE = re.compile(r"^([0-9]+)\s*[-–]\s*([0-9]+)$")
_EMPTIED_LINE = re.compile(r"^[\s\-*•·👉➜→:：]*$")


class GuardSource(Protocol):
    index: int
    url: str
    usable: bool


@dataclass
class CitationGuardReport:
    modified: bool = False
    removed_markers: list[int] = field(default_factory=list)
    removed_urls: list[str] = field(default_factory=list)


def normalize_citation_url(raw: str) -> str:
    trimmed = _TRAILING.sub("", raw)
    try:
        parts = urlsplit(trimmed)
        if not parts.scheme or not parts.netloc:
            return trimmed
    except ValueError:
        return trimmed
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), parts.query, ""))


def _expand(group: str) -> list[int] | None:
    values: list[int] = []
    for part in re.split(r"[,，、]", group):
        part = part.strip()
        ranged = _RANGE.match(part)
        if ranged:
            start, end = int(ranged.group(1)), int(ranged.group(2))
            if end < start or end - start > 20:
                return None
            values.extend(range(start, end + 1))
        elif part.isascii() and part.isdigit():
            values.append(int(part))
        else:
            return None
    return values


def _drop_emptied_lines(original: str, guarded: str) -> str:
    before = original.split("\n")
    after = guarded.split("\n")
    if len(before) != len(after):
        return guarded
    return "\n".join(line for line, old in zip(after, before) if not (old.strip() and _EMPTIED_LINE.match(line)))


def guard_citations(answer: str, sources: list[GuardSource]) -> tuple[str, CitationGuardReport]:
    usable_indexes = {source.index for source in sources if source.usable}
    usable_urls = {normalize_citation_url(source.url) for source in sources if source.usable}
    removed_markers: set[int] = set()
    removed_urls: dict[str, None] = {}

    def link(match: re.Match[str]) -> str:
        normalized = normalize_citation_url(match.group(2))
        if normalized in usable_urls:
            return match.group(0)
        removed_urls.setdefault(normalized, None)
        return match.group(1)

    def bare_url(match: re.Match[str]) -> str:
        trailing = _TRAILING.search(match.group(0))
        normalized = normalize_citation_url(match.group(0))
        if normalized in usable_urls:
            return match.group(0)
        removed_urls.setdefault(normalized, None)
        return trailing.group(0) if trailing else ""

    def marker(match: re.Match[str]) -> str:
        values = _expand(match.group(1))
        if values is None or any(value > 99 for value in values):
            return match.group(0)
        kept = [value for value in values if value in usable_indexes]
        removed_markers.update(value for value in values if value not in usable_indexes)
        if len(kept) == len(values):
            return match.group(0)
        if not kept:
            return ""
        prefix = " " if match.group(0).startswith(" ") else ""
        return prefix + "".join(f"[{value}]" for value in kept)

    guarded = _MARKDOWN_LINK.sub(link, answer)
    guarded = _URL.sub(bare_url, guarded)
    guarded = _MARKER_GROUP.sub(marker, guarded)
    if removed_urls:
        guarded = re.sub(r"\s*(?:（\s*）|\(\s*\))", "", guarded)
        guarded = re.sub(r"[：:][ \t]*(?=[，,。；;])", "", guarded)
        guarded = re.sub(r"[ \t]+$", "", guarded, flags=re.M)
    guarded = _drop_emptied_lines(answer, guarded)
    modified = guarded != answer
    return (
        guarded if modified else answer,
        CitationGuardReport(
            modified=modified,
            removed_markers=sorted(removed_markers),
            removed_urls=list(removed_urls),
        ),
    )
