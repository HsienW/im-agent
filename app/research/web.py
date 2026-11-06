"""Web tools for the research agent: Tavily search and an egress-checked page fetcher.

Egress policy for page fetches (basic SSRF protection):

* only ``http`` / ``https`` URLs on ports 80 / 443;
* every address the host resolves to must be a global unicast address (no private, loopback,
  link-local, CGNAT, benchmarking 198.18.0.0/15, multicast or reserved ranges);
* redirects are followed manually and every hop is checked again.

The check runs before the request; the connection itself is made by httpx, so it is not pinned
to the validated address (a DNS answer that changes between check and connect is not covered).

When a TUN / fake-ip proxy (Clash, sing-box) makes every host resolve to 198.18.0.0/15, pass
``doh_resolver(...)`` so the public-address check uses DNS over HTTPS answers instead.
"""

from __future__ import annotations

import asyncio
import ipaddress
import re
import socket
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

import httpx

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
USER_AGENT = "im-agent-research/0.1"
ALLOWED_SCHEMES = frozenset({"http", "https"})
ALLOWED_PORTS = frozenset({80, 443})
DEFAULT_PORTS = {"http": 80, "https": 443}
FETCH_TIMEOUT_S = 12.0
SEARCH_TIMEOUT_S = 15.0
MAX_RESPONSE_BYTES = 1_000_000
MAX_CONTENT_CHARS = 16_000
MAX_REDIRECTS = 5

Resolver = Callable[[str], Awaitable[list[str]]]


class EgressError(Exception):
    """The URL is not allowed by the egress policy."""


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    query: str
    source_type: str = "general"
    age: str | None = None


@dataclass(frozen=True)
class FetchResult:
    url: str
    content: str = ""
    content_type: str | None = None
    title: str | None = None
    error: str | None = None


# --------------------------------------------------------------------------- search


async def tavily_search(
    client: httpx.AsyncClient,
    api_key: str,
    query: str,
    *,
    max_results: int = 8,
    search_depth: str = "basic",
    time_range: str | None = None,
) -> list[SearchResult]:
    """Call the Tavily Search API. Raises on HTTP or transport errors; never fabricates results."""
    body: dict[str, object] = {
        "query": query,
        "max_results": max(1, min(max_results, 20)),
        "search_depth": search_depth,
        "topic": "general",
        "include_answer": False,
        "include_raw_content": False,
        "include_images": False,
    }
    if time_range:
        body["time_range"] = time_range
    response = await client.post(
        TAVILY_SEARCH_URL,
        json=body,
        headers={"Authorization": f"Bearer {api_key}", "User-Agent": USER_AGENT},
        timeout=SEARCH_TIMEOUT_S,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Tavily search failed: {response.status_code} {response.text[:300]}")
    data = response.json()
    results: list[SearchResult] = []
    for item in data.get("results") or []:
        url = item.get("url") or ""
        if not url:
            continue
        results.append(
            SearchResult(
                title=item.get("title") or "Untitled",
                url=url,
                snippet=item.get("content") or item.get("raw_content") or "",
                query=data.get("query") or query,
                age=item.get("published_date"),
            )
        )
    return results


# --------------------------------------------------------------------------- egress policy


async def system_resolver(host: str) -> list[str]:
    loop = asyncio.get_running_loop()
    infos = await loop.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    return sorted({info[4][0] for info in infos})


def doh_resolver(endpoint: str, client: httpx.AsyncClient) -> Resolver:
    """Resolve A / AAAA records with a JSON DNS-over-HTTPS endpoint (e.g. https://223.5.5.5/resolve)."""

    async def resolve(host: str) -> list[str]:
        addresses: list[str] = []
        for record_type, type_code in (("A", 1), ("AAAA", 28)):
            response = await client.get(
                endpoint,
                params={"name": host, "type": record_type},
                headers={"accept": "application/dns-json"},
                timeout=5.0,
            )
            if response.status_code >= 400:
                continue
            for answer in response.json().get("Answer") or []:
                if answer.get("type") == type_code:
                    addresses.append(str(answer.get("data")))
        if not addresses:
            raise EgressError(f"DoH resolution returned no address for {host}")
        return addresses

    return resolve


def is_public_address(raw: str) -> bool:
    try:
        address = ipaddress.ip_address(raw.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return address.is_global and not address.is_multicast


async def check_url(url: str, resolver: Resolver) -> None:
    """Raise EgressError unless the URL is http(s) on 80/443 and its host resolves only to public addresses."""
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError as error:
        raise EgressError(f"invalid URL: {error}") from error
    scheme = parts.scheme.lower()
    if scheme not in ALLOWED_SCHEMES:
        raise EgressError(f"scheme not allowed: {scheme or '(none)'}")
    host = parts.hostname
    if not host:
        raise EgressError("URL has no host")
    if parts.username or parts.password:
        raise EgressError("URL credentials are not allowed")
    effective_port = port if port is not None else DEFAULT_PORTS[scheme]
    if effective_port not in ALLOWED_PORTS:
        raise EgressError(f"port not allowed: {effective_port}")
    try:
        ipaddress.ip_address(host)
        addresses = [host]
    except ValueError:
        if host.lower() in {"localhost", "localhost.localdomain"} or host.lower().endswith(".localhost"):
            raise EgressError(f"host not allowed: {host}") from None
        try:
            addresses = await resolver(host)
        except EgressError:
            raise
        except Exception as error:  # DNS failures are reported as fetch errors
            raise EgressError(f"DNS resolution failed for {host}: {error}") from error
    if not addresses:
        raise EgressError(f"no address for {host}")
    blocked = [address for address in addresses if not is_public_address(address)]
    if blocked:
        raise EgressError(f"non-public address for {host}: {', '.join(blocked[:3])}")


# --------------------------------------------------------------------------- html to text


class _ReadableTextParser(HTMLParser):
    SKIP = frozenset({"script", "style", "noscript", "svg", "nav", "footer", "header", "template", "iframe"})
    VOID = frozenset({"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"})
    BLOCK = frozenset({"p", "div", "li", "br", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section", "article", "main", "pre", "td", "th"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._in_title = False
        self._regions = {"article": 0, "main": 0, "body": 0}
        self.text: dict[str, list[str]] = {"article": [], "main": [], "body": [], "all": []}
        self.title_parts: list[str] = []
        self.description: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "meta" and self.description is None:
            values = {key.lower(): (value or "") for key, value in attrs}
            if values.get("name", "").lower() == "description" or values.get("property", "").lower() == "og:description":
                self.description = values.get("content") or None
        if tag in self.VOID:
            if tag == "br":
                self._emit(" ")
            return
        if tag in self.SKIP:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        if tag in self._regions:
            self._regions[tag] += 1
        if tag in self.BLOCK:
            self._emit(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.VOID:
            return
        if tag in self.SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False
        if tag in self._regions and self._regions[tag]:
            self._regions[tag] -= 1
        if tag in self.BLOCK:
            self._emit(" ")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)
            return
        if self._skip_depth:
            return
        self._emit(data)

    def _emit(self, data: str) -> None:
        self.text["all"].append(data)
        for region, depth in self._regions.items():
            if depth:
                self.text[region].append(data)


def _compact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def html_to_text(html: str) -> tuple[str, str | None, str | None]:
    """Return (readable text, title, description); prefers <article>, then <main>, then <body>."""
    parser = _ReadableTextParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # html.parser is lenient; keep whatever was collected
        pass
    text = ""
    for region in ("article", "main", "body", "all"):
        text = _compact("".join(parser.text[region]))
        if text:
            break
    title = _compact("".join(parser.title_parts)) or None
    description = _compact(parser.description) if parser.description else None
    return text, title, description


# --------------------------------------------------------------------------- fetch


_CHARSET = re.compile(rb"""<meta[^>]+charset=["']?([A-Za-z0-9_\-]+)""", re.I)


def _decode(body: bytes, content_type: str) -> str:
    charset = None
    match = re.search(r"charset=([A-Za-z0-9_\-]+)", content_type, re.I)
    if match:
        charset = match.group(1)
    elif "html" in content_type:
        meta = _CHARSET.search(body[:4096])
        if meta:
            charset = meta.group(1).decode("ascii", "ignore")
    for candidate in (charset, "utf-8"):
        if not candidate:
            continue
        try:
            return body.decode(candidate)
        except (LookupError, UnicodeDecodeError):
            continue
    return body.decode("utf-8", errors="replace")


def _limit(content: str, max_chars: int) -> str:
    if len(content) <= max_chars:
        return content
    return f"{content[:max_chars]}\n\n[Truncated to {max_chars} characters]"


async def _fetch_once(
    client: httpx.AsyncClient, url: str, resolver: Resolver, max_bytes: int, max_chars: int
) -> FetchResult:
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        await check_url(current, resolver)
        async with client.stream(
            "GET",
            current,
            headers={"Accept": "text/html,text/plain,application/json;q=0.9,*/*;q=0.8", "User-Agent": USER_AGENT},
            follow_redirects=False,
        ) as response:
            if response.status_code in (301, 302, 303, 307, 308) and response.headers.get("location"):
                current = urljoin(current, response.headers["location"])
                continue
            if response.status_code >= 400:
                return FetchResult(url, error=f"Error: web_fetch failed - {response.status_code} {response.reason_phrase}")
            content_type = response.headers.get("content-type", "")
            declared = response.headers.get("content-length")
            if declared and declared.isdigit() and int(declared) > max_bytes:
                return FetchResult(url, error=f"Error: web_fetch failed - response is larger than {max_bytes} bytes")
            lowered = content_type.lower()
            if lowered and not any(kind in lowered for kind in ("text/", "html", "json", "xml")):
                return FetchResult(url, content_type=content_type, error=f"Error: web_fetch failed - unsupported content type {content_type}")
            chunks: list[bytes] = []
            size = 0
            async for chunk in response.aiter_bytes():
                size += len(chunk)
                if size > max_bytes:
                    return FetchResult(url, error=f"Error: web_fetch failed - response is larger than {max_bytes} bytes")
                chunks.append(chunk)
        raw = _decode(b"".join(chunks), lowered)
        if "html" in lowered or (not lowered and raw.lstrip().startswith("<")):
            text, title, _description = html_to_text(raw)
        else:
            text, title = _compact(raw), None
        return FetchResult(url, content=_limit(text, max_chars), content_type=content_type or None, title=title)
    return FetchResult(url, error=f"Error: web_fetch failed - more than {MAX_REDIRECTS} redirects")


async def fetch_page(
    client: httpx.AsyncClient,
    url: str,
    *,
    resolver: Resolver = system_resolver,
    timeout_s: float = FETCH_TIMEOUT_S,
    max_bytes: int = MAX_RESPONSE_BYTES,
    max_chars: int = MAX_CONTENT_CHARS,
) -> FetchResult:
    """Fetch a public page as readable text. Never raises: failures come back as ``FetchResult.error``."""
    try:
        return await asyncio.wait_for(_fetch_once(client, url, resolver, max_bytes, max_chars), timeout_s)
    except EgressError as error:
        return FetchResult(url, error=f"Error: web_fetch blocked - {error}")
    except asyncio.TimeoutError:
        return FetchResult(url, error=f"Error: web_fetch failed - timed out after {timeout_s:g}s")
    except httpx.HTTPError as error:
        return FetchResult(url, error=f"Error: web_fetch failed - {type(error).__name__}: {error}"[:300])
