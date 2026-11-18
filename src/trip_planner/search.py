from collections.abc import Callable
from typing import TypedDict
from urllib.parse import unquote

import httpx
from ddgs import DDGS
from ddgs.exceptions import DDGSException
from pydantic import AnyHttpUrl, ValidationError

MAX_RESULTS = 5
DUCKDUCKGO_REDIRECT_PREFIXES = ("//duckduckgo.com/l/?uddg=", "https://duckduckgo.com/l/?uddg=")


class SearchHit(TypedDict):
    url: str
    title: str
    excerpt: str


Search = Callable[[str], list[SearchHit]]


class SearchError(RuntimeError):
    """A search provider could not return usable search data."""


def source_url(url: str) -> str:
    url = url.strip()
    for prefix in DUCKDUCKGO_REDIRECT_PREFIXES:
        if url.startswith(prefix):
            destination = url.removeprefix(prefix).rsplit("&rut=", 1)[0]
            return destination if destination.startswith(("http://", "https://")) else unquote(destination)
    return url


def normalize_hits(results: object, url_field: str, excerpt_field: str) -> list[SearchHit]:
    if not isinstance(results, list):
        raise SearchError("The web search provider returned an invalid result list.")

    hits: list[SearchHit] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        url, title, excerpt = item.get(url_field), item.get("title"), item.get(excerpt_field)
        if not all(isinstance(value, str) and value.strip() for value in (url, title, excerpt)):
            continue
        url = source_url(url)
        try:
            AnyHttpUrl(url)
        except ValidationError:
            continue
        hits.append({"url": url, "title": title.strip(), "excerpt": excerpt[:3500]})
    return hits


def search_tavily(query: str, *, client: httpx.Client) -> list[SearchHit]:
    try:
        response = client.post("/search", json={
            "query": query,
            "search_depth": "advanced",
            "max_results": MAX_RESULTS,
            "include_answer": False,
        })
        response.raise_for_status()
        results = response.json()["results"]
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        raise SearchError("Tavily could not complete the search.") from exc
    return normalize_hits(results, "url", "content")


def search_duckduckgo(query: str) -> list[SearchHit]:
    try:
        results = DDGS(timeout=10).text(query, backend="duckduckgo", max_results=MAX_RESULTS)
    except DDGSException as exc:
        raise SearchError("DuckDuckGo could not return search results.") from exc
    hits = normalize_hits(results, "href", "body")
    if not hits:
        raise SearchError("DuckDuckGo returned no usable search results.")
    return hits
