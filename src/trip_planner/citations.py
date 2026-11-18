from collections.abc import Iterator
from threading import Lock
from typing import TypeVar
from urllib.parse import parse_qsl, urlsplit

from pydantic import BaseModel

from .models import Conference, Cost, Discovery, Entry, Lodging, Source, Transport, Travel
from .search import SearchHit, source_url

Sourced = TypeVar("Sourced", Conference, Cost, Transport, Lodging)
TRACKING_KEYS = {"fbclid", "gclid", "msclkid"}


def url_key(value: str) -> tuple | None:
    """Compare common display variants while retaining the page and its meaningful parameters."""
    try:
        parts = urlsplit(source_url(value))
        if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username is not None:
            return None
        query = tuple(
            (key, val) for key, val in parse_qsl(parts.query, keep_blank_values=True)
            if not (key.lower().startswith("utm_") or key.lower() in TRACKING_KEYS)
        )
        port = parts.port or (443 if parts.scheme == "https" else 80)
        fragment = parts.fragment if parts.fragment.startswith(("/", "!")) else ""
        return parts.scheme, parts.hostname, port, parts.path.removesuffix("/"), query, fragment
    except ValueError:
        return None


def cited_urls(data: object) -> Iterator[str]:
    if isinstance(data, dict):
        for key, value in data.items():
            if key in {"source_urls", "processing_source_urls"}:
                yield from value
            else:
                yield from cited_urls(value)
    elif isinstance(data, list):
        for item in data:
            yield from cited_urls(item)


class Evidence:
    def __init__(self) -> None:
        self.sources: dict[str, Source] = {}
        self._ids: dict[str, str] = {}
        self._urls_by_id: dict[str, str] = {}
        self._variants: dict[tuple, set[str]] = {}
        self._lock = Lock()

    def record(self, hits: list[SearchHit]) -> list[dict[str, str]]:
        results = []
        with self._lock:
            for hit in hits:
                url = hit["url"]
                if url not in self.sources:
                    self.sources[url] = Source(url=url, title=hit["title"])
                    source_id = f"S{len(self.sources)}"
                    self._ids[url] = source_id
                    self._urls_by_id[source_id] = url
                    key = url_key(url)
                    if key is not None:
                        self._variants.setdefault(key, set()).add(url)
                results.append({**hit, "source_id": self._ids[url]})
        return results

    def resolve(self, value: str) -> str | None:
        if value in self.sources:
            return value
        url = self._urls_by_id.get(value.upper())
        if url is not None:
            return url
        candidates = self._variants.get(url_key(value), set())
        if len(candidates) == 1:
            return next(iter(candidates))
        return None

    def references(self, citations: list[str], *, require_all: bool = True) -> list[str]:
        matches = [self.resolve(citation) for citation in citations]
        if require_all and None in matches:
            return []
        return list(dict.fromkeys(match for match in matches if match is not None))

    def ground(self, record: Sourced | None, *, require_all: bool = True) -> Sourced | None:
        if record is None:
            return None
        links = self.references(record.source_urls, require_all=require_all)
        return record.model_copy(update={"source_urls": links}) if links else None

    def cited_sources(self, result: BaseModel) -> dict[str, Source]:
        return {url: self.sources[url] for url in sorted(set(cited_urls(result.model_dump())))}


def ground_discovery(result: Discovery, evidence: Evidence) -> Discovery:
    conferences = []
    notes = list(result.notes)
    for event in result.conferences:
        grounded = evidence.ground(event, require_all=False)
        if grounded is None:
            notes.append(f"Could not match the citations for {event.name} to a search result.")
            continue
        if not evidence.references(event.source_urls):
            notes.append(f"Some citations for {event.name} could not be matched to search results.")
        registration = evidence.ground(event.registration)
        if event.registration is not None and registration is None:
            notes.append(f"Could not verify the registration price source for {event.name}.")
        conferences.append(grounded.model_copy(update={"registration": registration}))
    return Discovery(conferences=conferences, notes=notes)


def ground_travel(result: Travel, evidence: Evidence) -> Travel:
    transport = evidence.ground(result.transport)
    lodging = evidence.ground(result.lodging)
    if transport is not None:
        transport = transport.model_copy(update={"cost": evidence.ground(transport.cost)})
    if lodging is not None:
        lodging = lodging.model_copy(update={"cost": evidence.ground(lodging.cost)})
    return result.model_copy(update={
        "transport": transport,
        "lodging": lodging,
        "local_transport": evidence.ground(result.local_transport),
        "meals": evidence.ground(result.meals),
    })


def ground_entry(result: Entry, evidence: Evidence) -> Entry:
    sources = evidence.references(result.source_urls)
    timing_sources = evidence.references(result.processing_source_urls) if sources else []
    return result.model_copy(update={
        "status": result.status if sources or result.status == "domestic" else "unknown",
        "requirements": result.requirements if sources else [],
        "processing_time": result.processing_time if timing_sources else None,
        "processing_days_upper": result.processing_days_upper if timing_sources else None,
        "processing_source_urls": timing_sources,
        "cost": evidence.ground(result.cost) if sources else None,
        "source_urls": sources,
    })
