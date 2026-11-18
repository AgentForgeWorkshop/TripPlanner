from collections.abc import Callable
from datetime import date
from decimal import Decimal
from typing import Literal, TypedDict, TypeVar

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .agents import Research, ResearchError, Specialists
from .models import Conference, Entry, Finding, Source, Travel, TripOption, TripPlan, TripRequest
from .planning import calendar_conflict, entry_issues, make_budget, make_itinerary, trip_conflict


class WorkflowInput(TypedDict):
    request: TripRequest
    today: date


class WorkflowOutput(TypedDict):
    plan: TripPlan


class State(WorkflowInput, total=False):
    conferences: list[Conference]
    discovery_findings: list[Finding]
    discovery_incomplete: bool
    conference_sources: dict[str, Source]
    travel: dict[int, Research[Travel] | str]
    entry: dict[int, Research[Entry] | str]
    plan: TripPlan


def event_conflict(request: TripRequest, event: Conference, today: date) -> str | None:
    if request.destination_countries and event.country not in request.destination_countries:
        return "The destination is outside the requested countries."
    if event.registration_status == "closed" or (
        event.registration_deadline is not None and event.registration_deadline < today
    ):
        return "The registration deadline has passed or registration is closed."
    return calendar_conflict(request, event.start, event.end, today)


SpecialistResult = TypeVar("SpecialistResult", Travel, Entry)


def collect(
    attempts: dict[int, Research[SpecialistResult] | str], index: int, event: Conference,
    specialist: str, findings: list[Finding], sources: dict[str, Source],
) -> SpecialistResult | None:
    attempt = attempts[index]
    if isinstance(attempt, str):
        findings.append(Finding(conference=event.name, detail=f"{specialist} research failed. {attempt}"))
        return None
    sources.update(attempt.sources)
    return attempt.result


def assemble(state: State) -> WorkflowOutput:
    request, today = state["request"], state["today"]
    findings = list(state["discovery_findings"])
    sources = dict(state["conference_sources"])
    options = []
    incomplete = state["discovery_incomplete"]

    for index, event in enumerate(state["conferences"]):
        travel = collect(state["travel"], index, event, "Travel", findings, sources)
        entry = collect(state["entry"], index, event, "Country information", findings, sources)
        if (entry and entry.status == "ineligible" and entry.source_urls
                and event.country != request.origin_country):
            findings.append(Finding(
                conference=event.name, detail="The traveler is ineligible under the sourced entry rules."
            ))
            continue
        if travel is None:
            incomplete = True
            continue
        conflict = trip_conflict(request, event, travel, today)
        if conflict:
            findings.append(Finding(conference=event.name, detail=conflict))
            incomplete = True  # The specialist evaluated one route and stay for this event.
            continue

        budget = make_budget(request, event, travel, entry)
        if budget.known_subtotal > budget.limit:
            detail = f"Known costs of {budget.known_subtotal} {budget.currency} exceed the budget."
            findings.append(Finding(conference=event.name, detail=detail))
            continue

        assert travel.transport is not None
        issues = entry_issues(request, event, entry, travel.transport.depart_origin, today)
        if event.registration_status != "open":
            issues.append("Confirm when registration opens and that attendee places are available."
                          if event.registration_status == "not_open_yet"
                          else "Confirm that conference registration is open.")
        if budget.estimated_total is None:
            missing = ", ".join(item.category.lower() for item in budget.items if item.amount is None)
            issues.append(f"Verify missing costs for {missing}.")
        options.append(TripOption(
            status="needs_verification" if issues else "within_estimated_budget",
            conference=event,
            travel=travel,
            entry=entry,
            budget=budget,
            itinerary=make_itinerary(event, travel),
            issues=issues,
        ))

    options.sort(key=lambda option: (
        option.status != "within_estimated_budget",
        option.budget.estimated_total is None,
        option.budget.estimated_total if option.budget.estimated_total is not None else Decimal(0),
        option.budget.known_subtotal,
        option.conference.start,
    ))
    status: Literal["within_estimated_budget", "needs_verification", "no_match"]
    if options:
        status = options[0].status
    else:
        status = "needs_verification" if incomplete else "no_match"
    return {"plan": TripPlan(status=status, researched_on=today, options=options, findings=findings,
                             sources=[sources[url] for url in sorted(sources)])}


def build_workflow(specialists: Specialists) -> CompiledStateGraph:
    """Compile a coordinator that takes a validated request and a research date."""

    def discover(state: State) -> dict:
        request, today = state["request"], state["today"]
        initial = {"discovery_incomplete": False, "conferences": [], "conference_sources": {}}
        if request.window_end < today:
            return {**initial, "discovery_findings": [Finding(detail="The travel window has already ended.")]}
        try:
            research = specialists.discover(request, today)
        except ResearchError as exc:
            return {**initial, "discovery_incomplete": True,
                    "discovery_findings": [Finding(detail=f"Conference research failed. {exc}")]}

        findings = [Finding(detail=note) for note in research.result.notes]
        events = []
        seen = set()
        for event in research.result.conferences:
            key = (event.name.casefold(), event.city.casefold(), event.country, event.start)
            if key in seen:
                continue
            seen.add(key)
            conflict = event_conflict(request, event, today)
            if conflict:
                findings.append(Finding(conference=event.name, detail=conflict))
            else:
                events.append(event)
        if not events:
            findings.append(Finding(detail="No qualifying conference was found for this search."))
        return {**initial, "conferences": events, "conference_sources": research.sources,
                "discovery_findings": findings}

    def research_candidates(
        state: State,
        specialist: Callable[[TripRequest, Conference, date], Research[SpecialistResult]],
    ) -> dict[int, Research[SpecialistResult] | str]:
        attempts = {}
        for index, event in enumerate(state["conferences"]):
            try:
                attempts[index] = specialist(state["request"], event, state["today"])
            except ResearchError as exc:
                attempts[index] = str(exc)
        return attempts

    def research_travel(state: State) -> dict:
        return {"travel": research_candidates(state, specialists.travel)}

    def research_entry(state: State) -> dict:
        return {"entry": research_candidates(state, specialists.entry)}

    def route_after_discovery(state: State) -> list[str]:
        return ["research_travel", "research_entry"] if state["conferences"] else ["assemble"]

    return (
        StateGraph(State, input_schema=WorkflowInput, output_schema=WorkflowOutput)
        .add_node("discover_conferences", discover)
        .add_node("research_travel", research_travel)
        .add_node("research_entry", research_entry)
        .add_node("assemble", assemble)
        .add_edge(START, "discover_conferences")
        .add_conditional_edges(
            "discover_conferences", route_after_discovery,
            ["research_travel", "research_entry", "assemble"],
        )
        .add_edge(["research_travel", "research_entry"], "assemble")
        .add_edge("assemble", END)
        .compile()
    )
