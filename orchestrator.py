import argparse
from contextlib import nullcontext, redirect_stdout
from hashlib import sha256
import json
import os
from pathlib import Path
from decimal import Decimal
from typing import Literal
from pydantic import ValidationError
from langchain.agents.structured_output import ToolStrategy
from trip_report import ConferenceRow, Note, TripConfig, TripReport, render_report
from datetime import date
from dotenv import load_dotenv
from model_quota import pace_model_request
from langchain.agents import create_agent
from langchain.agents.middleware import before_model, wrap_tool_call
from langchain_core.messages import HumanMessage, ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.errors import GraphRecursionError

from VisaAgent import generate_visa_white_paper
from conference_agent import ExtendedConferenceOutput, find_conferences as search
from travel_agent import TravelRequest, plan_travel
from accommodation_agent import find_accommodation as search_accommodation
from schemas import ConferenceSearchInput, AccommodationSearchInput, VisaSearchInput

load_dotenv()


@tool(args_schema=ConferenceSearchInput)
def find_conferences(
    field: str,
    start_date: date,
    end_date: date,
    geographic_preferences: list[str] | None = None,
    max_results: int = 5,
    excluded_dates: list[date] | None = None,
    in_person_only: bool = True,
) -> str:
    """Delegate academic conference discovery to the conference expert.
    Pass the traveler's unavailable dates as excluded_dates."""

    request = ConferenceSearchInput(
        field=field,
        start_date=start_date,
        end_date=end_date,
        geographic_preferences=geographic_preferences or [],
        max_results=max_results,
        excluded_dates=excluded_dates or [],
        # This orchestrator plans travel, so online-only events never apply.
        in_person_only=True,
    )

    result = search(request)
    # The specialist remains unchanged. The orchestrator adds stable IDs only
    # to conference records that are eligible for downstream planning.
    payload = result.model_dump(mode="json")
    for record, data in zip(result.conferences, payload["conferences"]):
        data["conference_id"] = conference_id(record)
    for candidate, data in zip(result.provisional_candidates, payload["provisional_candidates"]):
        if candidate.status == "provisional":
            data["conference"]["conference_id"] = conference_id(candidate.conference)
    for record in eligible_conferences([result]):
        print(f"[ORCHESTRATOR] Eligible conference: {conference_id(record)} — {record.name}")
    return json.dumps(payload, ensure_ascii=False)


@tool
def find_itinerary(
    conference_id: str,
    origin: str,
    currency: str,
    runtime: ToolRuntime,
    transport_lodging_budget: Decimal | None = None,
    earliest_departure: date | None = None,
    latest_return: date | None = None,
    excluded_dates: list[date] | None = None,
    max_trip_days: int | None = None,
    priority: Literal["lowest_cost", "shortest_travel"] = "lowest_cost",
    preferences: list[str] | None = None,
) -> str:
    """Compare transport and lodging for an eligible conference. Pass its conference_id plus traveler constraints, not event dates or destination."""
    conference = _resolve_conference(conference_id, runtime.state)
    try:
        request = TravelRequest(
            origin=origin, destination=f"{conference.city}, {conference.country}",
            event_start=conference.start_date, event_end=conference.end_date,
            currency=currency, transport_lodging_budget=transport_lodging_budget,
            earliest_departure=earliest_departure, latest_return=latest_return,
            excluded_dates=excluded_dates or [], max_trip_days=max_trip_days,
            priority=priority, preferences=preferences or [],
        )
    except ValidationError as error:
        raise PlanningInputError(str(error)) from error
    return plan_travel(request).model_dump_json()


@tool
def find_visa_requirements(conference_id: str, nationality: str, runtime: ToolRuntime,
                           residence_country: str | None = None,
                           origin_country: str | None = None) -> str:
    """Look up entry requirements using an eligible conference_id and passport nationality. Pass residence and origin countries when known."""
    conference = _resolve_conference(conference_id, runtime.state)
    try:
        request = VisaSearchInput(nationality=nationality, destination_country=conference.country,
                                  residence_country=residence_country, origin_country=origin_country)
    except ValidationError as error:
        raise PlanningInputError(str(error)) from error
    return generate_visa_white_paper(request.nationality, request.destination_country,
                                     request.residence_country, request.origin_country).model_dump_json()


@tool
def find_accommodation(
    conference_id: str,
    runtime: ToolRuntime,
    max_price_per_night: float | None = None,
    currency: str | None = None,
    max_results: int = 5,
    check_in: date | None = None,
    check_out: date | None = None,
) -> str:
    """Find accommodation for an eligible conference_id. Python supplies the event location, name and dates. Set check-in or check-out only when the traveler explicitly requests different dates."""
    conference = _resolve_conference(conference_id, runtime.state)
    overrides = dict(max_price_per_night=max_price_per_night, currency=currency, max_results=max_results)
    if check_in is not None:
        overrides["check_in"] = check_in
    if check_out is not None:
        overrides["check_out"] = check_out
    try:
        request = AccommodationSearchInput.from_conference(conference, **overrides)
    except ValidationError as error:
        raise PlanningInputError(str(error)) from error
    if not request.check_in <= conference.start_date <= conference.end_date <= request.check_out:
        raise PlanningInputError("Accommodation dates must cover the eligible conference.")

    return search_accommodation(request).model_dump_json()


DOWNSTREAM_TOOLS = {"find_itinerary", "find_accommodation", "find_visa_requirements"}
SPECIALIST_TOOLS = DOWNSTREAM_TOOLS | {"find_conferences"}


def current_tool_messages(messages):
    """Only use actual tool results from the current user request, not AI prose."""
    start = 0
    for index, message in enumerate(messages):
        if isinstance(message, HumanMessage):
            start = index + 1
    return [message for message in messages[start:] if isinstance(message, ToolMessage)]


def conference_results(messages):
    """Read validated discovery results directly from the graph's message state."""
    results = []
    seen = False
    for message in current_tool_messages(messages):
        if message.name != "find_conferences":
            continue
        if message.status != "success":
            # Input validation errors may be corrected by the model. A specialist
            # execution failure, in contrast, is terminal for this trip request.
            seen = seen or bool(_previous_failure([message], "find_conferences"))
            continue
        seen = True
        if not isinstance(message.content, str):
            continue
        try:
            results.append(ExtendedConferenceOutput.model_validate_json(message.content))
        except ValidationError:
            # Malformed results must never authorize downstream planning.
            continue
    return results, seen


def eligible_conferences(results):
    return [conference for result in results for conference in result.conferences] + [
        candidate.conference for result in results for candidate in result.provisional_candidates
        if candidate.status == "provisional"
    ]


class PlanningInputError(ValueError):
    """Recoverable input error in an orchestrator-to-specialist request."""


def conference_id(conference):
    """Deterministic identifier for an exact conference record (not an LLM guess)."""
    key = json.dumps([conference.name, conference.city, conference.country,
                      conference.start_date.isoformat(), conference.end_date.isoformat()],
                     ensure_ascii=False, separators=(",", ":"))
    return "conf_" + sha256(key.encode("utf-8")).hexdigest()[:16]


def eligible_registry(messages):
    results, _ = conference_results(messages)
    # A verified record takes precedence over a provisional duplicate.
    registry = {conference_id(c.conference): c.conference for r in results
                for c in r.provisional_candidates if c.status == "provisional"}
    registry.update({conference_id(c): c for r in results for c in r.conferences})
    return registry


def _resolve_conference(identifier, state):
    conference = eligible_registry(state.get("messages", [])).get(identifier)
    if conference is None:
        raise PlanningInputError("Unknown or ineligible conference_id. Use an ID returned by find_conferences in this request.")
    return conference


def _error_message(request, code, detail):
    name = request.tool_call["name"]
    print(f"[ORCHESTRATOR] {name}: {detail}")
    return ToolMessage(
        content=json.dumps({"status": code, "specialist": name, "message": detail}),
        name=name, tool_call_id=request.tool_call["id"], status="error",
    )


def _previous_failure(messages, name):
    for message in current_tool_messages(messages):
        if message.name != name or message.status != "error" or not isinstance(message.content, str):
            continue
        try:
            payload = json.loads(message.content)
        except ValueError:
            continue
        if isinstance(payload, dict) and payload.get("status") == "specialist_unavailable":
            return payload.get("message", "Specialist unavailable.")
    return None


def guard_specialist_call(request, handler):
    """Validate before execution and contain expected specialist failures."""
    name = request.tool_call["name"]
    if name not in SPECIALIST_TOOLS:
        return handler(request)
    messages = request.state.get("messages", [])
    failure = _previous_failure(messages, name)
    if failure:
        return _error_message(request, "specialist_unavailable", failure)

    if name in DOWNSTREAM_TOOLS:
        results, seen = conference_results(messages)
        registry = eligible_registry(messages)
        if not registry:
            detail = ("No verified or provisional conference is eligible for planning."
                      if seen else "Conference discovery must finish before downstream planning. Call find_conferences first.")
            return _error_message(request, "blocked_by_eligibility", detail)
        identifier = request.tool_call["args"].get("conference_id")
        if not isinstance(identifier, str) or identifier not in registry:
            return _error_message(request, "blocked_by_eligibility",
                                  f"Unknown or ineligible conference_id: {identifier!r}. Available IDs: {', '.join(registry)}.")
        conference = registry[identifier]
        print(f"[ORCHESTRATOR] {name}: {identifier} — {conference.name} | "
              f"{conference.city}, {conference.country} | {conference.start_date} to {conference.end_date}")

    try:
        return handler(request)
    except PlanningInputError as error:
        return _error_message(request, "invalid_planning_input", str(error))
    except GraphRecursionError:
        return _error_message(request, "specialist_unavailable",
                              "Execution step limit reached. Findings unavailable; do not retry this specialist in this trip request.")
    except Exception as error:
        if getattr(error, "status_code", None) != 429:
            raise
        return _error_message(request, "specialist_unavailable",
                              "Provider rate limit persisted after retries. Findings unavailable; do not retry this specialist in this trip request.")


@wrap_tool_call
def report_specialist_rate_limit(request, handler):
    return guard_specialist_call(request, handler)


def no_eligible_report(messages):
    """Create an honest report without another LLM call if discovery found no usable event."""
    results, seen = conference_results(messages)
    if not seen or eligible_conferences(results):
        return None
    if not results:
        failure = _previous_failure(messages, "find_conferences")
        return TripReport(summary="Conference discovery unavailable. " + (failure or "No valid discovery results were returned."),
                          conferences=[], travel=[], entry=[])

    rows = []
    for result in results:
        for candidate in result.unverified_candidates:
            c = candidate.conference
            notes = [Note(text="Unverified — not eligible for trip planning. " + candidate.reason)]
            notes.extend(Note(text="Supporting source (not sufficient to verify event)", url=url)
                         for url in candidate.supporting_urls)
            notes.extend(Note(text="Conflicting source", url=url) for url in candidate.conflicting_urls)
            rows.append(ConferenceRow(
                conference=c.name, location=f"{c.city}, {c.country}",
                dates=f"{c.start_date} to {c.end_date} (unverified)", registration="Unknown",
                trip_total="Unknown", budget_status="Not eligible for planning", notes=notes,
            ))
    summary = ("No verified or provisional conferences were found. Travel, accommodation and visa research were skipped; "
               + ("unverified candidates are listed for review only." if rows else "no eligible events were available."))
    if not rows:
        summary += " " + " ".join(result.search_summary for result in results)
    return TripReport(summary=summary, conferences=rows, travel=[], entry=[])


@before_model(can_jump_to=["end"])
def stop_without_eligible_conferences(state, runtime):
    report = no_eligible_report(state.get("messages", []))
    if report is not None:
        print("[ORCHESTRATOR] No eligible conferences; ending trip planning before downstream research.")
        return {"structured_response": report, "jump_to": "end"}
    return None


orchestrator = create_agent(
    model=os.environ["MODEL"],
    middleware=[pace_model_request, report_specialist_rate_limit, stop_without_eligible_conferences],
    tools=[find_conferences, find_accommodation, find_itinerary, find_visa_requirements],
    response_format=ToolStrategy(TripReport),
    system_prompt="""
    You coordinate specialist agents to help users plan
    conference-related travel.

    If a specialist reports a persistent rate limit or execution step limit,
    mark its findings unavailable and report the available results without calling
    it again for that request. Do not substitute zero for missing costs. Record
    the failure explicitly in the report notes.

    Delegate conference discovery to the conference expert. Pass the
    traveler's unavailable dates as excluded_dates. Recommend only
    conferences from "conferences" (verified) or "provisional_candidates".
    Mention an "unverified_candidates" entry only with its reason and label
    it unverified. Never present "rejected_candidates". State each
    conference's verification status in its notes.
    Wait until conference discovery has returned before calling any downstream
    specialist. Each verified or provisional conference has a conference_id in
    its record. Use that exact ID to call all downstream tools. Python looks up
    the conference by ID and supplies the original event details. Do not invent
    IDs or attempt to plan travel for unverified or rejected events.
    If an ID is rejected, use one of the returned available IDs only if it
    corresponds to the intended event; otherwise report the limitation.

    Delegate accommodation search to the accommodation expert
    once a conference has been found or chosen. Pass conference_id, any nightly
    budget, currency and maximum results. Python automatically supplies the
    event name, city, country and check-in/check-out dates, including the
    single-day conference case. Pass optional check_in and check_out overrides
    only if the user explicitly requests different dates.

    Ask for clarification only when required information
    is missing. Geographic preferences and accommodation
    budget are optional.

    When reporting conference options:
    - Preserve conference dates and locations.
    - Include official website URLs.
    - Distinguish confirmed, estimated, and unknown costs.
    - Never interpret unknown costs as zero.
    - Report missing information rather than inventing it.

    For itinerary requests delegate to find_itinerary. Pass conference_id,
    origin, currency, constraints and preferences. Python constructs the actual
    destination and event dates from the eligible conference record.
    Default to lowest cost unless the traveler prioritizes travel time.
    The travel budget covers round-trip transport and lodging for one attendee.
    For whole-trip budgets deduct known registration in the same currency first.
    If registration is unknown, flag whole-trip affordability as unverified.
    Preserve the specialist's status, estimates, limitations and source URLs.
    If find_itinerary returns no_match, add a travel row saying no itinerary met
    the constraints and give the rejected_options reasons (for example "only
    option found lasts 8 days, over the 7-day maximum"), with its known cost.
    For conference trip plans, use find_visa_requirements
    for each distinct destination being recommended, passing the conference_id
    of one eligible event in that country. Use the traveler's stated
    passport nationality and pass their residence country as residence_country.
    Pass the trip's departure country as origin_country when known; never assume
    it from residence or citizenship. Distinguish rule shortcuts from model-reported sources.
    If nationality is missing, ask for it and report any trip
    findings already available. Never infer citizenship from the departure city.
    Report visa or entry authorization, permitted stay, documents, application
    steps, processing time and estimated visa fees in relevant currency. Preserve the
    specialist's caveats, source_urls and research_status, and identify details
    that need official confirmation. Never label model research independently verified.
    A null total visa fee is Unknown, not zero; report known components in notes.
    When reporting accommodation options:
    - State which conference and dates they are for.
    - Include booking or source URLs.
    - Label every price as confirmed, estimated, or unknown.
    - Never present an estimated price as confirmed or as the
      current rate; say what it is based on.
    - Never interpret unknown prices as zero.
    - Remind the user that prices must be checked on the
      booking site before relying on them.
    - Give a distance to the venue only when its
      distance_status is "sourced". Otherwise say the
      distance is not known and the user should check it.
    - Give an address or district only when its
      location_status is "sourced". Otherwise say the
      location is not known and the user should check it.

    Return the concise structured trip report. Use one comparison row per conference,
    rows for relevant routes and stays, and one entry row per destination. Keep cells
    short. Prefix estimated prices with ≈ and include currency and units. For unknown
    costs write Unknown. Put sources, estimate basis, exclusions and visa application
    details in row notes. Never count the same lodging twice or label a subtotal a
    whole-trip total. For no matches or missing input, explain in the summary and
    leave unsupported sections empty.
    """
)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Research conference trips and write an HTML comparison.")
    parser.add_argument("--config", type=Path, help="Trip criteria in JSON format")
    parser.add_argument("--output", type=Path, default=Path("trip_report.html"))
    parser.add_argument("--verbose", action="store_true", help="Show specialist diagnostics")
    args = parser.parse_args(argv)
    config = None
    if args.config:
        try:
            config = TripConfig.model_validate_json(args.config.read_bytes())
        except (OSError, ValidationError) as exc:
            parser.error(f"Unable to load {args.config}: {exc}")
        query = ("Find conference, travel, accommodation and visa options using these trip criteria. "
                 "Apply the travel window, unavailable dates, destination list, whole-trip budget "
                 "and maximum trip length to the proposed itineraries. "
                 "Country codes identify the origin, destinations, passport and residence.\n"
                 + config.model_dump_json())
    else:
        try:
            query = input("What trip would you like to plan? ").strip()
        except EOFError:
            parser.error("No trip request provided. Enter a prompt or use --config.")
        if not query:
            parser.error("Please enter a non-empty trip request.")

    print("Researching trip options…", flush=True)
    with open(os.devnull, "w") as quiet:
        with (nullcontext() if args.verbose else redirect_stdout(quiet)):
            result = orchestrator.invoke({"messages": [{"role": "user", "content": query}]},
                                         config={"recursion_limit": 30})
    report = TripReport.model_validate(result["structured_response"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_report(report, config), encoding="utf-8")
    print(f"Trip report saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
