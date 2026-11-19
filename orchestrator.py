import argparse
from contextlib import nullcontext, redirect_stdout
import os
from pathlib import Path
from pydantic import ValidationError
from langchain.agents.structured_output import ToolStrategy
from trip_report import TripConfig, TripReport, render_report
from datetime import date
from dotenv import load_dotenv
from model_quota import pace_model_request
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage
from langchain.tools import tool

from VisaAgent import generate_visa_white_paper
from conference_agent import find_conferences as search
from travel_agent import TravelRequest, plan_travel
from accommodation_agent import find_accommodation as search_accommodation
from schemas import ConferenceSearchInput, AccommodationSearchInput

load_dotenv()


@tool(args_schema=ConferenceSearchInput)
def find_conferences(
    field: str,
    start_date: date,
    end_date: date,
    geographic_preferences: list[str] | None = None,
    max_results: int = 5
) -> str:
    """Delegate academic conference discovery to the conference expert."""

    request = ConferenceSearchInput(
        field=field,
        start_date=start_date,
        end_date=end_date,
        geographic_preferences=geographic_preferences or [],
        max_results=max_results
    )

    return search(request).model_dump_json()


@tool
def find_itinerary(request: TravelRequest) -> str:
    """Compare transport and lodging for one conference and return the best eligible itinerary."""
    return plan_travel(request).model_dump_json()


@tool
def find_visa_requirements(nationality: str, destination_country: str) -> str:
    """Look up conference entry requirements for a given passport nationality and destination country."""
    return generate_visa_white_paper(nationality, destination_country).model_dump_json()


@tool(args_schema=AccommodationSearchInput)
def find_accommodation(
    country: str,
    check_in: date,
    check_out: date,
    city: str | None = None,
    conference_name: str | None = None,
    max_price_per_night: float | None = None,
    currency: str | None = None,
    max_results: int = 5
) -> str:
    """Delegate accommodation search for a conference to the
    accommodation expert. Use the country, city, name and exact
    dates returned by the conference expert."""

    request = AccommodationSearchInput(
        country=country,
        city=city,
        conference_name=conference_name,
        check_in=check_in,
        check_out=check_out,
        max_price_per_night=max_price_per_night,
        currency=currency,
        max_results=max_results
    )

    return search_accommodation(request).model_dump_json()


@wrap_tool_call
def report_specialist_rate_limit(request, handler):
    try:
        return handler(request)
    except Exception as error:
        if getattr(error, "status_code", None) != 429:
            raise
        return ToolMessage(
            content=f"{request.tool_call['name']} unavailable: provider rate limit persisted after retries.",
            tool_call_id=request.tool_call["id"], status="error",
        )


orchestrator = create_agent(
    model=os.environ["MODEL"],
    middleware=[pace_model_request, report_specialist_rate_limit],
    tools=[find_conferences, find_accommodation, find_itinerary, find_visa_requirements],
    response_format=ToolStrategy(TripReport),
    system_prompt="""
    You coordinate specialist agents to help users plan
    conference-related travel.

    If a specialist reports a persistent rate limit, mark its findings unavailable
    and report the available results without calling it again for that request.

    Delegate conference discovery to the conference expert.

    Delegate accommodation search to the accommodation expert
    once a conference has been found or chosen. Pass the
    conference's country, city, name, and exact start and end
    dates from the conference expert's output as the country,
    city, conference_name, check_in and check_out. Use other
    dates only if the user explicitly asks. If the conference
    lasts a single day, set check_out to the following day.

    Ask for clarification only when required information
    is missing. Geographic preferences and accommodation
    budget are optional.

    When reporting conference options:
    - Preserve conference dates and locations.
    - Include official website URLs.
    - Distinguish confirmed, estimated, and unknown costs.
    - Never interpret unknown costs as zero.
    - Report missing information rather than inventing it.

    For itinerary requests delegate to find_itinerary. Use a supplied event or
    discover conferences first and pass their actual locations and dates.
    Pass the origin, destination, event dates, currency, constraints and preferences.
    Default to lowest cost unless the traveler prioritizes travel time.
    The travel budget covers round-trip transport and lodging for one attendee.
    For whole-trip budgets deduct known registration in the same currency first.
    If registration is unknown, flag whole-trip affordability as unverified.
    Preserve the specialist's status, estimates, limitations and source URLs.
    For visa questions and conference trip plans, use find_visa_requirements
    for each distinct destination being recommended. Use the traveler's stated
    passport nationality. If it is missing, ask for it and report any trip
    findings already available. Never infer citizenship from the departure city.
    Report visa or entry authorization, permitted stay, documents, application
    steps, processing time and estimated visa fees in relevant currency. Preserve the
    specialist's caveats and identify details that need official confirmation.
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


DEFAULT_QUERY = (
    "Find an academic ecology conference in Europe and USA between "
    "January 1 and June 30, 2027, then find up to three places "
    "to stay for the conference dates and best itinerary with flight details in terms of duration and price. "
    "My total budget for the trip is 3000 USD. I prefer to travel by train if possible, but I am open to flights if they are significantly faster or cheaper. "
    "My departure city is New York, and I would like to attend the conference with the lowest registration fee. "
    "I currently hold an Indian passport. Include the visa requirements."
)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Research conference trips and write an HTML comparison.")
    parser.add_argument("--config", type=Path, help="Trip criteria in JSON format")
    parser.add_argument("--output", type=Path, default=Path("trip_report.html"))
    parser.add_argument("--verbose", action="store_true", help="Show specialist diagnostics")
    args = parser.parse_args(argv)
    config = None
    query = DEFAULT_QUERY
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
