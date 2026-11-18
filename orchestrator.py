import os
from datetime import date
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool

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


orchestrator = create_agent(
    model=os.environ["MODEL"],
    tools=[find_conferences, find_accommodation, find_itinerary],
    system_prompt="""
    You coordinate specialist agents to help users plan
    conference-related travel.

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
    Country-information research is unavailable.
    When reporting accommodation options:
    - State which conference and dates they are for.
    - Include booking or source URLs.
    - Distinguish confirmed, estimated, and unknown prices.
    - Never interpret unknown prices as zero.

    Travel and country-information agents are not yet
    available. Do not invent their findings.
    """
)


if __name__ == "__main__":
    query = (
        "Find an academic ecology conference in Europe and USA between "
        "January 1 and June 30, 2027, then find up to three places "
        "to stay for the conference dates and best itinerary with flight details in terms of duration and price."
        "My total budget for the trip is 3000 USD. I prefer to travel by train if possible, but I am open to flights if they are significantly faster or cheaper."
        "My departure city is New York, and I would like to attend the conference with the lowest registration fee."
    )


    result = orchestrator.invoke(
        {
            "messages": [
                {"role": "user", "content": query}
            ]
        },
        config={"recursion_limit": 20}
    )

    for message in result["messages"]:
        message.pretty_print()