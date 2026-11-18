import os
from datetime import date
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool

from conference_agent import find_conferences as search
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
    tools=[find_conferences, find_accommodation],
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

    Travel and country-information agents are not yet
    available. Do not invent their findings.
    """
)


if __name__ == "__main__":
    query = (
        "Find an academic ecology conference in Europe between "
        "March 1 and June 30, 2027, then find up to three places "
        "to stay for the conference dates."
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