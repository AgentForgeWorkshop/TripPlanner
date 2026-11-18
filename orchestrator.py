import os
from datetime import date
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool

from conference_agent import find_conferences as search
from schemas import ConferenceSearchInput

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


orchestrator = create_agent(
    model=os.environ["MODEL"],
    tools=[find_conferences],
    system_prompt="""
    You coordinate specialist agents to help users plan
    conference-related travel.

    Delegate conference discovery to the conference expert.

    Ask for clarification only when required information
    is missing. Geographic preferences are optional.

    When reporting conference options:
    - Preserve conference dates and locations.
    - Include official website URLs.
    - Distinguish confirmed, estimated, and unknown costs.
    - Never interpret unknown costs as zero.
    - Report missing information rather than inventing it.

    Travel and country-information agents are not yet
    available. Do not invent their findings.
    """
)


if __name__ == "__main__":
    query = input("What trip would you like to plan? ")

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