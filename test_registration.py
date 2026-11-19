from conference_agent import (
    ReviewedConference,
    estimate_registration_cost
)

conference = ReviewedConference(
    name="144th Annual Meeting of the American Ornithological Society",
    research_focus="Ornithology",
    city="Amherst",
    country="USA",
    start_date="2026-08-03",
    end_date="2026-08-07",
    registration_status="unknown",
    official_website=(
        "https://meeting.americanornithology.org/"
        "participate/registration/"
    ),
    official_website_verified=True,
    source_urls=[]
)

estimate_registration_cost(conference, "Ornithology")

print("\n--- REGISTRATION TEST RESULTS ---\n")
print(conference.model_dump_json(indent=2))