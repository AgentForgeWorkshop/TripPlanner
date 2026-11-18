from datetime import date, timedelta
from typing import Literal
from pydantic import BaseModel, Field, model_validator


class ConferenceSearchInput(BaseModel):
    field: str
    start_date: date
    end_date: date
    geographic_preferences: list[str] = Field(default_factory=list)
    max_results: int = Field(default=5, ge=1, le=10)

    @model_validator(mode="after")
    def check_dates(self):
        if self.end_date < self.start_date:
            raise ValueError("End date must follow start date.")
        return self


class Conference(BaseModel):
    name: str
    research_focus: str
    city: str
    country: str
    start_date: date
    end_date: date

    registration_cost: float | None = Field(default=None, ge=0)
    currency: str | None = None

    registration_status: Literal[
        "confirmed", "estimated", "unknown"
    ]

    official_website: str
    source_urls: list[str]
    notes: str = ""

    @model_validator(mode="after")
    def check_registration(self):
        if self.registration_status == "unknown":
            if self.registration_cost is not None:
                raise ValueError("Unknown registration must have no cost.")
        else:
            if self.registration_cost is None or not self.currency:
                raise ValueError("Known costs require an amount and currency.")
        return self


class ConferenceSearchOutput(BaseModel):
    conferences: list[Conference]
    search_summary: str


class AccommodationSearchInput(BaseModel):
    country: str
    city: str | None = None
    conference_name: str | None = None
    check_in: date
    check_out: date
    max_price_per_night: float | None = Field(default=None, ge=0)
    currency: str | None = None
    max_results: int = Field(default=5, ge=1, le=10)

    @model_validator(mode="after")
    def check_stay(self):
        if self.check_out <= self.check_in:
            raise ValueError("Check-out must be after check-in.")
        if self.max_price_per_night is not None and not self.currency:
            raise ValueError("A price limit requires a currency.")
        return self

    @classmethod
    def from_conference(
        cls, conference: Conference, **overrides
    ) -> "AccommodationSearchInput":
        """Build a search request from a conference agent result."""
        check_out = conference.end_date
        if check_out <= conference.start_date:
            # One-day conference: still need at least one night.
            check_out = conference.start_date + timedelta(days=1)

        values = {
            "country": conference.country,
            "city": conference.city,
            "conference_name": conference.name,
            "check_in": conference.start_date,
            "check_out": check_out,
        }
        values.update(overrides)
        return cls(**values)


DISTANCE_NOT_KNOWN = (
    "Not known: the distance to the venue could not be verified. "
    "Please check it yourself."
)
ADDRESS_NOT_KNOWN = (
    "Not known: the address could not be verified. "
    "Please check it yourself."
)
DISTRICT_NOT_KNOWN = "Not known"

# Facts about an accommodation that must be backed by a source.
# Each fact has <fact>_status, <fact>_source_url and <fact>_evidence
# fields, plus the value fields below with their "not known" text.
NOT_KNOWN_VALUES = {
    "distance": {"distance_to_venue": DISTANCE_NOT_KNOWN},
    "location": {
        "address": ADDRESS_NOT_KNOWN,
        "district": DISTRICT_NOT_KNOWN,
    },
}


class Accommodation(BaseModel):
    name: str
    accommodation_type: Literal[
        "hotel", "hostel", "apartment", "guesthouse",
        "university_housing", "other"
    ]
    city: str
    country: str

    # Address and district. "sourced" only when a retrieved page
    # states where this property is (with evidence); otherwise
    # "not_known", and both are set to a standard warning.
    location_status: Literal["sourced", "not_known"] = Field(
        default="not_known",
        description=(
            "'sourced' only if a retrieved page states this property's "
            "address or district; else 'not_known'."
        )
    )
    address: str = Field(
        default=ADDRESS_NOT_KNOWN,
        description="Street address as stated by the source."
    )
    district: str = Field(
        default=DISTRICT_NOT_KNOWN,
        description="District or neighbourhood as stated by the source."
    )
    location_source_url: str | None = Field(
        default=None,
        description="URL of the retrieved page that states the location."
    )
    location_evidence: str | None = Field(
        default=None,
        description="Short quote or paraphrase of what the source says."
    )

    # Distance to the conference venue. "sourced" only when a retrieved
    # page states it (with evidence); otherwise "not_known", and
    # distance_to_venue is set to a standard warning.
    distance_status: Literal["sourced", "not_known"] = Field(
        default="not_known",
        description=(
            "'sourced' only if a retrieved page states the distance "
            "from this property to the venue itself; else 'not_known'."
        )
    )
    distance_to_venue: str = Field(
        default=DISTANCE_NOT_KNOWN,
        description=(
            "Distance to the conference venue as stated by the source. "
            "Leave unset when distance_status is 'not_known'."
        )
    )
    distance_source_url: str | None = Field(
        default=None,
        description="URL of the retrieved page that states the distance."
    )
    distance_evidence: str | None = Field(
        default=None,
        description="Short quote or paraphrase of what the source says."
    )

    price_per_night: float | None = Field(default=None, ge=0)
    currency: str | None = None

    price_status: Literal[
        "confirmed", "estimated", "unknown"
    ]

    # Evidence for the price: which page shows it and what it says.
    # Required for confirmed and estimated prices (see check_price).
    price_source_url: str | None = Field(
        default=None,
        description="URL of the retrieved page that shows this price."
    )
    price_evidence: str | None = Field(
        default=None,
        description=(
            "Short quote or paraphrase of what the source says: "
            "amount, currency, and the dates/conditions it applies to."
        )
    )

    booking_url: str
    source_urls: list[str]
    notes: str = ""

    @model_validator(mode="after")
    def check_price(self):
        if self.price_status == "unknown":
            if self.price_per_night is not None:
                raise ValueError("Unknown price must have no amount.")
        else:
            if self.price_per_night is None or not self.currency:
                raise ValueError("Known prices require an amount and currency.")
            if not self.price_source_url or not self.price_evidence:
                raise ValueError(
                    "Confirmed and estimated prices require price_source_url "
                    "and price_evidence. If no retrieved page shows a price, "
                    "use price_status 'unknown'."
                )
        return self

    def mark_not_known(self, fact: str) -> None:
        """Replace a fact ("distance" or "location") with its warning text."""
        setattr(self, f"{fact}_status", "not_known")
        for field, not_known in NOT_KNOWN_VALUES[fact].items():
            setattr(self, field, not_known)
        setattr(self, f"{fact}_source_url", None)
        setattr(self, f"{fact}_evidence", None)

    @model_validator(mode="after")
    def check_sourced_facts(self):
        for fact, not_known_values in NOT_KNOWN_VALUES.items():
            if getattr(self, f"{fact}_status") == "not_known":
                # Overwrite any guessed value with the standard warning.
                self.mark_not_known(fact)
                continue

            has_value = any(
                getattr(self, field) not in (None, "", not_known)
                for field, not_known in not_known_values.items()
            )
            if (
                not has_value
                or not getattr(self, f"{fact}_source_url")
                or not getattr(self, f"{fact}_evidence")
            ):
                raise ValueError(
                    f"A sourced {fact} requires a value, "
                    f"{fact}_source_url and {fact}_evidence. If no "
                    f"retrieved page states it, use {fact}_status "
                    "'not_known'."
                )
        return self


class AccommodationSearchOutput(BaseModel):
    accommodations: list[Accommodation]
    search_summary: str