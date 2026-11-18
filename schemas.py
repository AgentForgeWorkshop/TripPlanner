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


class Accommodation(BaseModel):
    name: str
    accommodation_type: Literal[
        "hotel", "hostel", "apartment", "guesthouse",
        "university_housing", "other"
    ]
    city: str
    country: str
    address: str | None = None
    distance_to_venue: str | None = None

    price_per_night: float | None = Field(default=None, ge=0)
    currency: str | None = None

    price_status: Literal[
        "confirmed", "estimated", "unknown"
    ]

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
        return self


class AccommodationSearchOutput(BaseModel):
    accommodations: list[Accommodation]
    search_summary: str