from datetime import date
from decimal import Decimal
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
CountryCode = Annotated[str, StringConstraints(to_upper=True, pattern=r"^[A-Za-z]{2}$")]
Currency = Annotated[str, StringConstraints(to_upper=True, pattern=r"^[A-Za-z]{3}$")]
Amount = Annotated[Decimal, Field(ge=0, max_digits=16, decimal_places=4)]
Links = Annotated[list[Text], Field(min_length=1)]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TripRequest(Record):
    field: Text
    origin_city: Text
    origin_country: CountryCode
    passport_country: CountryCode
    residence_country: CountryCode | None = None
    existing_entry_permission: Text | None = None
    window_start: date
    window_end: date
    unavailable_dates: list[date] = Field(default_factory=list)
    max_trip_days: int | None = Field(default=None, ge=1)
    destination_countries: list[CountryCode] = Field(default_factory=list)
    preferences: list[Text] = Field(default_factory=list)
    budget: Amount = Field(gt=0)
    currency: Currency

    @model_validator(mode="after")
    def check_window(self) -> Self:
        if self.window_end < self.window_start:
            raise ValueError("window_end must be on or after window_start")
        return self


class Cost(Record):
    amount: Amount
    currency: Currency
    source_urls: Links
    basis: Text


class Conference(Record):
    name: Text
    city: Text
    country: CountryCode
    venue: Text | None
    start: date
    end: date
    registration_status: Literal["open", "not_open_yet", "closed", "unknown"]
    registration_deadline: date | None
    relevance: Text
    registration: Cost | None
    source_urls: Links


class Discovery(Record):
    conferences: list[Conference] = Field(max_length=3)
    notes: list[Text]


class Transport(Record):
    mode: Literal["flight", "rail", "bus", "ferry", "car", "local"]
    outbound: Text
    inbound: Text
    transit_countries: list[CountryCode]
    depart_origin: date
    arrive_destination: date
    depart_destination: date
    arrive_origin: date
    cost: Cost | None
    source_urls: Links


class Lodging(Record):
    name: Text
    check_in: date
    check_out: date
    cost: Cost | None
    source_urls: Links


class Travel(Record):
    transport: Transport | None
    lodging: Lodging | None
    local_transport: Cost | None
    meals: Cost | None
    notes: list[Text]


class Entry(Record):
    status: Literal[
        "domestic", "visa_exempt", "authorization_required", "visa_required",
        "existing_permission", "ineligible", "unknown",
    ]
    requirements: list[Text]
    processing_time: Text | None
    processing_days_upper: int | None = Field(ge=0)
    processing_source_urls: list[Text]
    cost: Cost | None
    source_urls: list[Text]
    notes: list[Text]


class Source(Record):
    url: Text
    title: Text


class CostLine(Record):
    category: Text
    amount: Amount | None
    basis: Text
    source_urls: list[Text] = Field(default_factory=list)


class Budget(Record):
    currency: Currency
    limit: Amount
    items: list[CostLine]
    known_subtotal: Decimal
    estimated_total: Decimal | None
    remaining: Decimal | None


class ItineraryDay(Record):
    date: date
    activities: list[Text]


class TripOption(Record):
    status: Literal["within_estimated_budget", "needs_verification"]
    conference: Conference
    travel: Travel
    entry: Entry | None
    budget: Budget
    itinerary: list[ItineraryDay]
    issues: list[Text]


class Finding(Record):
    conference: Text | None = None
    detail: Text


class TripPlan(Record):
    status: Literal["within_estimated_budget", "needs_verification", "no_match"]
    researched_on: date
    options: list[TripOption]
    findings: list[Finding]
    sources: list[Source]
