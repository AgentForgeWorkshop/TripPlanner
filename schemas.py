from datetime import date
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