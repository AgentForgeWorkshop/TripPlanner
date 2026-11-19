from datetime import date
from decimal import Decimal
from html import escape
from typing import Annotated
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TripConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    field: Annotated[str, Field(min_length=1)]
    origin_city: Annotated[str, Field(min_length=1)]
    origin_country: str
    passport_country: str
    residence_country: str
    window_start: date
    window_end: date
    unavailable_dates: list[date] = Field(default_factory=list)
    max_trip_days: int = Field(gt=0)
    destination_countries: list[str] = Field(min_length=1)
    preferences: list[str] = Field(default_factory=list)
    budget: Decimal = Field(ge=0, allow_inf_nan=False)
    currency: str = Field(pattern=r"^[A-Z]{3}$")

    @model_validator(mode="after")
    def check_window(self):
        if self.window_end < self.window_start:
            raise ValueError("window_end must be on or after window_start")
        return self


class Note(BaseModel):
    text: str = Field(description="Short source label or pricing assumption.")
    url: str | None = Field(default=None, description="Supporting URL from specialist evidence, if available.")


class ConferenceRow(BaseModel):
    conference: str
    location: str
    dates: str
    registration: str
    trip_total: str = Field(description="Whole-trip amount and currency, or Unknown when incomplete.")
    budget_status: str
    notes: list[Note]


class TravelRow(BaseModel):
    conference: str
    dates: str
    route: str = Field(description="Short round-trip route and duration if known.")
    transport_cost: str
    accommodation: str
    lodging_cost: str = Field(description="State whether this is per night or for the stay.")
    notes: list[Note]


class EntryRow(BaseModel):
    destination: str
    requirement: str
    processing: str
    fee: str
    notes: list[Note]


class TripReport(BaseModel):
    summary: str = Field(description="One or two sentences giving the decision, blockers or clarification.")
    conferences: list[ConferenceRow]
    travel: list[TravelRow]
    entry: list[EntryRow]


def render_report(report: TripReport, config: TripConfig | None = None) -> str:
    notes = {}

    def table(title, rows, columns):
        if not rows:
            return ""
        headers = "".join(f'<th scope="col">{label}</th>' for _, label in columns)
        body = []
        for row in rows:
            refs = []
            for note in row.notes:
                key = (note.text, note.url)
                number = notes.setdefault(key, len(notes) + 1)
                refs.append(f'<a href="#note-{number}" aria-label="Note {number}">[{number}]</a>')
            cells = [escape(getattr(row, key)) for key, _ in columns]
            cells[0] += " <sup>" + " ".join(refs) + "</sup>" if refs else ""
            body.append("<tr>" + "".join(f"<td>{v}</td>" for v in cells) + "</tr>")
        return (f'<section><h2>{title}</h2><div class="scroll"><table><thead><tr>{headers}</tr>'
                f'</thead><tbody>{"".join(body)}</tbody></table></div></section>')

    sections = [
        table("Conference comparison", report.conferences, [
            ("conference", "Conference"), ("location", "Location"), ("dates", "Event dates"),
            ("registration", "Registration"), ("trip_total", "Whole trip"), ("budget_status", "Budget status")]),
        table("Transport and accommodation", report.travel, [
            ("conference", "Conference"), ("dates", "Travel dates"), ("route", "Route"),
            ("transport_cost", "Transport"), ("accommodation", "Stay"), ("lodging_cost", "Lodging")]),
        table("Visa and entry", report.entry, [
            ("destination", "Destination"), ("requirement", "Requirement"),
            ("processing", "Processing time"), ("fee", "Visa cost")]),
    ]
    references = []
    for (label, url), number in notes.items():
        item = escape(label)
        try:
            target = urlsplit(url or "")
            if target.scheme in ("http", "https") and target.hostname:
                item = f'<a href="{escape(url, quote=True)}">{item}</a>'
        except ValueError:
            pass
        references.append(f'<li id="note-{number}">{item}</li>')
    if references:
        sections.append('<section><h2>Notes and sources</h2><ol>' + "".join(references) + "</ol></section>")
    subtitle = (f"{config.field} · From {config.origin_city} · {config.window_start} – "
                f"{config.window_end} · Budget {config.currency} {config.budget:,.2f}") if config else ""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Conference trip options</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f6f5ef;color:#20332d;font:16px/1.55 system-ui,sans-serif}}
main{{max-width:1160px;margin:auto;padding:clamp(16px,4vw,56px)}}header{{border-top:5px solid #277059;padding:24px 0}}
h1,h2{{font-family:Georgia,serif}}h1{{font-size:clamp(32px,5vw,56px);margin:0}}h2{{margin:32px 0 10px}}
.summary{{border-left:4px solid #29745e;padding:12px 18px;background:#e5eee6;max-width:85ch}}
.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;background:white;font-size:.94rem}}
th,td{{text-align:left;padding:12px;vertical-align:top;border-bottom:1px solid #d5ddd6}}th{{background:#204c40;color:white}}
td:first-child{{font-weight:600}}tr:nth-child(even){{background:#eef2ec}}a{{color:#155e51}}li{{margin:.4em 0}}
@media print{{body{{background:white}}main{{padding:0}}.scroll{{overflow:visible}}tr{{break-inside:avoid}}}}
</style></head><body><main><header><h1>Conference trip options</h1>{escape(subtitle)}</header>
<p class="summary">{escape(report.summary)}</p>{''.join(sections)}</main></body></html>"""
