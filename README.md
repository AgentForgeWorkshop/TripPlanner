# TripPlanner

TripPlanner is a multi-agent assistant for planning academic conference trips.
You give it a research field, a travel window, your departure city, your passport
and a budget. It then:

1. finds matching in-person conferences and checks them against their official websites,
2. plans the round-trip travel and accommodation for the conference dates,
3. looks up visa and entry requirements for your nationality and residence,
4. writes an HTML report comparing the options.

The original workflow design is in `conference_agent_workflow_corrected.pdf`.

> **Results are AI-generated.** Before booking anything, check every price,
> date and visa rule against the official sources linked in the report.

---

## Requirements

- Python **3.10 or newer** (developed with 3.13)
- An [OpenRouter](https://openrouter.ai) API key
- Optional: a [Tavily](https://tavily.com) API key, which gives better web search

## Installation

```bash
git clone <repository-url> TripPlanner
cd TripPlanner

python3 -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Run `deactivate` to leave the virtual environment.

## Configuration

Create a `.env` file in the repository root:

```dotenv
OPENROUTER_API_KEY=sk-or-...
MODEL=openrouter:openai/gpt-4.1-mini
TAVILY_API_KEY=tvly-...          # optional
SEARCH_PROVIDER=tavily           # optional: tavily or openrouter
TRIPPLANNER_MODEL_RPM=20         # optional
```

### Web search
The specialist agents can search the web in two ways (configured in `search_config.py`):
- **Tavily** — the agent calls a Tavily search tool and can run several targeted searches. Needs `TAVILY_API_KEY`.
- **OpenRouter** — OpenRouter's built-in web search, configured by `search_config.py`. Needs only `OPENROUTER_API_KEY`; web results are billed by OpenRouter on top of model usage.
=======
| Variable | Required | Meaning |
|---|---|---|
| `OPENROUTER_API_KEY` | yes | Key for model calls through OpenRouter. |
| `MODEL` | yes | OpenRouter model id with the `openrouter:` prefix, e.g. `openrouter:openai/gpt-4.1-mini`. |
| `TAVILY_API_KEY` | no | Enables Tavily search. |
| `SEARCH_PROVIDER` | no | `tavily` or `openrouter`. If unset, Tavily is used when `TAVILY_API_KEY` exists, otherwise OpenRouter. |
| `TRIPPLANNER_MODEL_RPM` | no | Maximum model calls per minute, shared by all agents in one run. Default `20`. |

**Rate limits:** if the provider returns HTTP 429, all agents pause for its
`Retry-After` interval (60 seconds if none is given), and the call is retried up to
twice. Lower `TRIPPLANNER_MODEL_RPM` if other applications share your account.

`.env` is git-ignored. Never commit your API keys.


## Usage

Activate the virtual environment first, or prefix commands with `.venv/bin/`.

### Quick start

```bash

python conference_agent.py
python accommodation_agent.py
python VisaAgent.py
python orchestrator.py
```

This runs a built-in example request (an ecology conference, departing New York)
and writes `trip_report.html` to the current directory. Open the file in a browser.

### Plan your own trip

Describe your trip in a JSON file (see `request.json`):

```json
{
  "field": "Artificial intelligence for chemistry and materials science",
  "origin_city": "Chicago",
  "origin_country": "US",
  "passport_country": "India",
  "residence_country": "US",
  "window_start": "2027-01-01",
  "window_end": "2027-06-30",
  "unavailable_dates": ["2027-05-04", "2027-05-05"],
  "max_trip_days": 7,
  "destination_countries": ["SG", "MY", "TH", "ID", "VN", "US", "JP"],
  "preferences": ["Prefer public transport when practical", "Prefer accommodation near the venue"],
  "budget": "2500.00",
  "currency": "USD"
}
```

Then run:

```bash
python orchestrator.py --config request.json --output reports/trip.html
```

| Option | Meaning |
|---|---|
| `--config PATH` | Trip criteria JSON. Without it, the built-in example is used. |
| `--output PATH` | Where to write the HTML report (default `trip_report.html`). Folders are created as needed. |
| `--verbose` | Print specialist diagnostics (searches, verification decisions). |

A run usually takes several minutes and makes many billed model and search calls.

#### Request file fields

| Field | Required | Description |
|---|---|---|
| `field` | yes | Research area of the conference. |
| `origin_city` | yes | City you travel from. |
| `origin_country` | yes | Country you travel from (name or code, e.g. `UK`). |
| `passport_country` | yes | Your nationality. Used for visa checks. |
| `residence_country` | yes | Country where you live. Residence can change entry rules. |
| `window_start`, `window_end` | yes | Dates the conference must fall between (`YYYY-MM-DD`). |
| `unavailable_dates` | no | Dates you cannot attend. Conferences on these dates are excluded. |
| `max_trip_days` | yes | Longest trip you accept, travel days included. |
| `destination_countries` | yes | Countries or regions to search, e.g. `["DE", "FR"]` or `["Europe"]`. |
| `preferences` | no | Free-text preferences, e.g. public transport or proximity to the venue. |
| `budget` | yes | **Whole-trip** budget: registration, travel and lodging. |
| `currency` | yes | Three-letter currency code, e.g. `USD`, `GBP`, `EUR`. |

Unknown keys are rejected, so check the spelling if the file fails to load.

### Run a single specialist

Each specialist runs a small built-in demo, which is useful for checking your setup:

```bash
python conference_agent.py      # conference search
python accommodation_agent.py   # conference search, then accommodation
python VisaAgent.py             # visa white paper (Indian passport, Germany)
```

## How it works

```
orchestrator.py ── find_conferences ──► conference_agent.py   (search + verify on official sites)
       │──────── find_accommodation ──► accommodation_agent.py
       │──────── find_itinerary ──────► travel_agent.py       (transport + lodging vs. budget)
       │──────── find_visa_requirements ► VisaAgent.py
       ▼
  TripReport ──► trip_report.render_report ──► HTML file
```

| Module | Role |
|---|---|
| `orchestrator.py` | Command-line entry point. Coordinates the specialists and produces the structured report. |
| `conference_agent.py` | Finds conferences, checks them against official pages, labels them verified, provisional or unverified, and looks up registration fees. |
| `accommodation_agent.py` | Finds places to stay near the venue for the conference dates. |
| `travel_agent.py` | Researches transport options and picks an itinerary that fits the budget and trip length. |
| `VisaAgent.py` | Visa and entry requirements, steps, timeline and fees. |
| `trip_report.py` | Request (`TripConfig`) and report (`TripReport`) models, plus the HTML renderer. |
| `schemas.py` | Shared conference search input and conference models. |
| `search_config.py` | Chooses Tavily or OpenRouter web search. |
| `model_quota.py` | Shared rate limiting and 429 retries. |

## Running tests

The tests use the standard library `unittest`, mock all model calls, and need no
API keys or network access:

```bash
python -m unittest discover -s tests
```

## Limitations

- Results differ between runs, because web search and model output are not deterministic.
- Prices are labelled *confirmed*, *estimated* or *unknown*. Estimates are marked with ≈.
- Visa information is guidance only. Check it with the destination country's official sources.
