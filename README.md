# TripPlanner

For the preliminary design of the workflow, check the conference_agent_workflow_corrected.pdf file. 

## Installation
To test the agents, install the requirements into a fresh virtual environment:

```bash
# from the repository root
python3 -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Then create a `.env` file in the repository root with your keys and model choice:

```
OPENROUTER_API_KEY=...
MODEL=...                        # an OpenRouter model id, e.g. openrouter:openai/gpt-4.1-mini
TAVILY_API_KEY=...               # optional, see below
SEARCH_PROVIDER=...              # optional: tavily or openrouter
```

### Web search
The specialist agents can search the web in two ways (configured in `search_config.py`):
- **Tavily** — the agent calls a Tavily search tool and can run several targeted searches. Needs `TAVILY_API_KEY`.
- **OpenRouter** — OpenRouter's built-in web search, configured by `search_config.py`. Needs only `OPENROUTER_API_KEY`; web results are billed by OpenRouter on top of model usage.

If `SEARCH_PROVIDER` is not set, Tavily is used when `TAVILY_API_KEY` is present, otherwise OpenRouter.

Run an agent with the virtual environment active, e.g.:

```bash
python conference_agent.py
python accommodation_agent.py
python VisaAgent.py
python orchestrator.py
```

Run `deactivate` to leave the virtual environment.

## Agents
- `conference_agent.py` — finds academic conferences matching a field, date range and region.
- `accommodation_agent.py` — finds accommodation in the conference's country (near its city/venue) for the exact conference dates. Build its input from a conference result with `AccommodationSearchInput.from_conference(conference)`.
- `travel_agent.py` — compares transport and lodging itineraries.
- `VisaAgent.py` — researches entry requirements, visa fees and procedures with LangChain; uses deterministic shortcuts for citizenship, domestic trips and free movement. The same `generate_visa_white_paper()` entry point is retained. Visa models now live in `schemas.py`; unknown fees are null, and sources are research leads rather than independently verified advice.
- `orchestrator.py` — coordinates all four specialists and renders the HTML trip report.

All agents use LangChain and Pydantic V2. PydanticAI is no longer required.
When upgrading an existing environment, run `python -m pip install -r requirements.txt`
and `python -m pip check`. Removing a dependency from requirements does not uninstall it;
if no other project uses the old libraries in this environment, optionally run
`python -m pip uninstall pydantic-ai pydantic-ai-slim`.

### Tests (no API charges)

From the repository root, with the virtual environment active:

```bash
MODEL=openrouter:example/model OPENROUTER_API_KEY=test TAVILY_API_KEY=test SEARCH_PROVIDER=tavily python -m unittest discover -s tests -v
```

`tests/test_registration.py` is a separate live integration script, not an offline unit test.
Run it intentionally from the root with real keys: `PYTHONPATH=. python tests/test_registration.py`.

### Shared model rate limit

The orchestrator and its specialists share a request schedule within one running
Python process. The default is 20 model calls per minute. Set
`TRIPPLANNER_MODEL_RPM` in `.env` to match your provider limit, or set it lower
if other applications use the same account. HTTP 429 responses pause the shared
schedule for the provider's `Retry-After` interval (60 seconds if absent).
The affected call gets up to two retries.

### HTML trip report

Run the built-in example with `.venv/bin/python orchestrator.py`, or supply your
own JSON request and output path:

```sh
.venv/bin/python orchestrator.py --config ~/Downloads/request.json --output reports/trip.html
```

Open the resulting HTML file in a browser. The default output is
`trip_report.html` in the current directory. Add `--verbose` for specialist
diagnostics. The config uses `field`, `origin_city`, `origin_country`,
`passport_country`, `residence_country`, `window_start`, `window_end`,
`max_trip_days`, `destination_countries`, `budget`, and `currency`.
Dates use YYYY-MM-DD. Optional lists are `unavailable_dates` and `preferences`.
