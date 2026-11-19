# AGENT.md: Guide for coding agents

This file is for AI coding agents (Claude Code, Codex, Cursor, etc.) working on
this repository. For installation and usage, see `README.md`.

## What this project is

TripPlanner is a Python multi-agent app. An orchestrator LLM agent calls four
specialist agents as tools: conference, accommodation, travel and visa. It returns
a structured `TripReport` that is rendered to HTML. The main entry point is
`orchestrator.py`.

## Repository map

| File | Key symbols | Notes |
|---|---|---|
| `orchestrator.py` | `orchestrator` (LangChain agent), tools `find_conferences` / `find_accommodation` / `find_itinerary` / `find_visa_requirements`, `main()` | The CLI (`--config`, `--output`, `--verbose`). Most behaviour rules live in its long system prompt. |
| `conference_agent.py` | `find_conferences`, `discovery_cache_path`, verification and pricing agents | Largest file (~1.7k lines). Discovery → official-site verification → registration pricing. Has several inner `recursion_limit`s. |
| `accommodation_agent.py` | `find_accommodation`, `enforce_price_evidence`, `enforce_fact_evidence` | `AccommodationSearchInput.from_conference(conf)` builds input from a conference. |
| `travel_agent.py` | `TravelRequest`, `TravelResult`, `plan_travel`, `select_itinerary` | `select_itinerary` filters the LLM's research by budget, dates and `max_trip_days` in plain Python. |
| `VisaAgent.py` | `generate_visa_white_paper`, `entry_rule_shortcut`, `visa_agent` | Uses **pydantic-ai**, not LangChain. Runs on one shared background event loop (`_run_on_shared_loop`) so that it is safe to call from worker threads. `entry_rule_shortcut` handles the citizen, resident and EU-to-EU cases without an LLM call. |
| `schemas.py` | `ConferenceSearchInput`, `Conference` | Shared conference contracts. |
| `trip_report.py` | `TripConfig`, `TripReport`, `ConferenceRow`, `TravelRow`, `EntryRow`, `Note`, `render_report` | `TripConfig` is the `--config` JSON schema (`extra="forbid"`). The HTML is escaped here. |
| `search_config.py` | `search_provider()`, `search_model_and_tools()` | Returns a Tavily tool or an OpenRouter model with the web plugin. Always get search through this function. |
| `model_quota.py` | `pace_model_request` (LangChain middleware), `call_with_quota` | A process-wide RPM limiter with 429 back-off. |
| `tests/` | `test_*.py` | unittest. Offline, with mocks. |
| `request.json` | | Example `--config` input. |
| `reports/` | | Example HTML output. |

## Stack

- LangChain v1 `create_agent` with `ToolStrategy(<PydanticModel>)` for structured output.
- pydantic-ai (visa agent only), Pydantic v2 for every input and output contract.
- Models are called through OpenRouter (`MODEL=openrouter:<provider>/<model>`).
- `python-dotenv`. Modules call `load_dotenv()` and read `MODEL` **at import time**.

## Environment

- Use the project venv: `.venv/bin/python`. Install with `.venv/bin/pip install -r requirements.txt`.
- `.env` holds `OPENROUTER_API_KEY`, `MODEL`, and optionally `TAVILY_API_KEY`,
  `SEARCH_PROVIDER` and `TRIPPLANNER_MODEL_RPM`. Never print, log or commit its contents.
- Because modules read `MODEL` on import, tests must patch the environment before
  importing them:
  ```python
  with patch.dict(os.environ, {"MODEL": "openrouter:example/model", "OPENROUTER_API_KEY": "test"}):
      import VisaAgent
  ```

## Testing

```bash
.venv/bin/python -m unittest discover -s tests
```

- pytest is **not** installed. Write `unittest.TestCase` tests in `tests/test_*.py`.
- Tests must not make network or LLM calls. Mock the agent `.invoke` / `run_sync`
  or the HTTP fetch. Test deterministic logic directly, for example `select_itinerary`,
  `entry_rule_shortcut`, `render_report` and the evidence enforcers.
- Run the suite before and after every change and report failures as they are.

## End-to-end runs (real API calls)

- They **cost money** and the results **differ between runs**. Ask the user before
  running them, and compare several runs before drawing conclusions.
- Command: `.venv/bin/python orchestrator.py --config <file>.json --output <file>.html --verbose`.
  Put temporary configs and reports in your scratchpad or temp directory, not in the repo.
- `TRIPPLANNER_REPLAY_DISCOVERY=1` replays the last saved raw conference-discovery
  results for an identical request (cached under `$TMPDIR/tripplanner-discovery/`).
  This lets you debug verification without searching again.
- A run can take several minutes. The orchestrator uses `recursion_limit=30`.

## Domain rules to preserve

These rules come from user requirements. Keep them when you edit prompts or code:

- **Never turn an unknown cost into 0 or null without saying so.** Label every price
  `confirmed`, `estimated` or `unknown`, and say why a fee could not be found.
- Do not present estimates (for example route-planner fares) as confirmed prices.
- The budget covers the **whole trip**. If registration is unknown, say affordability
  is *unverified*, never "within budget".
- Never count lodging twice. That includes venue-provided lodging plus a hotel.
- When the orchestrator passes on specialist results, keep their status, caveats
  and source URLs. Put URLs in `Note.url`, not in the note text.
- Never infer nationality from the departure city. Pass `residence_country` to the visa tool.
- Both search paths (Tavily and OpenRouter hosted search) must apply the same
  conference verification. Do not add checks to only one of them.
- Every new model call must go through `model_quota` (`pace_model_request`
  middleware or `call_with_quota`), so the shared rate limit is respected.
- When you change a Pydantic schema, update the prompts that describe it in the same change.

## Known open issues

- Venue-provided lodging is not marked, so it can be double-counted.
- The orchestrator's accommodation check-out is end date + 1, but the itinerary uses the end date.
- Report honesty: "within budget" can still appear when registration is unknown,
  and implausible prices are not flagged.
- Aggregator or predatory conference sites are sometimes shown as official.
- Tool errors reach the orchestrator without detail, so it may guess the cause.
- `tests/test_travel_agent.py` expects `TravelResult.rejected_options`, which
  `travel_agent.py` does not define yet. Two tests currently fail.

## Git workflow

- Work on a feature branch and open PRs against `main`.
- Commit only when the user asks.
- Never commit `.env`, `.venv/`, `__pycache__/`, generated reports or personal trip configs.
