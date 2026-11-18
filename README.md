# TripPlanner

Find conferences in your field and get a sourced trip itinerary with a whole-trip
budget and passport-specific entry research. The Python CLI implements the three
specialist roles in [the workflow draft](conference_agent_workflow_corrected.pdf).

## Run it

For a fresh checkout, install the planner using Python 3.11 or later.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Run the CLI from the project directory. If you need a local `.env` file, copy
[`.env.example`](.env.example) to `.env` and fill in the API key for your model
provider. Git ignores `.env`. Exported shell variables take precedence over
values in the file.

For direct OpenAI access, fill in `OPENAI_API_KEY`.

```sh
conference-trip examples/request.json --model gpt-5-mini > trip-plan.json
```

For OpenRouter, fill in `OPENROUTER_API_KEY` and use the full model ID from its catalog.

```sh
conference-trip examples/request.json --provider openrouter --model openai/gpt-5-mini > trip-plan.json
```

You can set `TRIP_PLANNER_MODEL` in place of `--model`. Choose a model that supports
tool calling. The selected model powers all three specialists.
[OpenRouter tool calling](https://openrouter.ai/docs/guides/features/tool-calling)

The CLI uses DuckDuckGo when `TAVILY_API_KEY` is unset or blank. When the key is
present it uses Tavily by default. Set `--search-provider duckduckgo` to use
DuckDuckGo explicitly, or `--search-provider tavily` to require Tavily.
DuckDuckGo uses public search results through DDGS and requires no search API key.
DuckDuckGo can challenge automated searches. If it returns a challenge, the
planner reports that research could not be completed.
[DDGS backends](https://github.com/deedy5/ddgs#3-supported-backends) ·
[Tavily Search API](https://docs.tavily.com/documentation/api-reference/endpoint/search)

```sh
conference-trip examples/request.json --provider openrouter --model openai/gpt-5-mini --search-provider duckduckgo
```

Copy [the example request](examples/request.json) and update the field, departure
city and country, passport country, inclusive travel window, budget, and currency.
Use two-letter ISO country codes, a three-letter currency code, and `YYYY-MM-DD`
local dates. The budget is for one traveler in the requested currency. Optional
fields cover residence country, existing entry permission, unavailable dates,
maximum trip length, destination countries, and preferences. Use a brief
description for existing entry permission and omit passport numbers and other
document identifiers. Dates are evaluated against the local date of the machine
running the CLI.

## How the workflow runs

1. The **conference specialist** researches up to three relevant in-person
   conferences, including registration availability and fees.
2. The **travel specialist** researches a round trip, accommodation, local
   transport, and meals for each candidate. In parallel, the **country information
   specialist** checks applicable entry rules, fees, and current visa or
   authorization processing times using official sources.
3. The **LangGraph orchestrator** waits for both branches, checks the calendar,
   totals sourced cost estimates, and produces ranked options with day-by-day
   itineraries. The output links to sources and reports any research gaps.
   [LangGraph parallel execution](https://docs.langchain.com/oss/python/langgraph/graph-api-control-flow#run-nodes-in-parallel)

The planner evaluates one travel option per conference and supports routes
without third-country transit. Travel prices are web estimates, and the output
records the basis and sources of each price. Check current prices, availability,
and entry rules before booking.

| Result status | Meaning |
| --- | --- |
| `within_estimated_budget` | At least one itinerary fits the calendar and budget with estimates for every cost category and no unresolved issue flagged by the workflow. |
| `needs_verification` | Research is incomplete, or the itinerary depends on an unconfirmed cost, registration status, or entry requirement. `findings` and each option's `issues` explain what to verify. |
| `no_match` | The search found no qualifying conference, or all evaluated options were ruled out. |

When a cost is missing, `known_subtotal` sums only the available estimates and
`estimated_total` is `null`. A missing amount and a sourced zero have different
meanings. Decimal amounts are serialized as JSON strings.

## Verify locally

```sh
python -m unittest discover -s tests -v
```

The tests exercise the LangGraph workflow and live agent wiring with mocked
provider responses. They run without provider keys or network access.
