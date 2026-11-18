# TripPlanner
Repository for a group project on a trip planner

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
- **OpenRouter** — OpenRouter's built-in web search, enabled by adding `:online` to the model id. Needs only `OPENROUTER_API_KEY`; web results are billed by OpenRouter on top of model usage.

If `SEARCH_PROVIDER` is not set, Tavily is used when `TAVILY_API_KEY` is present, otherwise OpenRouter.

Run an agent with the virtual environment active, e.g.:

```bash
python conference_agent.py
python accommodation_agent.py
python orchestrator.py
```

Run `deactivate` to leave the virtual environment.

## Agents
- `conference_agent.py` — finds academic conferences matching a field, date range and region.
- `accommodation_agent.py` — finds accommodation in the conference's country (near its city/venue) for the exact conference dates. Build its input from a conference result with `AccommodationSearchInput.from_conference(conference)`.
- `orchestrator.py` — coordinates both specialists: conference search first, then accommodation using the conference's country and dates.
