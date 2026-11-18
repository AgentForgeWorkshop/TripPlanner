# TripPlanner

The orchestrator delegates conference discovery and itinerary research to specialist agents. For a known conference, you can ask directly for round-trip transport and a place to stay. The preliminary design is in the [workflow PDF](conference_agent_workflow_corrected.pdf).

## Environment setup

From the project directory, create and activate a virtual environment, then install the dependencies. These commands are for macOS or Linux with Python 3.13.

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

If this checkout already has the populated `.venv`, only the activation command is needed in a new shell.

Create a file named `.env` in the project directory. For OpenRouter, fill in your credentials and a model ID that supports tool calling. Both specialists use Tavily for web search.

```dotenv
MODEL=openrouter:your-provider/your-model
OPENROUTER_API_KEY=your-openrouter-key
TAVILY_API_KEY=your-tavily-key
```

## Plan a trip

```sh
python orchestrator.py
```

At the prompt, try this example with your own dates and destination.

> Find a round-trip itinerary from Singapore for a conference in Bangkok on March 16–18, 2027. I can leave March 15 and must return by March 19. My transport and lodging budget is SGD 900 for one person. Prefer nonstop flights and minimize total cost.

Include the origin, destination or conference search criteria, event dates, and currency. You can specify unavailable dates, maximum trip length, and lodging or transport preferences. Ask for the shortest travel time to rank by combined outbound and return duration. The default ranks by total transport and lodging cost.

The specialist compares the options it finds. A recommendation includes the route, stay, sources, and any pricing assumptions. When a budget is given, a partially priced option can be returned for verification if no fully priced option qualifies. Registration and other conference expenses are separate from the travel estimate.
