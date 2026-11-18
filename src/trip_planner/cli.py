import argparse
import os
import sys
from collections.abc import Sequence
from contextlib import ExitStack
from datetime import date
from functools import partial
from pathlib import Path

import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openrouter import ChatOpenRouter
from pydantic import ValidationError

from .agents import build_live_agents
from .models import TripRequest
from .search import Search, search_duckduckgo, search_tavily
from .workflow import build_workflow


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Find and plan a conference trip within your budget.")
    parser.add_argument("request", type=Path, help="Path to a JSON trip request")
    parser.add_argument("--provider", choices=("openai", "openrouter"), default="openai",
                        help="LLM provider (default: openai)")
    parser.add_argument("--search-provider", choices=("duckduckgo", "tavily"),
                        help="Web search provider (default: Tavily with a key, otherwise DuckDuckGo)")
    parser.add_argument("--model",
                        help="Model ID with tool calling (or set TRIP_PLANNER_MODEL)")
    args = parser.parse_args(argv)

    try:
        request = TripRequest.model_validate_json(args.request.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValidationError) as exc:
        parser.error(f"Invalid trip request. {exc}")

    try:
        load_dotenv(Path(".env"), override=False)
    except (OSError, UnicodeError):
        parser.error("Could not read .env. Check its permissions and UTF-8 encoding.")

    model_key = "OPENROUTER_API_KEY" if args.provider == "openrouter" else "OPENAI_API_KEY"
    tavily_key = os.environ.get("TAVILY_API_KEY")
    search_provider = args.search_provider or ("tavily" if tavily_key else "duckduckgo")
    required_keys = [model_key]
    if search_provider == "tavily":
        required_keys.append("TAVILY_API_KEY")
    missing = [key for key in required_keys if not os.environ.get(key)]
    if missing:
        parser.error(f"Set {', '.join(missing)} in .env or export them in your shell.")
    model_name = args.model or os.environ.get("TRIP_PLANNER_MODEL")
    if not model_name:
        parser.error("Specify --model or set TRIP_PLANNER_MODEL.")

    if args.provider == "openrouter":
        model = ChatOpenRouter(
            model=model_name, timeout=45_000, max_retries=2,
            openrouter_provider={"require_parameters": True},
        )
    else:
        model = ChatOpenAI(model=model_name, temperature=None, timeout=45, max_retries=2)
    with ExitStack() as stack:
        search: Search = search_duckduckgo
        if search_provider == "tavily":
            client = stack.enter_context(httpx.Client(
                base_url="https://api.tavily.com",
                headers={"Authorization": f"Bearer {tavily_key}"},
                timeout=25,
            ))
            search = partial(search_tavily, client=client)
        graph = build_workflow(build_live_agents(model, search))
        result = graph.invoke({"request": request, "today": date.today()})

    sys.stdout.write(result["plan"].model_dump_json(indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
