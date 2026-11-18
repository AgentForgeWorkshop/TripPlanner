"""Choose how specialist agents search the web.

Two providers are supported:
- "tavily": the agent gets a Tavily search tool and decides what
  to search for (needs TAVILY_API_KEY).
- "openrouter": OpenRouter's web search plugin (Exa engine), which
  searches before each model call (needs only OPENROUTER_API_KEY).

Set SEARCH_PROVIDER in .env to choose explicitly. If it is not set,
Tavily is used when TAVILY_API_KEY exists, otherwise OpenRouter.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def search_provider() -> str:
    provider = os.getenv("SEARCH_PROVIDER", "").strip().lower()
    if not provider:
        provider = "tavily" if os.getenv("TAVILY_API_KEY") else "openrouter"
    if provider not in ("tavily", "openrouter"):
        raise ValueError(
            f"Unknown SEARCH_PROVIDER '{provider}'. "
            "Use 'tavily' or 'openrouter'."
        )
    return provider


def search_model_and_tools(max_results: int = 8):
    """Return (model, tools) for a web-searching specialist agent.

    The model is either a model id string (Tavily) or a configured
    chat model (OpenRouter); create_agent accepts both.
    """
    model = os.environ["MODEL"]

    if search_provider() == "tavily":
        # Imported here so Tavily is only required when used.
        from langchain_tavily import TavilySearch

        return model, [
            TavilySearch(max_results=max_results, search_depth="basic")
        ]

    if not model.startswith("openrouter:"):
        raise ValueError(
            "OpenRouter web search needs an OpenRouter model, "
            "e.g. MODEL=openrouter:openai/gpt-4.1-mini"
        )

    # The "exa" engine runs the search on OpenRouter's side and
    # injects the results into the prompt before every model call.
    # (The ":online" suffix uses the provider's native search, which
    # the model skips when the agent's structured-output tool is bound.)
    from langchain.chat_models import init_chat_model

    return init_chat_model(
        model,
        plugins=[{"id": "web", "engine": "exa", "max_results": max_results}],
    ), []
