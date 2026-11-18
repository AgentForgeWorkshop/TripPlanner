import json
from contextlib import chdir, redirect_stderr, redirect_stdout
from datetime import date
from decimal import Decimal
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
from urllib.parse import quote

import httpx
from ddgs.exceptions import DDGSException
from langchain_openai import ChatOpenAI

from trip_planner.agents import Research, ResearchError, Specialists
from trip_planner.cli import main
from trip_planner.models import Conference, Cost, Discovery, Entry, Lodging, Source, Transport, Travel, TripPlan, TripRequest
from trip_planner.search import SearchError, search_duckduckgo
from trip_planner.workflow import build_workflow

TODAY = date(2030, 4, 1)
EVENT_URL = "https://conference.example.com/event"
TRAVEL_URL = "https://travel.example.com/rates"
ENTRY_URL = "https://entry.example.com/requirements"


def cost(amount: str, url: str) -> Cost:
    return Cost(amount=Decimal(amount), currency="SGD", source_urls=[url], basis="Fixture estimate")


class ConferenceWorkflowTest(TestCase):
    def setUp(self):
        self.request = TripRequest(
            field="Materials science", origin_city="Singapore", origin_country="SG", passport_country="IN",
            window_start=date(2030, 4, 10), window_end=date(2030, 4, 25),
            budget=Decimal("900"), currency="SGD", max_trip_days=5,
        )
        event = Conference(
            name="Materials Research Forum", city="Kuala Lumpur", country="MY", venue="Conference Hall",
            start=date(2030, 4, 15), end=date(2030, 4, 16), registration_deadline=date(2030, 4, 12),
            registration_status="open", relevance="Materials science program",
            registration=cost("220.10", EVENT_URL), source_urls=[EVENT_URL],
        )
        self.discovery = Discovery(conferences=[event], notes=[])
        self.travel = Travel(
            transport=Transport(
                mode="flight", outbound="Nonstop flight from Singapore", inbound="Nonstop flight to Singapore",
                transit_countries=[], depart_origin=date(2030, 4, 14), arrive_destination=date(2030, 4, 14),
                depart_destination=date(2030, 4, 17), arrive_origin=date(2030, 4, 17),
                cost=cost("180.15", TRAVEL_URL), source_urls=[TRAVEL_URL],
            ),
            lodging=Lodging(
                name="Venue Hotel", check_in=date(2030, 4, 14), check_out=date(2030, 4, 17),
                cost=cost("250.05", TRAVEL_URL), source_urls=[TRAVEL_URL],
            ),
            local_transport=cost("35.10", TRAVEL_URL), meals=cost("95.60", TRAVEL_URL), notes=[],
        )
        self.entry = Entry(
            status="visa_exempt", requirements=["Bring proof of onward travel"], processing_time=None,
            processing_days_upper=None, processing_source_urls=[], cost=cost("0", ENTRY_URL),
            source_urls=[ENTRY_URL], notes=[],
        )

    def plan(self, *, specialists=None, request=None):
        if specialists is None:
            specialists = Specialists(
                discover=lambda *_: Research(self.discovery, {EVENT_URL: Source(url=EVENT_URL, title="Event")}),
                travel=lambda *_: Research(self.travel, {TRAVEL_URL: Source(url=TRAVEL_URL, title="Travel")}),
                entry=lambda *_: Research(self.entry, {ENTRY_URL: Source(url=ENTRY_URL, title="Entry")}),
            )
        return build_workflow(specialists).invoke({"request": request or self.request, "today": TODAY})["plan"]

    def test_specialists_use_search_and_join_into_a_plan(self):
        fixtures = {
            "Discovery": (self.discovery, EVENT_URL), "Travel": (self.travel, TRAVEL_URL),
            "Entry": (self.entry, ENTRY_URL),
        }

        def fake_model(request, model):
            payload = json.loads(request.content)
            self.assertEqual(payload["model"], model)
            names = {tool["function"]["name"] for tool in payload["tools"]}
            self.assertIn("search_web", names)
            schema_name, = names - {"search_web"}
            context = json.loads(next(msg["content"] for msg in payload["messages"] if msg["role"] == "user"))
            self.assertEqual("passport_country" in context["request"], schema_name == "Entry")
            if payload["messages"][-1]["role"] == "tool":
                name, arguments = schema_name, fixtures[schema_name][0].model_dump(mode="json")
            else:
                name, arguments = "search_web", {"query": schema_name}
            return httpx.Response(200, request=request, json={
                "id": "chatcmpl-fixture", "object": "chat.completion", "model": model, "created": 0,
                "system_fingerprint": "fixture",
                "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                    "role": "assistant", "content": None,
                    "tool_calls": [{"id": f"call_{name}", "type": "function", "function": {
                        "name": name, "arguments": json.dumps(arguments),
                    }}],
                }}],
            })

        def fake_search(request):
            self.assertEqual((request.method, request.url.path), ("POST", "/search"))
            name = json.loads(request.content)["query"]
            return httpx.Response(200, request=request, json={"results": [{
                "url": fixtures[name][1], "title": name, "content": "Excerpt used by the fixture.",
            }]})

        def run_cli(provider, model, key_name, endpoint, *, shell_tavily=False,
                    file_tavily=True, search_provider=None):
            def fake_http(client, request, **kwargs):
                if request.url.host == "api.tavily.com":
                    self.assertEqual(request.headers["Authorization"], "Bearer tavily-test-key")
                    return fake_search(request)
                self.assertEqual(str(request.url), endpoint)
                self.assertEqual(request.headers["Authorization"], "Bearer model-test-key")
                if provider == "openrouter":
                    self.assertTrue(json.loads(request.content)["provider"]["require_parameters"])
                return fake_model(request, model)

            def fake_duckduckgo(query, *, backend, max_results):
                self.assertEqual((backend, max_results), ("duckduckgo", 5))
                href = f"//duckduckgo.com/l/?uddg={quote(fixtures[query][1], safe='')}&rut=fixture"
                return [{"title": query, "href": href, "body": "DuckDuckGo fixture excerpt."}]

            output = StringIO()
            shell = {"TAVILY_API_KEY": "tavily-test-key"} if shell_tavily else {}
            file_key = "overridden-key" if shell_tavily else "tavily-test-key"
            args = ["request.json", "--provider", provider]
            if search_provider:
                args.extend(["--search-provider", search_provider])
            with (
                TemporaryDirectory() as directory,
                chdir(directory),
                httpx.Client() as openai_client,
                patch.dict("os.environ", shell, clear=True),
                patch("trip_planner.cli.date") as clock,
                patch("trip_planner.cli.ChatOpenAI",
                      side_effect=lambda **kwargs: ChatOpenAI(http_client=openai_client, **kwargs)),
                patch.object(httpx.Client, "send", autospec=True, side_effect=fake_http),
                patch("trip_planner.search.DDGS") as ddgs,
                redirect_stdout(output),
            ):
                Path("request.json").write_text(self.request.model_dump_json(), encoding="utf-8")
                settings = f"{key_name}=model-test-key\nTRIP_PLANNER_MODEL={model}\n"
                if file_tavily:
                    settings += f"TAVILY_API_KEY={file_key}\n"
                Path(".env").write_text(settings, encoding="utf-8")
                duck_search = ddgs.return_value.text
                duck_search.side_effect = fake_duckduckgo
                clock.today.return_value = TODAY
                self.assertEqual(main(args), 0)
                if search_provider == "duckduckgo" or not (shell_tavily or file_tavily):
                    self.assertEqual(duck_search.call_count, 3)
                else:
                    duck_search.assert_not_called()
            return TripPlan.model_validate_json(output.getvalue())

        openai = ("openai", "gpt-5-mini", "OPENAI_API_KEY", "https://api.openai.com/v1/chat/completions")
        openrouter = ("openrouter", "openai/gpt-5-mini", "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1/chat/completions")
        for provider, settings in (
            (openai, {}),
            (openrouter, {"file_tavily": False}),
            (openrouter, {"search_provider": "duckduckgo"}),
        ):
            with self.subTest(provider=provider[0], settings=settings):
                plan = run_cli(*provider, **settings)
                self.assertEqual(plan.status, "within_estimated_budget", plan.findings)
                option, = plan.options
                self.assertEqual(option.budget.estimated_total, Decimal("781.00"))
                self.assertEqual(option.budget.remaining, Decimal("119.00"))
                self.assertEqual([day.date.day for day in option.itinerary], [14, 15, 16, 17])
                self.assertIn("Attend Materials Research Forum at Conference Hall.", option.itinerary[1].activities)
                self.assertEqual({source.url for source in plan.sources}, {EVENT_URL, TRAVEL_URL, ENTRY_URL})

        # A model-generated citation must have been seen by the specialist that cites it.
        fixtures["Entry"] = (self.entry, TRAVEL_URL)
        plan = run_cli(*openai, shell_tavily=True)
        self.assertEqual(plan.status, "needs_verification")
        self.assertIsNone(plan.options[0].entry)
        self.assertIn("URL that its search did not return", plan.findings[0].detail)

    def test_openrouter_reports_the_required_key(self):
        error = StringIO()
        with (
            TemporaryDirectory() as directory,
            chdir(directory),
            patch.dict("os.environ", {"OPENAI_API_KEY": "test-key", "TAVILY_API_KEY": "test-key"}, clear=True),
            redirect_stderr(error),
        ):
            path = Path(directory) / "request.json"
            path.write_text(self.request.model_dump_json(), encoding="utf-8")
            with self.assertRaises(SystemExit) as result:
                main([str(path), "--provider", "openrouter", "--model", "openai/gpt-5-mini"])
            self.assertEqual(result.exception.code, 2)
        self.assertIn("OPENROUTER_API_KEY", error.getvalue())

    def test_explicit_tavily_requires_its_key(self):
        error = StringIO()
        with (
            TemporaryDirectory() as directory,
            chdir(directory),
            patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}, clear=True),
            redirect_stderr(error),
        ):
            Path("request.json").write_text(self.request.model_dump_json(), encoding="utf-8")
            with self.assertRaises(SystemExit) as result:
                main(["request.json", "--provider", "openrouter", "--search-provider", "tavily"])
            self.assertEqual(result.exception.code, 2)
        self.assertIn("TAVILY_API_KEY", error.getvalue())

    def test_duckduckgo_empty_response_reports_search_failure(self):
        with patch("trip_planner.search.DDGS") as ddgs:
            ddgs.return_value.text.side_effect = DDGSException("No results found.")
            with self.assertRaisesRegex(SearchError, "DuckDuckGo could not return search results"):
                search_duckduckgo("conference query")

    def test_missing_or_mismatched_prices_and_visa_lead_time(self):
        self.travel.meals = None
        self.travel.lodging.cost = self.travel.lodging.cost.model_copy(update={"currency": "MYR"})
        self.entry.status = "visa_required"
        self.entry.processing_time = "Up to 21 calendar days"
        self.entry.processing_days_upper = 21
        self.entry.processing_source_urls = [ENTRY_URL]
        option, = self.plan().options

        self.assertEqual(option.status, "needs_verification")
        self.assertIsNone(option.budget.estimated_total)
        self.assertEqual(option.budget.known_subtotal, Decimal("435.35"))
        self.assertTrue(any("upper bound exceeds" in issue for issue in option.issues))

    def test_filters_travel_day_blackout_and_over_budget(self):
        for request, reason, status in [
            (self.request.model_copy(update={"unavailable_dates": [date(2030, 4, 14)]}),
             "overlap an unavailable day", "needs_verification"),
            (self.request.model_copy(update={"budget": Decimal("750")}),
             "exceed the budget", "no_match"),
        ]:
            with self.subTest(reason=reason):
                plan = self.plan(request=request)
                self.assertEqual(plan.status, status)
                self.assertEqual(plan.options, [])
                self.assertIn(reason, plan.findings[0].detail)

    def test_discovery_failure_skips_other_specialists(self):
        def failed_discovery(*_):
            raise ResearchError("Search provider unavailable.")

        def unexpected_call(*_):
            self.fail("The specialist was called without a conference.")

        plan = self.plan(specialists=Specialists(failed_discovery, unexpected_call, unexpected_call))
        self.assertEqual(plan.status, "needs_verification")
        self.assertEqual(plan.options, [])
        self.assertIn("Search provider unavailable", plan.findings[0].detail)
