import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import VisaAgent
from schemas import BudgetBreakdown, ConferenceVisaWhitePaper, VisaRequirementDetails, VisaSearchInput


class EntryRuleShortcutTest(unittest.TestCase):
    def test_resident_of_destination_needs_nothing(self):
        paper = VisaAgent.entry_rule_shortcut("Czechia", "United Kingdom", "UK", "UK")
        self.assertFalse(paper.visa_details.visa_required)
        self.assertIn("domestic", paper.executive_summary)
        self.assertEqual(paper.financial_budget.total_visa_cost_usd, 0)
        self.assertEqual(paper.research_status, "rule_shortcut")

    def test_eu_citizen_within_eu_has_free_movement(self):
        paper = VisaAgent.entry_rule_shortcut("Czech", "Germany", "United Kingdom")
        self.assertFalse(paper.visa_details.visa_required)
        self.assertNotIn("90 days", paper.visa_details.allowed_stay_duration)

    def test_citizen_of_destination(self):
        paper = VisaAgent.entry_rule_shortcut("CZ", "Czech Republic")
        self.assertIn("citizenship", paper.executive_summary)

    def test_other_cases_go_to_the_model(self):
        self.assertIsNone(VisaAgent.entry_rule_shortcut("Indian", "Germany", "United Kingdom"))
        self.assertIsNone(VisaAgent.entry_rule_shortcut("Czechia", "United Kingdom", "Czechia"))
        self.assertIsNone(VisaAgent.entry_rule_shortcut("Czechia", "Atlantis"))
        self.assertIsNone(VisaAgent.entry_rule_shortcut("Indian", "UK", "UK"))
        self.assertIsNone(VisaAgent.entry_rule_shortcut("Indian", "UK", "UK", "France"))

    def test_shortcut_skips_model_call(self):
        with patch.object(VisaAgent, "_visa_agent") as call:
            VisaAgent.generate_visa_white_paper("Czechia", "France", "United Kingdom")
        call.assert_not_called()


class LangChainVisaTest(unittest.TestCase):
    def paper(self, urls):
        return ConferenceVisaWhitePaper(
            executive_summary="Fixture", applicant_nationality="India", destination_country="Germany",
            visa_details=VisaRequirementDetails(visa_required=True, visa_type="Example visa", allowed_stay_duration="Example duration"),
            required_documents=["Example document"], step_by_step_procedure=["Example step"],
            processing_time="Example time", recommended_timeline=[],
            financial_budget=BudgetBreakdown(visa_application_fee_usd=80,
                                            biometrics_or_service_fee_usd=20,
                                            mandatory_visa_insurance_fee_usd=10,
                                            total_visa_cost_usd=999, budget_notes="Fixture"),
            disclaimer="Fixture only", source_urls=urls,
        )

    def test_model_path_uses_structured_response_and_preserves_inputs(self):
        agent = Mock()
        agent.invoke.return_value = {"structured_response": self.paper(["https://example.org/visa"])}
        with patch.object(VisaAgent, "_visa_agent", return_value=agent):
            paper = VisaAgent.generate_visa_white_paper("India", "Germany", "UK", "France")
        prompt = agent.invoke.call_args.args[0]["messages"][0]["content"]
        self.assertEqual(VisaSearchInput.model_validate_json(prompt.split("\n", 1)[1]).origin_country, "France")
        self.assertEqual(paper.financial_budget.total_visa_cost_usd, 110)
        self.assertIn("not independently verified", paper.disclaimer)
        self.assertEqual(paper.research_status, "sources_reported")
        self.assertEqual(json.loads(paper.model_dump_json())["source_urls"], ["https://example.org/visa"])

    def test_without_sources_report_unknown_not_zero(self):
        agent = Mock()
        agent.invoke.return_value = {"structured_response": self.paper([])}
        with patch.object(VisaAgent, "_visa_agent", return_value=agent):
            paper = VisaAgent.generate_visa_white_paper("India", "Germany")
        self.assertIsNone(paper.visa_details.visa_required)
        self.assertIsNone(paper.financial_budget.total_visa_cost_usd)
        self.assertEqual(paper.research_status, "unverified")

    def test_unknown_fee_prevents_complete_total(self):
        budget = BudgetBreakdown(visa_application_fee_usd=80, total_visa_cost_usd=80,
                                 budget_notes="Other charges unknown")
        self.assertIsNone(budget.total_visa_cost_usd)

    def test_agent_factory_uses_shared_model_tools_middleware(self):
        VisaAgent._visa_agent.cache_clear()
        self.addCleanup(VisaAgent._visa_agent.cache_clear)
        with patch.object(VisaAgent, "search_model_and_tools", return_value=("model", ["search"])), \
                patch.object(VisaAgent, "create_agent") as create:
            VisaAgent._visa_agent()
        self.assertEqual(create.call_args.kwargs["model"], "model")
        self.assertEqual(create.call_args.kwargs["tools"], ["search"])
        self.assertEqual(create.call_args.kwargs["middleware"], [VisaAgent.pace_model_request])
        self.assertIs(create.call_args.kwargs["response_format"].schema, ConferenceVisaWhitePaper)

    def test_orchestrator_tool_uses_shared_input_and_json_output(self):
        from orchestrator import find_visa_requirements, conference_id
        runtime, conference = conference_fixture(country="Germany", city="Berlin")
        with patch("orchestrator.generate_visa_white_paper", return_value=self.paper([])) as generate:
            output = find_visa_requirements.func(conference_id(conference), "India", runtime,
                                                 residence_country="UK", origin_country="France")
        generate.assert_called_once_with("India", "Germany", "UK", "France")
        self.assertEqual(json.loads(output)["destination_country"], "Germany")


def conference_fixture(country="Slovakia", city="Stará Lesná, High Tatras", status="verified",
                       start="2027-06-08", end="2027-06-10"):
    from conference_agent import ExtendedConferenceOutput, ReviewedCandidate, ReviewedConference
    from langchain_core.messages import ToolMessage
    conference = ReviewedConference(
        name="20th International Symposium on Landscape Ecology (Landscape 20)",
        research_focus="Ecology", city=city, country=country, start_date=start, end_date=end,
        registration_status="unknown", official_website="https://example.org/event", source_urls=[])
    result = ExtendedConferenceOutput(conferences=[], search_summary="Fixture")
    if status == "verified":
        result.conferences.append(conference)
    else:
        getattr(result, f"{status}_candidates").append(ReviewedCandidate(
            conference=conference, status=status, reason="Fixture"))
    message = ToolMessage(content=result.model_dump_json(), name="find_conferences", tool_call_id="discovery")
    return SimpleNamespace(state={"messages": [message]}), conference


class OrchestratorConferenceIdTest(unittest.TestCase):
    def test_discovery_ids_and_canonical_handoffs(self):
        import orchestrator as o
        from conference_agent import ExtendedConferenceOutput
        runtime, conference = conference_fixture()
        original = ExtendedConferenceOutput.model_validate_json(runtime.state["messages"][0].content)
        with patch.object(o, "search", return_value=original):
            payload = json.loads(o.find_conferences.func("Ecology", conference.start_date, conference.end_date))
        identifier = payload["conferences"][0]["conference_id"]
        self.assertEqual(identifier, o.conference_id(conference))
        with patch.object(o, "plan_travel") as travel:
            o.find_itinerary.func(identifier, "New York", "USD", runtime)
        request = travel.call_args.args[0]
        self.assertEqual(request.destination, "Stará Lesná, High Tatras, Slovakia")
        self.assertEqual(request.event_start, conference.start_date)
        self.assertEqual(request.event_end, conference.end_date)
        with patch.object(o, "search_accommodation") as accommodation:
            o.find_accommodation.func(identifier, runtime)
        request = accommodation.call_args.args[0]
        self.assertEqual(request.city, conference.city)
        self.assertEqual(request.conference_name, conference.name)

    def test_unknown_and_unverified_ids_are_blocked(self):
        import orchestrator as o
        for state, identifier in [(conference_fixture()[0].state, "conf_imaginary"),
                                  (conference_fixture(status="unverified")[0].state,
                                   o.conference_id(conference_fixture(status="unverified")[1]))]:
            request = SimpleNamespace(state=state, tool_call={"name": "find_itinerary", "id": "call", "args": {"conference_id": identifier}})
            handler = Mock()
            result = o.guard_specialist_call(request, handler)
            self.assertEqual(json.loads(result.content)["status"], "blocked_by_eligibility")
            handler.assert_not_called()

    def test_provisional_allowed_and_no_eligible_report(self):
        import orchestrator as o
        runtime, conference = conference_fixture(status="provisional")
        identifier = o.conference_id(conference)
        self.assertIsNone(o.no_eligible_report(runtime.state["messages"]))
        self.assertEqual(o._resolve_conference(identifier, runtime.state).name, conference.name)
        runtime, _ = conference_fixture(status="unverified")
        report = o.no_eligible_report(runtime.state["messages"])
        self.assertEqual(report.travel, [])
        self.assertEqual(report.entry, [])

    def test_recursion_errors_still_contained(self):
        import orchestrator as o
        from langgraph.errors import GraphRecursionError
        runtime, conference = conference_fixture()
        identifier = o.conference_id(conference)
        request = SimpleNamespace(state=runtime.state, tool_call={"name": "find_itinerary", "id": "call", "args": {"conference_id": identifier}})
        response = o.guard_specialist_call(request, Mock(side_effect=GraphRecursionError("limit")))
        self.assertEqual(json.loads(response.content)["status"], "specialist_unavailable")
        request.state["messages"].append(response)
        retry = Mock()
        o.guard_specialist_call(request, retry)
        retry.assert_not_called()

    def test_one_day_stay_and_invalid_override(self):
        import orchestrator as o
        from datetime import timedelta
        runtime, conference = conference_fixture(end="2027-06-08")
        identifier = o.conference_id(conference)
        with patch.object(o, "search_accommodation") as accommodation:
            o.find_accommodation.func(identifier, runtime)
            self.assertEqual(accommodation.call_args.args[0].check_out, conference.start_date + timedelta(days=1))
            with self.assertRaises(o.PlanningInputError):
                o.find_accommodation.func(identifier, runtime, check_in=conference.start_date + timedelta(days=1))
            self.assertEqual(accommodation.call_count, 1)


if __name__ == "__main__":
    unittest.main()
