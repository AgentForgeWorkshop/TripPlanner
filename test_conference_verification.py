import json
import os
import unittest
from unittest.mock import patch

from schemas import Conference, ConferenceSearchInput


class VerificationBudgetTest(unittest.TestCase):
    def test_oversized_pages_are_bounded_at_model_call(self):
        with patch.dict(os.environ, {"MODEL": "openrouter:example/model",
                                     "OPENROUTER_API_KEY": "test", "TAVILY_API_KEY": "test"}):
            import conference_agent as agent

        event = Conference(name="Tropical Ecology Congress", research_focus="Ecology",
                           city="Aarhus", country="Denmark", start_date="2027-04-01",
                           end_date="2027-04-03", registration_status="unknown",
                           official_website="https://example.org", source_urls=[])
        request = ConferenceSearchInput(field="Ecology", start_date=event.start_date,
                                        end_date=event.end_date)
        evidence = "Tropical Ecology Congress convenes in Aarhus in April 2027."
        page = ('界"\\\n' * 100_000) + evidence
        sources = [dict(source_id=i, url=f"https://example.org/{i}", retrieval_method="full_page",
                        content=evidence if i == 1 else page) for i in range(1, 7)]
        with patch.object(agent, "collect_sources", return_value=sources), \
                patch.object(agent.verification_agent, "invoke") as invoke:
            invoke.return_value = {"structured_response": agent.VerificationResult(
                assessments=[], explanation="Fixture")}
            agent.verify_conference(event, request, {})
        prompt = invoke.call_args.args[0]["messages"][0]["content"]
        self.assertLessEqual(len(prompt.encode("utf-8")), agent.MAX_VERIFICATION_BYTES)
        entries = json.loads(prompt)["sources"]
        self.assertEqual([entry["source_id"] for entry in entries], list(range(1, 7)))
        self.assertEqual(entries[0]["content"], evidence)
        for entry in entries[1:]:
            self.assertIn(evidence, entry["content"])
