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


class HostedSearchReviewTest(unittest.TestCase):
    """Hosted (no Tavily) discovery must go through the same review as Tavily."""

    def setUp(self):
        with patch.dict(os.environ, {"MODEL": "openrouter:example/model",
                                     "OPENROUTER_API_KEY": "test", "TAVILY_API_KEY": "test"}):
            import conference_agent
        self.agent = conference_agent

    def event(self, name, start, end, city="Bochum", country="Germany"):
        return Conference(name=name, research_focus="MLIPs", city=city, country=country,
                          start_date=start, end_date=end, registration_status="unknown",
                          official_website="https://example.org", source_urls=[])

    def test_filters_and_verification_apply_without_tavily(self):
        agent = self.agent
        events = [
            self.event("Clashing School", "2027-05-03", "2027-05-05"),
            self.event("MLIP Webinar", "2027-01-21", "2027-01-21", "Online", "Online"),
            self.event("Off-topic Meeting", "2027-04-21", "2027-04-23"),
        ]
        request = ConferenceSearchInput(field="ML interatomic potentials", start_date="2027-01-01",
                                        end_date="2027-06-30", excluded_dates=["2027-05-04"],
                                        in_person_only=True)

        def verify(conference, *_):
            return agent.ReviewedCandidate(
                conference=agent.ReviewedConference.from_discovery(conference),
                status="unverified", reason="Scientific relevance not supported.")

        with patch.object(agent, "search_tool", None), \
                patch.object(agent, "discover_candidates_hosted", return_value=(events, None)), \
                patch.object(agent, "verify_conference", side_effect=verify) as verify_mock:
            output = agent.find_conferences(request)

        reasons = {c.conference.name: c.reason for c in output.rejected_candidates}
        self.assertIn("2027-05-04", reasons["Clashing School"])
        self.assertIn("Online-only", reasons["MLIP Webinar"])
        # Only the remaining candidate reaches page verification; nothing is
        # waved through as provisional just because search was hosted.
        verify_mock.assert_called_once()
        self.assertEqual(output.provisional_candidates, [])
        self.assertEqual([c.conference.name for c in output.unverified_candidates],
                         ["Off-topic Meeting"])

    def test_html_text_extraction(self):
        parser = self.agent._TextExtractor()
        parser.feed("<html><head><title>x</title><script>var a=1;</script></head>"
                    "<body><h1>MLPfits 2027</h1><p>15&ndash;19 February 2027,  Bochum</p></body></html>")
        self.assertEqual(parser.text(), "MLPfits 2027\n15–19 February 2027, Bochum")
