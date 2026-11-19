import os
import unittest
from unittest.mock import patch

with patch.dict(os.environ, {"MODEL": "openrouter:example/model", "OPENROUTER_API_KEY": "test"}):
    import VisaAgent


class EntryRuleShortcutTest(unittest.TestCase):
    def test_resident_of_destination_needs_nothing(self):
        paper = VisaAgent.entry_rule_shortcut("Czechia", "United Kingdom", "UK")
        self.assertFalse(paper.visa_details.visa_required)
        self.assertIn("domestic", paper.executive_summary)
        self.assertEqual(paper.financial_budget.total_visa_cost_usd, 0)

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

    def test_shortcut_skips_model_call(self):
        with patch.object(VisaAgent, "call_with_quota") as call:
            VisaAgent.generate_visa_white_paper("Czechia", "France", "United Kingdom")
        call.assert_not_called()


if __name__ == "__main__":
    unittest.main()
