import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from trip_report import ConferenceRow, Note, TripReport, render_report


class ReportTest(unittest.TestCase):
    def test_escaping_and_shared_notes(self):
        source = Note(text="Official & current", url="https://example.org/fees")
        row = ConferenceRow(conference="<event>", location="Tokyo", dates="June",
                            registration="Unknown", trip_total="Unknown", budget_status="Unverified",
                            notes=[source, Note(text="Offline", url="javascript:alert(1)")])
        html = render_report(TripReport(summary="No confirmed cost.", conferences=[row, row],
                                       travel=[], entry=[]))
        self.assertIn("&lt;event&gt;", html)
        self.assertNotIn("javascript:", html)
        self.assertEqual(html.count('id="note-1"'), 1)
        self.assertEqual(html.count('href="#note-1"'), 2)

    def test_cli_config_overrides_example(self):
        from orchestrator import main
        import json
        settings = dict(field="Machine learning", origin_city="Singapore", origin_country="SG",
                        residence_country="SG", passport_country="SG", destination_countries=["JP"],
                        window_start="2027-01-01", window_end="2027-06-30", max_trip_days=7,
                        unavailable_dates=["2027-05-04"], budget="2200.00", currency="SGD")
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / "request.json"
            out = Path(folder) / "reports/trip.html"
            config.write_text(json.dumps(settings))
            with patch("orchestrator.orchestrator.invoke") as invoke, redirect_stdout(io.StringIO()):
                invoke.return_value = {"structured_response": TripReport(summary="No matches",
                                          conferences=[], travel=[], entry=[])}
                main(["--config", str(config), "--output", str(out)])
            prompt = invoke.call_args.args[0]["messages"][0]["content"]
            self.assertNotIn("ecology", prompt)
            self.assertIn('"unavailable_dates":["2027-05-04"]', prompt)
            self.assertIn("No matches", out.read_text())
