import os
import unittest
from unittest.mock import patch

from travel_agent import Itinerary, TravelRequest, TravelResearch, select_itinerary


class TravelTest(unittest.TestCase):
    def setUp(self):
        self.request = TravelRequest(origin="Singapore", destination="Bangkok",
                                     event_start="2027-03-16", event_end="2027-03-18",
                                     currency="SGD", transport_lodging_budget=900)

    def option(self, fare, minutes):
        def leg(day):
            return dict(route_and_service="Fixture service", arrival=day, departure=day,
                        duration_minutes=minutes, fare=fare, sources=["https://example.org/fare"])
        return Itinerary(outbound=leg("2027-03-15"), inbound=leg("2027-03-19"),
                         accommodation="Fixture hotel", check_in="2027-03-15",
                         check_out="2027-03-19", lodging_total=300, currency="SGD",
                         accommodation_sources=["https://example.org/hotel"], notes="Fixture")

    def test_ranking(self):
        research = TravelResearch(options=[self.option(400, 60), self.option(200, 90),
                                           self.option(150, 120)], search_notes="")
        result = select_itinerary(self.request, research)
        self.assertEqual(result.estimated_transport_lodging_total, 600)
        self.request.priority = "shortest_travel"
        self.assertEqual(select_itinerary(self.request, research).itinerary.duration, 180)

    def test_orchestrator_tool(self):
        with patch.dict(os.environ, {"MODEL": "openrouter:example/model",
                                     "OPENROUTER_API_KEY": "test", "TAVILY_API_KEY": "test"}):
            from orchestrator import find_itinerary
        response = TravelResearch(options=[self.option(None, 90)], search_notes="Fare unpublished")
        with patch("travel_agent._travel_agent") as agent:
            agent.return_value.invoke.return_value = {"structured_response": response}
            from travel_agent import TravelResult
            result = TravelResult.model_validate_json(find_itinerary.invoke(
                {"request": self.request.model_dump(mode="json")}))
        self.assertEqual(result.status, "needs_verification")
        self.assertIsNone(result.estimated_transport_lodging_total)

    def test_excluded_day(self):
        research = TravelResearch(options=[self.option(150, 120)], search_notes="")
        self.request.excluded_dates = [self.request.event_end]
        self.assertEqual(select_itinerary(self.request, research).status, "no_match")

    def test_rejection_reasons_reported(self):
        # The fixture trip runs 2027-03-15..19 = 5 days.
        research = TravelResearch(options=[self.option(150, 120)], search_notes="Found one rail option")
        self.request.max_trip_days = 4
        result = select_itinerary(self.request, research)
        self.assertEqual(result.status, "no_match")
        self.assertEqual(len(result.rejected_options), 1)
        self.assertEqual(result.rejected_options[0].reasons,
                         ["trip lasts 5 days, over the 4-day maximum"])
        self.assertEqual(result.rejected_options[0].known_cost, 600)
        self.assertIn("broke the traveler's constraints", result.notes)

    def test_eligible_option_has_no_rejections(self):
        research = TravelResearch(options=[self.option(150, 120)], search_notes="")
        result = select_itinerary(self.request, research)
        self.assertEqual(result.status, "recommended")
        self.assertEqual(result.rejected_options, [])
