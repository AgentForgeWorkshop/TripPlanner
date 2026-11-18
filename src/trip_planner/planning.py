from collections import defaultdict
from datetime import date, timedelta
from decimal import Decimal

from .models import (
    Budget, Conference, Cost, CostLine, Entry, ItineraryDay, Travel, TripRequest,
)


def calendar_conflict(request: TripRequest, start: date, end: date, today: date) -> str | None:
    if start > end or start < max(request.window_start, today) or end > request.window_end:
        return "Dates fall outside the available travel window."
    if any(start <= day <= end for day in request.unavailable_dates):
        return "Dates overlap an unavailable day."
    if request.max_trip_days is not None and (end - start).days + 1 > request.max_trip_days:
        return "Dates exceed the maximum trip length."
    return None


def trip_conflict(request: TripRequest, event: Conference, travel: Travel, today: date) -> str | None:
    route = travel.transport
    if route is None:
        return "The travel specialist could not substantiate a round trip."
    if set(route.transit_countries) - {request.origin_country, event.country}:
        return "The route connects through a country outside the entry checks."
    if not (route.depart_origin <= route.arrive_destination <= event.start <= event.end
            <= route.depart_destination <= route.arrive_origin):
        return "Travel dates do not cover the conference."
    if route.mode == "local":
        if (request.origin_country != event.country
                or request.origin_city.casefold() != event.city.casefold()):
            return "The local transport option does not connect the origin to the conference city."
    elif not (route.arrive_destination < event.start and event.end < route.depart_destination):
        return "Travel dates leave no full conference days between arrival and departure."
    if travel.lodging is not None:
        if (travel.lodging.check_in != route.arrive_destination
                or travel.lodging.check_out != route.depart_destination
                or travel.lodging.check_in >= travel.lodging.check_out):
            return "Accommodation dates do not cover the stay."
    elif route.mode != "local" and route.arrive_destination < route.depart_destination:
        return "The travel specialist could not substantiate accommodation for the full stay."
    return calendar_conflict(request, route.depart_origin, route.arrive_origin, today)


def entry_issues(request: TripRequest, event: Conference, entry: Entry | None,
                 leave_on: date, today: date) -> list[str]:
    domestic = request.origin_country == event.country
    if domestic and (entry is None or entry.status in {"domestic", "unknown"}):
        return []
    if (entry is None or not entry.source_urls or entry.status in {"unknown", "domestic"}
            or (entry.status == "existing_permission" and not request.existing_entry_permission)):
        return ["Applicable entry requirements could not be established from the available sources."]
    if entry.status not in {"authorization_required", "visa_required"}:
        return []
    issues = ["Obtain the required visa or travel authorization before departure."]
    if entry.processing_days_upper is None or not entry.processing_time or not entry.processing_source_urls:
        issues.append("Current end-to-end processing time could not be verified.")
    elif entry.processing_days_upper > (leave_on - today).days:
        issues.append("The published processing-time upper bound exceeds the time before departure.")
    return issues


def make_budget(request: TripRequest, event: Conference, travel: Travel, entry: Entry | None) -> Budget:
    def line(category: str, cost: Cost | None) -> CostLine:
        if cost is None:
            return CostLine(category=category, amount=None, basis="No supported estimate available.")
        if cost.currency != request.currency:
            return CostLine(category=category, amount=None, source_urls=cost.source_urls,
                            basis=f"The estimate is in {cost.currency} and has no sourced conversion.")
        return CostLine(category=category, amount=cost.amount, basis=cost.basis, source_urls=cost.source_urls)

    lodging = line("Accommodation", travel.lodging.cost) if travel.lodging else CostLine(
        category="Accommodation", amount=Decimal(0), basis="No accommodation in the itinerary."
    )
    entry_fees = (CostLine(category="Entry fees", amount=Decimal(0), basis="Domestic itinerary.")
                  if event.country == request.origin_country else line("Entry fees", entry.cost if entry else None))
    items = [
        line("Conference registration", event.registration),
        line("Round-trip transportation", travel.transport.cost if travel.transport else None),
        lodging,
        line("Local transport", travel.local_transport),
        line("Meals", travel.meals),
        entry_fees,
    ]
    subtotal = sum((item.amount for item in items if item.amount is not None), Decimal(0))
    total = subtotal if all(item.amount is not None for item in items) else None
    return Budget(currency=request.currency, limit=request.budget, items=items, known_subtotal=subtotal,
                  estimated_total=total, remaining=request.budget - total if total is not None else None)


def make_itinerary(event: Conference, travel: Travel) -> list[ItineraryDay]:
    route = travel.transport
    assert route is not None  # The orchestrator checks the route before building an itinerary.
    activities: dict[date, list[str]] = defaultdict(list)
    activities[route.depart_origin].append(f"Outbound: {route.outbound}")
    if route.arrive_destination != route.depart_origin:
        activities[route.arrive_destination].append(f"Arrive in {event.city}.")
    if travel.lodging:
        activities[travel.lodging.check_in].append(f"Check in at {travel.lodging.name}.")
    location = f" at {event.venue}" if event.venue else ""
    for offset in range((event.end - event.start).days + 1):
        day = event.start + timedelta(days=offset)
        activities[day].append(f"Attend {event.name}{location}.")
    if travel.lodging:
        activities[travel.lodging.check_out].append(f"Check out of {travel.lodging.name}.")
    activities[route.depart_destination].append(f"Return: {route.inbound}")
    if route.arrive_origin != route.depart_destination:
        activities[route.arrive_origin].append("Arrive back at the origin.")

    itinerary = []
    for offset in range((route.arrive_origin - route.depart_origin).days + 1):
        day = route.depart_origin + timedelta(days=offset)
        itinerary.append(ItineraryDay(date=day, activities=activities[day] or ["No scheduled activity."]))
    return itinerary
