import asyncio
import os
import json
import threading
from typing import List
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from model_quota import call_with_quota
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 0. Load Environment Variables from .env
# ---------------------------------------------------------------------------
load_dotenv()

# Read the model identifier from the .env file (e.g., openrouter:openai/gpt-4.1-mini)
model_name = os.getenv("MODEL")
if not model_name:
    raise ValueError("MODEL is not set in the .env file!")

# ---------------------------------------------------------------------------
# 1. Structured Data Models (Output Schema)
# ---------------------------------------------------------------------------

class VisaRequirementDetails(BaseModel):
    visa_required: bool = Field(description="Whether a visa is required for this nationality to enter the destination country.")
    visa_type: str = Field(description="The specific visa category required (e.g., Short-stay Schengen Business Visa (Type C), B-1 Visitor Visa, e-Visa, Visa on Arrival).")
    allowed_stay_duration: str = Field(description="Maximum length of stay permitted under this visa type (e.g., up to 90 days within 180 days).")

class TimelineStep(BaseModel):
    timeframe: str = Field(description="Recommended timeframe relative to conference date (e.g., 6-8 weeks prior, 2 weeks prior).")
    action: str = Field(description="Specific task to complete in this timeframe.")

class BudgetBreakdown(BaseModel):
    visa_application_fee_usd: float = Field(description="Estimated official government visa application fee in USD.")
    biometrics_or_service_fee_usd: float = Field(default=0.0, description="VFS/TLS/TLScontact appointment or service center fee in USD, if applicable.")
    mandatory_visa_insurance_fee_usd: float = Field(default=0.0, description="Mandatory travel health insurance fee required specifically for the visa application process, if applicable.")
    total_visa_cost_usd: float = Field(description="Total estimated cost strictly associated with obtaining the visa.")
    budget_notes: str = Field(description="Explanation of visa fees, currency conversions, or extra administrative costs.")

class ConferenceVisaWhitePaper(BaseModel):
    executive_summary: str = Field(description="High-level overview of the travel compliance and visa procedure for the given applicant.")
    applicant_nationality: str = Field(description="Country of citizenship/passport of the attendee.")
    destination_country: str = Field(description="Host country of the conference.")
    visa_details: VisaRequirementDetails = Field(description="Core visa rules and classification.")
    required_documents: List[str] = Field(description="Itemized checklist of essential documents required for application submission.")
    step_by_step_procedure: List[str] = Field(description="Sequential guide detailing how to apply (online portal, embassy appointment, biometrics, etc.).")
    processing_time: str = Field(description="Normal processing duration (e.g., 15-30 calendar days).")
    recommended_timeline: List[TimelineStep] = Field(description="Timeline roadmap mapping out actions from registration to departure.")
    financial_budget: BudgetBreakdown = Field(description="Itemized cost analysis focused strictly on visa expenses.")
    disclaimer: str = Field(description="Standard legal disclaimer regarding embassy policy updates and official verification.")


# ---------------------------------------------------------------------------
# 2. Initialize PydanticAI Agent with OpenRouter Model
# ---------------------------------------------------------------------------

visa_agent = Agent(
    model=model_name,
    output_type=ConferenceVisaWhitePaper,
    defer_model_check=True,
    system_prompt=(
        "You are an expert global immigration and travel compliance consultant specializing in academic and corporate conference travel. "
        "Your task is to generate a comprehensive, highly accurate White Paper for an individual attending an international conference. "
        "You will evaluate the user's nationality and destination country, then detail the visa category, documentation checklist, "
        "step-by-step application procedure, normal processing times, milestone timeline, and a budget breakdown "
        "that focuses strictly on visa-related expenses (government fees, service/biometric fees, and mandatory visa insurance)."
    )
)

# ---------------------------------------------------------------------------
# 3. Main Execution Function
# ---------------------------------------------------------------------------

# Free movement area: EU member states plus the other EEA states and Switzerland.
_FREE_MOVEMENT = {
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IE",
    "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK", "SI", "ES", "SE",
    "IS", "LI", "NO", "CH",
}
_COUNTRY_NAMES = {
    "austria": "AT", "belgium": "BE", "bulgaria": "BG", "croatia": "HR", "cyprus": "CY",
    "czechia": "CZ", "czech republic": "CZ", "czech": "CZ", "denmark": "DK", "estonia": "EE",
    "finland": "FI", "france": "FR", "germany": "DE", "greece": "GR", "hungary": "HU",
    "ireland": "IE", "italy": "IT", "latvia": "LV", "lithuania": "LT", "luxembourg": "LU",
    "malta": "MT", "netherlands": "NL", "the netherlands": "NL", "poland": "PL",
    "portugal": "PT", "romania": "RO", "slovakia": "SK", "slovenia": "SI", "spain": "ES",
    "sweden": "SE", "iceland": "IS", "liechtenstein": "LI", "norway": "NO",
    "switzerland": "CH", "united kingdom": "GB", "uk": "GB", "great britain": "GB",
    "britain": "GB", "england": "GB", "scotland": "GB", "wales": "GB",
    "northern ireland": "GB", "united states": "US", "united states of america": "US",
    "usa": "US", "us": "US", "india": "IN", "singapore": "SG",
}
# Demonyms the orchestrator may pass as "nationality".
_DEMONYMS = {"czech": "CZ", "german": "DE", "french": "FR", "british": "GB",
             "american": "US", "indian": "IN", "singaporean": "SG", "slovak": "SK",
             "polish": "PL", "italian": "IT", "spanish": "ES", "dutch": "NL"}


def country_code(name: str | None) -> str | None:
    """ISO-3166 alpha-2 code for a country name, demonym or code; None if unknown."""
    if not name:
        return None
    key = " ".join(name.strip().casefold().replace(".", "").split())
    if len(key) == 2 and key.isalpha():
        return "GB" if key == "uk" else key.upper()
    return _COUNTRY_NAMES.get(key) or _DEMONYMS.get(key)


def _no_visa_needed(nationality: str, destination: str, summary: str, stay: str,
                    documents: list[str]) -> ConferenceVisaWhitePaper:
    return ConferenceVisaWhitePaper(
        executive_summary=summary, applicant_nationality=nationality,
        destination_country=destination,
        visa_details=VisaRequirementDetails(visa_required=False, visa_type="None required",
                                            allowed_stay_duration=stay),
        required_documents=documents, step_by_step_procedure=[],
        processing_time="Not applicable", recommended_timeline=[],
        financial_budget=BudgetBreakdown(
            visa_application_fee_usd=0, total_visa_cost_usd=0,
            budget_notes="No visa or travel authorisation fee applies."),
        disclaimer=("Determined from citizenship and residence rules, not a live lookup. "
                    "Check passport/ID validity before travel."),
    )


def entry_rule_shortcut(nationality: str, destination: str,
                        residence: str | None = None) -> ConferenceVisaWhitePaper | None:
    """Cases settled by citizenship or residence alone, without asking the model.

    The model otherwise applies third-country rules to EU citizens (90-day limit,
    Schengen insurance) and visitor-visa rules to residents of the destination.
    """
    nat, dest, res = country_code(nationality), country_code(destination), country_code(residence)
    if not dest:
        return None
    if nat == dest:
        return _no_visa_needed(
            nationality, destination,
            f"{destination} is the traveler's country of citizenship; no visa or entry authorisation is needed.",
            "Unlimited (citizen)", ["Valid passport or national identity card"])
    if res == dest:
        return _no_visa_needed(
            nationality, destination,
            f"The traveler lives in {destination}, so this is domestic travel and no visa "
            f"or entry authorisation is needed. Keep proof of residence status (for example "
            f"a residence permit or settled/pre-settled status) for re-entry from abroad.",
            "Not applicable (resident)",
            ["Valid passport or national identity card", "Proof of residence status"])
    if nat in _FREE_MOVEMENT and dest in _FREE_MOVEMENT:
        return _no_visa_needed(
            nationality, destination,
            f"As a citizen of an EU/EEA country or Switzerland, the traveler has free movement "
            f"rights in {destination}: no visa, no travel authorisation, and the Schengen "
            f"90/180-day short-stay rule and visa insurance requirement do not apply.",
            "No limit for a conference visit (free movement; registration may be required after 3 months)",
            ["Valid passport or national identity card"])
    return None


def generate_visa_white_paper(nationality: str, destination_country: str,
                              residence_country: str | None = None) -> ConferenceVisaWhitePaper:
    shortcut = entry_rule_shortcut(nationality, destination_country, residence_country)
    if shortcut is not None:
        print(f"Visa rule shortcut for {nationality} national (resident in "
              f"{residence_country or 'unknown'}) travelling to {destination_country}.\n")
        return shortcut

    prompt = (
        f"Generate a detailed Conference Visa White Paper for a conference attendee.\n"
        f"- Applicant Passport / Nationality: {nationality}\n"
        f"- Country of residence: {residence_country or 'not stated'}\n"
        f"- Conference Destination Country: {destination_country}\n"
        f"Include official visa types, document requirements, submission steps, processing times, timeline, and a visa-specific budget breakdown.\n"
        f"Consider whether the residence permit changes the requirement. If no visa is needed "
        f"but an electronic travel authorisation is (for example UK ETA, US ESTA, Canada eTA, "
        f"EU ETIAS once in operation), report it as the requirement with its fee and validity. "
        f"State that rules change and must be confirmed on the official government site."
    )

    print(f"Running Visa Agent for {nationality} national travelling to {destination_country}...\n")
    result = call_with_quota(_run_on_shared_loop, prompt)
    return result.output


# The orchestrator calls this tool from worker threads. run_sync would give each
# call its own event loop, but pydantic-ai reuses pooled HTTP connections that
# stay bound to the loop that opened them, so later calls fail. Run every call on
# one long-lived loop instead.
_loop = asyncio.new_event_loop()
threading.Thread(target=_loop.run_forever, name="visa-agent-loop", daemon=True).start()


def _run_on_shared_loop(prompt: str):
    return asyncio.run_coroutine_threadsafe(visa_agent.run(prompt), _loop).result()

if __name__ == "__main__":
    user_nationality = "Indian"
    conference_country = "Germany"
    
    white_paper = generate_visa_white_paper(
        nationality=user_nationality, 
        destination_country=conference_country
    )

    print("=========================================================")
    print(f" VISA WHITE PAPER: {user_nationality} -> {conference_country}")
    print("=========================================================\n")
    print(json.dumps(white_paper.model_dump(), indent=2))
