import os
import json
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

def generate_visa_white_paper(nationality: str, destination_country: str) -> ConferenceVisaWhitePaper:
    prompt = (
        f"Generate a detailed Conference Visa White Paper for a conference attendee.\n"
        f"- Applicant Passport / Nationality: {nationality}\n"
        f"- Conference Destination Country: {destination_country}\n"
        f"Include official visa types, document requirements, submission steps, processing times, timeline, and a visa-specific budget breakdown."
    )
    
    print(f"Running Visa Agent for {nationality} national travelling to {destination_country}...\n")
    result = call_with_quota(visa_agent.run_sync, prompt)
    return result.output

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
