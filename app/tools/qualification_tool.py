from __future__ import annotations

from google.genai import types

from app.config import Settings, get_settings
from app.services.qualification_service import evaluate_lead_fit, sanitize_tool_result

EVALUATE_LEAD_FIT_NAME = "evaluate_lead_fit"

EVALUATE_LEAD_FIT_DECLARATION = types.FunctionDeclaration(
    name=EVALUATE_LEAD_FIT_NAME,
    description=(
        "Evaluate whether a prospect's project is a technical and commercial fit for Xirvo. "
        "Call this when the prospect has provided enough information about the project type, "
        "technologies, or pricing expectations. Pass null for anything the prospect did not state. "
        "Do not invent missing values."
    ),
    parameters_json_schema={
        "type": "object",
        "properties": {
            "software_service_needed": {
                "type": ["boolean", "null"],
                "description": "True if they want software, AI, or automation work. False if not. Null if unknown.",
            },
            "project_type": {
                "type": ["string", "null"],
                "description": (
                    "Normalized project type actually described by the prospect, such as "
                    "saas_web_application, ai_chatbot, crm, web_scraping, native_android, "
                    "native_ios, desktop_application, or non_software. Null if unknown."
                ),
            },
            "technologies": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Technologies the prospect named. Empty if none were stated.",
            },
            "hourly_budget": {
                "type": ["number", "null"],
                "description": "Hourly USD rate the prospect said they can pay. Null if not stated.",
            },
            "project_budget": {
                "type": ["number", "null"],
                "description": "Total project budget in USD if stated. Null if not stated.",
            },
            "pricing_negotiation_requested": {
                "type": ["boolean", "null"],
                "description": "True if they asked for a lower rate or commercial flexibility.",
            },
        },
    },
)

EVALUATE_LEAD_FIT_TOOL = types.Tool(function_declarations=[EVALUATE_LEAD_FIT_DECLARATION])


def run_evaluate_lead_fit(
    arguments: dict,
    current_technical_fit: str = "pending",
    current_commercial_fit: str = "pending",
    settings: Settings | None = None,
) -> dict:
    cfg = settings or get_settings()
    result = evaluate_lead_fit(
        software_service_needed=arguments.get("software_service_needed"),
        project_type=arguments.get("project_type"),
        technologies=arguments.get("technologies") or [],
        hourly_budget=arguments.get("hourly_budget"),
        project_budget=arguments.get("project_budget"),
        pricing_negotiation_requested=bool(arguments.get("pricing_negotiation_requested")),
        current_technical_fit=current_technical_fit,
        current_commercial_fit=current_commercial_fit,
        settings=cfg,
    )
    return sanitize_tool_result(result.to_public_dict(), cfg)
