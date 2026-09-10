from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


NextAction = Literal[
    "answer_question",
    "understand_problem",
    "understand_requirements",
    "clarify_project_type",
    "clarify_technology",
    "build_value",
    "show_relevant_portfolio",
    "technical_qualification",
    "commercial_discussion",
    "handle_objection",
    "test_buying_intent",
    "close",
    "human_handoff",
    "politely_disqualify",
    "continue_discovery",
]


class SalesObservations(BaseModel):
    model_config = ConfigDict(extra="ignore")

    software_service_needed: bool | None = None
    project_type: str | None = None
    technologies: list[str] = Field(default_factory=list)
    hourly_budget: float | None = None
    project_budget: float | None = None
    name_provided: bool = False
    email_provided: bool = False
    price_objection: bool = False
    pricing_negotiation_requested: bool = False
    customer_asked_about_pricing: bool = False
    asks_for_human: bool = False
    contract_or_nda_question: bool = False
    ready_to_proceed: bool = False

    @field_validator("technologies", mode="before")
    @classmethod
    def _coerce_technologies(cls, value: object) -> object:
        if not value:
            return []
        if isinstance(value, str):
            return [value]
        return value


class SalesTurnOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    assistant_response: str
    next_action: str = "continue_discovery"
    observations: SalesObservations = Field(default_factory=SalesObservations)
    qualification_tool_needed: bool = False
    handoff_required: bool = False

    @field_validator("next_action", mode="before")
    @classmethod
    def _coerce_next_action(cls, value: object) -> str:
        allowed = set(NextAction.__args__)
        if isinstance(value, str) and value in allowed:
            return value
        return "continue_discovery"

    @field_validator("observations", mode="before")
    @classmethod
    def _coerce_observations(cls, value: object) -> object:
        return value or {}

    def tool_args(self) -> dict:
        observations = self.observations
        return {
            "software_service_needed": observations.software_service_needed,
            "project_type": observations.project_type,
            "technologies": list(observations.technologies or []),
            "hourly_budget": observations.hourly_budget,
            "project_budget": observations.project_budget,
            "pricing_negotiation_requested": observations.pricing_negotiation_requested,
        }
