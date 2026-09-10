from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings, get_settings
from app.db.models import CommercialFit, TechnicalFit

_QUALIFIED_PROJECT_TYPES = {
    "ai chatbot",
    "ai agent",
    "ai voice agent",
    "voice agent",
    "rag",
    "saas",
    "saas web application",
    "web application",
    "web app",
    "crm",
    "workflow automation",
    "automation",
    "ai automation",
    "backend",
    "backend system",
    "frontend",
    "frontend web system",
    "web scraping",
    "browser automation",
    "api integration",
    "api integrations",
    "software automation",
}

_ANDROID_TYPES = {
    "native android",
    "android",
    "android app",
    "android application",
}

_IOS_TYPES = {
    "native ios",
    "ios",
    "ios app",
    "ios application",
    "iphone app",
}

_DESKTOP_TYPES = {
    "desktop",
    "desktop application",
    "desktop app",
    "desktop only",
}

_NON_SOFTWARE_TYPES = {
    "non software",
    "not software",
}

_ALLOWED_TECH = {
    "python",
    "django",
    "flask",
    "fastapi",
    "pandas",
    "numpy",
    "pytorch",
    "tensorflow",
    "langchain",
    "langgraph",
    "celery",
    "sqlalchemy",
    "javascript",
    "js",
    "node",
    "node.js",
    "nodejs",
    "express",
    "vue",
    "vue.js",
    "angular",
    "svelte",
    "next",
    "next.js",
    "nuxt",
    "react",
    "react.js",
    "react native",
    "jquery",
    "nest",
    "nestjs",
    "remix",
    "typescript",
    "ts",
    "tsx",
    "html",
    "css",
    "n8n",
    "hubspot",
}

_DISALLOWED_TECH = {
    "kotlin",
    "swift",
    "swiftui",
    "objc",
    "objective c",
    "c#",
    "csharp",
    "go",
    "golang",
    "ruby",
    "php",
    "rust",
    "c++",
    "cpp",
    "scala",
    "perl",
    "matlab",
    "dart",
    "flutter",
    "java",
    "wpf",
    "winforms",
    "qt",
    "gtk",
    "electron",
}

_ANDROID_TECH = {"android", "kotlin"}
_IOS_TECH = {"ios", "swift", "swiftui", "objc", "objective c"}
_DESKTOP_TECH = {"desktop", "wpf", "winforms", "electron", "qt", "gtk"}

_PUBLIC_TOOL_KEYS = (
    "technical_fit",
    "commercial_fit",
    "public_pricing_guidance",
    "human_handoff_recommended",
)


@dataclass(frozen=True)
class LeadFitResult:
    technical_fit: str
    commercial_fit: str
    public_pricing_guidance: str
    human_handoff_recommended: bool

    def to_public_dict(self) -> dict:
        return {
            "technical_fit": self.technical_fit,
            "commercial_fit": self.commercial_fit,
            "public_pricing_guidance": self.public_pricing_guidance,
            "human_handoff_recommended": self.human_handoff_recommended,
        }


def _norm(value: str | None) -> str:
    if not value:
        return ""
    cleaned = value.lower().replace("_", " ").replace("-", " ").replace("/", " ")
    return " ".join(cleaned.split())


def _tech_list(technologies: list[str] | None) -> list[str]:
    return [_norm(item) for item in (technologies or []) if _norm(item)]


def _has_any(values: list[str], catalog: set[str]) -> bool:
    return any(item in catalog for item in values)


def _is_react_native(technologies: list[str]) -> bool:
    joined = " ".join(technologies)
    return "react native" in joined or (
        "react" in technologies and "native" in technologies and "android" not in technologies
    )


def evaluate_technical_fit(
    software_service_needed: bool | None,
    project_type: str | None = None,
    technologies: list[str] | None = None,
) -> str:
    project = _norm(project_type)
    techs = _tech_list(technologies)

    if software_service_needed is False or project in _NON_SOFTWARE_TYPES:
        return TechnicalFit.unqualified.value

    if project in _ANDROID_TYPES or project in _IOS_TYPES or project in _DESKTOP_TYPES:
        return TechnicalFit.unqualified.value

    if not _is_react_native(techs):
        if _has_any(techs, _ANDROID_TECH) or "native android" in " ".join(techs):
            return TechnicalFit.unqualified.value
        if _has_any(techs, _IOS_TECH) or "native ios" in " ".join(techs):
            return TechnicalFit.unqualified.value
        if _has_any(techs, _DESKTOP_TECH) or "desktop" in project:
            return TechnicalFit.unqualified.value

    if techs and not _has_any(techs, _ALLOWED_TECH) and _has_any(techs, _DISALLOWED_TECH):
        return TechnicalFit.unqualified.value

    if project in _QUALIFIED_PROJECT_TYPES:
        return TechnicalFit.qualified.value

    if _has_any(techs, _ALLOWED_TECH):
        return TechnicalFit.qualified.value

    return TechnicalFit.pending.value


def evaluate_commercial_fit(
    technical_fit: str,
    hourly_budget: float | None = None,
    project_budget: float | None = None,
    pricing_negotiation_requested: bool | None = False,
    settings: Settings | None = None,
) -> str:
    cfg = settings or get_settings()
    if technical_fit != TechnicalFit.qualified.value:
        return CommercialFit.pending.value

    if project_budget is not None and project_budget < cfg.min_project_amount:
        return CommercialFit.unqualified.value

    if hourly_budget is not None:
        if hourly_budget >= cfg.standard_hourly_rate:
            return CommercialFit.qualified.value
        if hourly_budget >= cfg.internal_min_hourly_rate:
            return CommercialFit.negotiation_required.value
        return CommercialFit.unqualified.value

    if pricing_negotiation_requested:
        return CommercialFit.negotiation_required.value

    return CommercialFit.pending.value


def _pricing_was_raised(
    hourly_budget: float | None,
    project_budget: float | None,
    pricing_negotiation_requested: bool | None,
) -> bool:
    return (
        hourly_budget is not None
        or project_budget is not None
        or bool(pricing_negotiation_requested)
    )


def _do_not_mention_pricing_guidance() -> str:
    return (
        "The customer has not asked about price, cost, rate, budget, payment, billing, "
        "or commercial terms. Do not mention hourly rates, minimum engagement amounts, "
        "billing cadence, deposits, discounts, payment terms, or commercial flexibility "
        "in assistant_response. Technical qualification does not authorize a pricing "
        "discussion. Continue with the problem, proposed solution, value, implementation "
        "approach, or relevant Xirvo experience."
    )


def _public_pricing_guidance(
    technical_fit: str,
    commercial_fit: str,
    settings: Settings,
    pricing_raised: bool,
) -> str:
    if not pricing_raised:
        return _do_not_mention_pricing_guidance()
    standard = int(settings.standard_hourly_rate)
    minimum_project = int(settings.min_project_amount)
    if technical_fit != TechnicalFit.qualified.value:
        return (
            f"Xirvo's standard development rate is ${standard} per hour. "
            "Do not negotiate pricing yet. First understand the project and determine "
            "whether it is a technical fit."
        )
    if commercial_fit == CommercialFit.negotiation_required.value:
        return (
            f"Maintain the standard ${standard}/hour rate and explain that "
            "project-specific pricing can be discussed with the Xirvo team."
        )
    if commercial_fit == CommercialFit.unqualified.value:
        return (
            f"Maintain the standard ${standard}/hour rate and the "
            f"${minimum_project} minimum engagement. Do not offer a discount, invent a "
            "quote, or mention any other hourly figure."
        )
    if commercial_fit == CommercialFit.qualified.value:
        return (
            f"The prospect is aligned with Xirvo's standard ${standard}/hour rate "
            f"and ${minimum_project} minimum engagement. Do not invent a fixed quote "
            "or hour estimate."
        )
    return (
        f"Xirvo's standard development rate is ${standard} per hour, with a "
        f"minimum engagement of ${minimum_project}."
    )


def _confidential_tokens(settings: Settings) -> tuple[str, ...]:
    rate = settings.internal_min_hourly_rate
    as_int = int(rate)
    return (str(rate), str(as_int), f"${as_int}", f"${rate}")


def sanitize_tool_result(payload: dict, settings: Settings | None = None) -> dict:
    cfg = settings or get_settings()
    cleaned = {key: payload.get(key) for key in _PUBLIC_TOOL_KEYS}
    blob = str(cleaned)
    if any(token in blob for token in _confidential_tokens(cfg)):
        cleaned["public_pricing_guidance"] = (
            "Do not mention any non-public rate. If the customer asked about price, "
            "state only Xirvo's public standard development rate from the system prompt. "
            "If they did not ask, do not mention pricing at all."
        )
    return cleaned


def resolve_fit_state(current: str, incoming: str) -> str:
    if not incoming or incoming == TechnicalFit.pending.value:
        return current or TechnicalFit.pending.value
    return incoming


def evaluate_lead_fit(
    software_service_needed: bool | None = None,
    project_type: str | None = None,
    technologies: list[str] | None = None,
    hourly_budget: float | None = None,
    project_budget: float | None = None,
    pricing_negotiation_requested: bool | None = False,
    current_technical_fit: str = TechnicalFit.pending.value,
    current_commercial_fit: str = CommercialFit.pending.value,
    settings: Settings | None = None,
) -> LeadFitResult:
    cfg = settings or get_settings()
    technical = evaluate_technical_fit(
        software_service_needed=software_service_needed,
        project_type=project_type,
        technologies=technologies,
    )
    technical = resolve_fit_state(current_technical_fit, technical)
    commercial = evaluate_commercial_fit(
        technical_fit=technical,
        hourly_budget=hourly_budget,
        project_budget=project_budget,
        pricing_negotiation_requested=bool(pricing_negotiation_requested),
        settings=cfg,
    )
    commercial = resolve_fit_state(current_commercial_fit, commercial)
    return LeadFitResult(
        technical_fit=technical,
        commercial_fit=commercial,
        public_pricing_guidance=_public_pricing_guidance(
            technical,
            commercial,
            cfg,
            _pricing_was_raised(hourly_budget, project_budget, pricing_negotiation_requested),
        ),
        human_handoff_recommended=(
            technical == TechnicalFit.qualified.value
            and commercial == CommercialFit.negotiation_required.value
        ),
    )
