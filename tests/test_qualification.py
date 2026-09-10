from app.services.qualification_service import evaluate_commercial_fit, evaluate_lead_fit, evaluate_technical_fit
from app.tools.qualification_tool import run_evaluate_lead_fit


def test_python_ai_automation_is_technically_qualified() -> None:
    assert evaluate_technical_fit(True, "ai_automation", ["python"]) == "qualified"


def test_react_typescript_saas_is_technically_qualified() -> None:
    assert evaluate_technical_fit(True, "saas_web_application", ["typescript", "react"]) == "qualified"


def test_javascript_web_application_is_technically_qualified() -> None:
    assert evaluate_technical_fit(True, "web_application", ["javascript"]) == "qualified"


def test_crm_and_web_scraping_are_technically_qualified() -> None:
    assert evaluate_technical_fit(True, "crm", []) == "qualified"
    assert evaluate_technical_fit(True, "web_scraping", []) == "qualified"


def test_native_android_is_technically_unqualified() -> None:
    assert evaluate_technical_fit(True, "native_android", []) == "unqualified"


def test_native_ios_is_technically_unqualified() -> None:
    assert evaluate_technical_fit(True, "native_ios", ["swift"]) == "unqualified"


def test_desktop_only_application_is_technically_unqualified() -> None:
    assert evaluate_technical_fit(True, "desktop_application", []) == "unqualified"


def test_unknown_technology_stays_pending() -> None:
    assert evaluate_technical_fit(True, None, None) == "pending"
    assert evaluate_technical_fit(None, None, []) == "pending"


def test_non_software_request_is_technically_unqualified() -> None:
    assert evaluate_technical_fit(False, None, None) == "unqualified"
    assert evaluate_technical_fit(True, "non_software", []) == "unqualified"


def test_price_before_technical_fit_does_not_start_negotiation(settings) -> None:
    result = evaluate_lead_fit(
        software_service_needed=True,
        hourly_budget=30,
        current_technical_fit="pending",
        current_commercial_fit="pending",
        settings=settings,
    )
    assert result.technical_fit == "pending"
    assert result.commercial_fit == "pending"
    assert "40" in result.public_pricing_guidance
    assert "do not negotiate" in result.public_pricing_guidance.lower()
    assert not result.human_handoff_recommended


def test_qualified_technical_fit_accepting_standard_rate(settings) -> None:
    assert (
        evaluate_commercial_fit(
            "qualified",
            hourly_budget=settings.standard_hourly_rate,
            settings=settings,
        )
        == "qualified"
    )


def test_qualified_technical_fit_with_mid_rate_needs_negotiation(settings) -> None:
    assert evaluate_commercial_fit("qualified", hourly_budget=30, settings=settings) == "negotiation_required"


def test_qualified_technical_fit_with_hard_low_rate_is_unqualified(settings) -> None:
    assert evaluate_commercial_fit("qualified", hourly_budget=20, settings=settings) == "unqualified"


def test_tool_result_never_includes_confidential_threshold(settings) -> None:
    payload = run_evaluate_lead_fit(
        {
            "software_service_needed": True,
            "project_type": "saas_web_application",
            "technologies": ["typescript"],
            "hourly_budget": 20,
            "pricing_negotiation_requested": True,
        },
        current_technical_fit="qualified",
        settings=settings,
    )
    serialized = str(payload)
    confidential = str(int(settings.internal_min_hourly_rate))
    assert confidential not in serialized
    assert f"${confidential}" not in serialized
    assert "hidden" not in serialized.lower()
    assert "threshold" not in serialized.lower()
    assert payload["technical_fit"] == "qualified"
    assert payload["commercial_fit"] == "unqualified"
    assert set(payload) == {
        "technical_fit",
        "commercial_fit",
        "public_pricing_guidance",
        "human_handoff_recommended",
    }


def test_negotiation_after_technical_qualification_recommends_handoff(settings) -> None:
    payload = run_evaluate_lead_fit(
        {
            "software_service_needed": True,
            "project_type": "saas_web_application",
            "technologies": ["typescript"],
            "hourly_budget": 30,
            "pricing_negotiation_requested": True,
        },
        current_technical_fit="qualified",
        settings=settings,
    )
    assert payload["commercial_fit"] == "negotiation_required"
    assert payload["human_handoff_recommended"] is True
    assert "40" in payload["public_pricing_guidance"]
    assert "Xirvo team" in payload["public_pricing_guidance"]


def test_technical_qualification_without_price_question_does_not_instruct_pricing(settings) -> None:
    payload = run_evaluate_lead_fit(
        {
            "software_service_needed": True,
            "project_type": "saas_web_application",
            "technologies": ["typescript"],
        },
        settings=settings,
    )
    guidance = payload["public_pricing_guidance"].lower()
    assert payload["technical_fit"] == "qualified"
    assert payload["commercial_fit"] == "pending"
    assert "40" not in payload["public_pricing_guidance"]
    assert "500" not in payload["public_pricing_guidance"]
    assert "do not mention" in guidance
    assert "technical qualification does not authorize" in guidance
