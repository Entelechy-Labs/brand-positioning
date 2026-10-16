"""Agent-level tests: normal cases, edge cases, and malformed inputs."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from agent_brand_positioning.agent import (
    BrandPositioningAgent,
    build_brand_profile,
    build_positioning,
    build_prompt,
    normalize_payload,
    validate_input,
)
from agent_brand_positioning.config import BRAND_POSITIONING_SLUG
from agent_framework.errors import AgentInputError, AgentOutputError
from linkgen_contracts.errors.envelope import ErrorCategory
from linkgen_contracts.research.models import ResearchRequest
from linkgen_contracts.strategy.models import (
    AudienceProfile,
    BrandProfile,
    Positioning,
)
from linkgen_contracts.workflow.models import TaskStatus

from conftest import (
    AUDIENCE_PROFILE_ID,
    TASK_ID,
    make_audience,
    make_brand,
    make_input,
    make_request,
)

FIXED_AT = datetime(2026, 9, 18, 12, 0, tzinfo=timezone.utc)


class TestInputValidation:
    def test_valid_input_round_trips(self) -> None:
        request, brand_name, brand, audiences = validate_input(make_input())
        assert isinstance(request, ResearchRequest)
        assert brand_name == "Jane Doe"
        assert brand is None
        assert len(audiences) == 1 and isinstance(audiences[0], AudienceProfile)

    def test_existing_brand_is_accepted(self) -> None:
        _, _, brand, _ = validate_input(make_input(brand=make_brand()))
        assert isinstance(brand, BrandProfile)
        assert brand.name == "Jane Doe"

    def test_minimal_input_accepted(self) -> None:
        request, brand_name, brand, audiences = validate_input(
            {"request": make_request()}
        )
        assert brand_name == "" and brand is None and audiences == []

    def test_non_dict_input_rejected(self) -> None:
        with pytest.raises(AgentInputError):
            validate_input(42)

    def test_missing_request_rejected(self) -> None:
        with pytest.raises(AgentInputError):
            validate_input({"brand_name": "x"})

    def test_empty_focus_rejected(self) -> None:
        with pytest.raises(AgentInputError):
            validate_input(make_input(request=make_request(focus="   ")))

    def test_invalid_brand_rejected(self) -> None:
        with pytest.raises(AgentInputError):
            validate_input(make_input(brand={"name": ""}))

    def test_invalid_audience_rejected(self) -> None:
        with pytest.raises(AgentInputError):
            validate_input(make_input(audiences=["junk"]))


class TestNormalRun:
    def test_process_produces_contract_valid_output(
        self, agent: BrandPositioningAgent, run_id, task_id
    ) -> None:
        output = agent.process(
            run_id=run_id, task_id=task_id, input_data=make_input()
        )
        assert set(output) == {"brand_profile", "positioning"}
        brand = BrandProfile.model_validate(output["brand_profile"])
        positioning = Positioning.model_validate(output["positioning"])
        assert brand.name
        assert brand.mission and brand.tone
        assert positioning.statement and positioning.justification
        # The positioning must reference the brand it was derived from.
        assert positioning.brand_id == brand.brand_id
        # ... and the audience segment supplied in the input.
        assert str(positioning.audience_id) == AUDIENCE_PROFILE_ID

    def test_supplied_brand_is_used_verbatim(
        self, agent: BrandPositioningAgent, run_id, task_id
    ) -> None:
        output = agent.process(
            run_id=run_id,
            task_id=task_id,
            input_data=make_input(brand=make_brand()),
        )
        brand = BrandProfile.model_validate(output["brand_profile"])
        assert str(brand.brand_id) == "profile_" + "aa" * 16
        assert brand.mission == "Help ops leads ship onboarding that works."
        assert brand.tone == "direct and practical"
        positioning = Positioning.model_validate(output["positioning"])
        assert str(positioning.brand_id) == str(brand.brand_id)

    def test_run_reports_success(
        self, agent: BrandPositioningAgent, run_id, task_id
    ) -> None:
        result = agent.run(run_id=run_id, task_id=task_id, input_data=make_input())
        assert result.status is TaskStatus.SUCCEEDED
        assert result.errors == []

    def test_config_declares_expected_boundaries(
        self, agent: BrandPositioningAgent
    ) -> None:
        assert agent.config.slug == BRAND_POSITIONING_SLUG
        assert "publish" in agent.config.forbidden_tools
        assert "schedule" in agent.config.forbidden_tools
        assert agent.config.response_format.value == "json"
        assert set(agent.config.output_contracts) == {"BrandProfile", "Positioning"}


class TestRunFailures:
    def test_malformed_input_maps_to_failed_result(
        self, agent: BrandPositioningAgent, run_id, task_id
    ) -> None:
        result = agent.run(run_id=run_id, task_id=TASK_ID, input_data="junk")
        assert result.status is TaskStatus.FAILED
        assert result.errors[0].category is ErrorCategory.VALIDATION

    def test_missing_request_maps_to_failed_result(
        self, agent: BrandPositioningAgent, run_id, task_id
    ) -> None:
        result = agent.run(run_id=run_id, task_id=task_id, input_data={})
        assert result.status is TaskStatus.FAILED
        assert result.errors[0].code == "agent_input_invalid"


class TestArtifactBuilders:
    def test_normalize_payload_requires_objects(self) -> None:
        brand, positioning = normalize_payload({"brand": {}, "positioning": {}})
        assert brand == {} and positioning == {}
        brand, positioning = normalize_payload({})
        assert brand == {} and positioning == {}
        with pytest.raises(AgentOutputError):
            normalize_payload({"brand": "nope"})
        with pytest.raises(AgentOutputError):
            normalize_payload({"positioning": ["nope"]})

    def test_build_brand_profile_fallbacks(self, run_id) -> None:
        request = ResearchRequest.model_validate(make_request())
        brand = build_brand_profile(
            run_id=run_id,
            request=request,
            brand_name="",
            existing=None,
            brand_payload={},
            captured_at=FIXED_AT,
        )
        assert request.focus in brand.name
        assert brand.mission
        assert brand.tone  # default tone, never empty
        assert str(brand.brand_id).startswith("profile_")

    def test_build_brand_profile_sanitizes_lists(self, run_id) -> None:
        request = ResearchRequest.model_validate(make_request())
        brand = build_brand_profile(
            run_id=run_id,
            request=request,
            brand_name="Jane Doe",
            existing=None,
            brand_payload={
                "name": "  Jane   Doe  ",
                "mission": "m",
                "tone": "t",
                "values": ["a", "", 5, "b"],
                "topics_we_own": "not a list",
                "topics_to_avoid": [""] ,
            },
            captured_at=FIXED_AT,
        )
        assert brand.name == "Jane Doe"
        assert brand.values == ["a", "b"]
        assert brand.topics_we_own == []
        assert brand.topics_to_avoid == []

    def test_build_positioning_fallbacks_and_reference(self, run_id) -> None:
        request = ResearchRequest.model_validate(make_request())
        brand = build_brand_profile(
            run_id=run_id,
            request=request,
            brand_name="Jane Doe",
            existing=None,
            brand_payload={"name": "Jane Doe", "mission": "m", "tone": "t"},
            captured_at=FIXED_AT,
        )
        audience = AudienceProfile.model_validate(make_audience())
        positioning = build_positioning(
            run_id=run_id,
            request=request,
            brand=brand,
            audiences=[audience],
            positioning_payload={},
            generated_at=FIXED_AT,
        )
        assert request.focus in positioning.statement
        assert positioning.justification
        assert positioning.brand_id == brand.brand_id
        assert positioning.audience_id == audience.profile_id

    def test_prompt_contains_focus_and_ground_rules(self) -> None:
        request = ResearchRequest.model_validate(make_request())
        prompt = build_prompt(request, "Jane Doe", [])
        assert request.focus in prompt
        assert "Content owner: Jane Doe" in prompt
        assert "desperate client-seeker" in prompt
        assert "Return JSON with 'brand' and 'positioning' objects." in prompt
