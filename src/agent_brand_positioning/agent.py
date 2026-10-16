"""Brand-positioning agent — Phase 09 strategy system.

Consumes the research/audience phase outputs and produces contract-conforming
brand artifacts:

- ``BrandProfile`` (tone, voice, owned/avoided topics)
- ``Positioning`` (statement + differentiation + justification)

Design rules (shared with the Phase 08 research agents):
- The LLM is used only for content judgment; ids, timestamps, dedupe, capping,
  and confidence bounds are deterministic Python.
- Model output is untrusted: it arrives router-validated as JSON, and every
  field is sanitized before entering a contract model.
- Ids are derived deterministically (uuid5) so identical inputs produce
  identical artifacts (idempotent replay, golden tests, checkpoint recovery).
- No network, clock, or randomness: time comes from the injected context
  clock; failures come from the router as duck-typed-normalized errors.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from linkgen_contracts.common.ids import ProfileId, RunId, TaskId
from linkgen_contracts.errors.envelope import ErrorCategory
from linkgen_contracts.research.models import ResearchRequest
from linkgen_contracts.strategy.models import (
    AudienceProfile,
    BrandProfile,
    Positioning,
)
from pydantic import ValidationError

from agent_brand_positioning.config import (
    BRAND_POSITIONING_OUTPUT_SCHEMA,
    build_brand_positioning_config,
)
from agent_framework.base import AgentBase
from agent_framework.config import AgentConfig
from agent_framework.context import AgentContext
from agent_framework.errors import (
    AgentError,
    AgentInputError,
    AgentOutputError,
)
from agent_framework.ids import deterministic_id
from agent_framework.sanitize import (
    bounded_str_list,
    clean_text,
    normalize_key,
    stable_probability,
)
from agent_framework.structured import extract_json_dict

# Bounded resource usage (AGENTS rule 11).
MAX_TOPICS = 12
MAX_GUIDELINES = 10
MAX_DIFFERENTIATION = 8
DEFAULT_TONE = "practical, specific, and direct"

__all__ = [
    "BrandPositioningAgent",
    "build_brand_profile",
    "build_positioning",
    "build_prompt",
    "normalize_payload",
    "validate_input",
    "normalize_router_failure",
]


# ---------------------------------------------------------------------------
# Untrusted-input normalization (agent-specific; sanitizers are shared).
# ---------------------------------------------------------------------------

def normalize_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split untrusted model output into ``brand`` / ``positioning`` dicts."""
    brand = payload.get("brand")
    positioning = payload.get("positioning")
    if brand is None:
        brand = {}
    if positioning is None:
        positioning = {}
    if not isinstance(brand, dict) or not isinstance(positioning, dict):
        raise AgentOutputError(
            "model output fields 'brand' and 'positioning' must be objects"
        )
    return brand, positioning


def _validate_model_list(raw: Any, model: type, label: str) -> list[Any]:
    """Validate an optional list of already-contracted input artifacts."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise AgentInputError(f"brand-positioning input '{label}' must be a list")
    validated: list[Any] = []
    for index, item in enumerate(raw):
        try:
            validated.append(model.model_validate(item))
        except ValidationError as exc:
            first = exc.errors()[0]
            location = ".".join(str(part) for part in first.get("loc", ()))
            raise AgentInputError(
                f"{label}[{index}] failed contract validation at '{location}': "
                f"{first.get('msg', 'invalid')}"
            ) from exc
    return validated


def validate_input(
    input_data: Any,
) -> tuple[ResearchRequest, str, BrandProfile | None, list[AudienceProfile]]:
    """Validate the agent input against the research/strategy contracts.

    Input shape::

        {"request": ResearchRequest,              # required
         "brand_name": str,                        # optional content owner
         "brand": BrandProfile,                    # optional existing profile
         "audiences": [AudienceProfile, ...]}      # optional
    """
    if not isinstance(input_data, dict):
        raise AgentInputError(
            "brand-positioning input must be a dict with a 'request' key"
        )
    raw_request = input_data.get("request")
    if raw_request is None:
        raise AgentInputError(
            "brand-positioning input is missing the 'request' object"
        )
    try:
        request = ResearchRequest.model_validate(raw_request)
    except ValidationError as exc:
        first = exc.errors()[0]
        location = ".".join(str(part) for part in first.get("loc", ()))
        raise AgentInputError(
            f"research request failed contract validation at '{location}': "
            f"{first.get('msg', 'invalid')}"
        ) from exc

    brand_name = clean_text(input_data.get("brand_name"), max_len=200)

    raw_brand = input_data.get("brand")
    brand: BrandProfile | None = None
    if raw_brand is not None:
        try:
            brand = BrandProfile.model_validate(raw_brand)
        except ValidationError as exc:
            first = exc.errors()[0]
            location = ".".join(str(part) for part in first.get("loc", ()))
            raise AgentInputError(
                f"brand profile failed contract validation at '{location}': "
                f"{first.get('msg', 'invalid')}"
            ) from exc

    audiences = _validate_model_list(
        input_data.get("audiences"), AudienceProfile, "audiences"
    )
    return request, brand_name, brand, audiences


class _RouterFailure(AgentError):
    """Agent-level wrapper for a router failure (category/code preserved)."""


def normalize_router_failure(exc: Exception) -> AgentError:
    """Normalize a router failure without importing provider internals.

    Router errors carry ``category``/``code``/``retryable``; they are preserved
    by duck-typing so ``TaskResult`` mapping keeps retry semantics honest.
    """
    failure = _RouterFailure(f"router call failed: {getattr(exc, 'message', None) or exc}")
    raw_category = getattr(exc, "category", ErrorCategory.INTERNAL)
    failure.category = raw_category if isinstance(raw_category, ErrorCategory) else ErrorCategory.INTERNAL
    failure.code = str(getattr(exc, "code", "router_failure"))
    failure.retryable = bool(getattr(exc, "retryable", False))
    return failure


def build_prompt(
    request: ResearchRequest,
    brand_name: str,
    audiences: list[AudienceProfile],
) -> str:
    """Deterministic user prompt for the brand-positioning LLM call."""
    lines = [
        "Define brand positioning for a LinkedIn content system.",
        f"Focus: {request.focus}",
    ]
    if brand_name:
        lines.append(f"Content owner: {brand_name}")
    if request.objectives:
        lines.append("Objectives: " + "; ".join(request.objectives))
    if audiences:
        lines.append("Target audience segments:")
        lines.extend(f"- {a.name}: {a.description}" for a in audiences[:6])
    lines.append(
        "Position the content owner as a credible practitioner/expert. The "
        "result must NOT sound like a generic agency, motivational speaker, "
        "corporate social manager, salesperson, AI-generated content account, "
        "or desperate client-seeker. Never invent credentials, results, "
        "statistics, or personal stories that were not supplied. Ground the "
        "statement, differentiation, and justification in the supplied "
        "research."
    )
    lines.append("Return JSON with 'brand' and 'positioning' objects.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Artifact builders (pure functions of their inputs).
# ---------------------------------------------------------------------------

def brand_profile_id(run_id: RunId, request: ResearchRequest, brand_name: str) -> ProfileId:
    """Deterministic brand profile id — stable across runs of the same input."""
    identity = normalize_key(brand_name) or normalize_key(request.focus)
    return ProfileId(
        deterministic_id(ProfileId, str(run_id), "brand", identity or "unspecified")
    )


def build_brand_profile(
    *,
    run_id: RunId,
    request: ResearchRequest,
    brand_name: str,
    existing: BrandProfile | None,
    brand_payload: dict[str, Any],
    captured_at: datetime,
) -> BrandProfile:
    """Build the run's ``BrandProfile`` (an existing brand wins verbatim)."""
    if existing is not None:
        return existing

    name = clean_text(brand_payload.get("name"), max_len=200) or brand_name
    if not name:
        name = f"Practitioner brand for {clean_text(request.focus) or 'this focus'}"
    mission = clean_text(brand_payload.get("mission")) or (
        f"Help the audience of '{clean_text(request.focus) or 'this focus'}' "
        f"make better decisions with practitioner experience."
    )
    return BrandProfile(
        # The id derives from the *input* identity (brand name, else focus),
        # never from model-proposed text, so the brand id stays stable across
        # model variations while the descriptive fields remain model judgment.
        brand_id=brand_profile_id(run_id, request, brand_name),
        name=name,
        mission=mission,
        values=bounded_str_list(brand_payload.get("values"), limit=MAX_TOPICS),
        tone=clean_text(brand_payload.get("tone"), max_len=200) or DEFAULT_TONE,
        voice_guidelines=bounded_str_list(
            brand_payload.get("voice_guidelines"), limit=MAX_GUIDELINES
        ),
        topics_we_own=bounded_str_list(
            brand_payload.get("topics_we_own"), limit=MAX_TOPICS
        ),
        topics_to_avoid=bounded_str_list(
            brand_payload.get("topics_to_avoid"), limit=MAX_TOPICS
        ),
        captured_at=captured_at,
    )


def build_positioning(
    *,
    run_id: RunId,
    request: ResearchRequest,
    brand: BrandProfile,
    audiences: list[AudienceProfile],
    positioning_payload: dict[str, Any],
    generated_at: datetime,
) -> Positioning:
    """Build the run's ``Positioning`` (model fields, focus-derived fallbacks)."""
    statement = clean_text(positioning_payload.get("statement")) or (
        f"Practitioner expertise on {clean_text(request.focus) or 'the focus'} "
        f"for the people who need it."
    )
    justification = clean_text(positioning_payload.get("justification")) or (
        f"Derived from research focus '{clean_text(request.focus) or 'unspecified'}' "
        f"and {len(audiences)} audience segment(s)."
    )
    return Positioning(
        positioning_id=ProfileId(
            deterministic_id(
                ProfileId, str(run_id), "positioning", str(request.request_id)
            )
        ),
        brand_id=brand.brand_id,
        audience_id=(
            audiences[0].profile_id
            if audiences
            else ProfileId(
                deterministic_id(
                    ProfileId, str(run_id), "audience", normalize_key(request.focus)
                )
            )
        ),
        statement=statement,
        differentiation=bounded_str_list(
            positioning_payload.get("differentiation"), limit=MAX_DIFFERENTIATION
        ),
        justification=justification,
        generated_at=generated_at,
    )


# ---------------------------------------------------------------------------
# The agent.
# ---------------------------------------------------------------------------

class BrandPositioningAgent(AgentBase):
    """Define expert positioning (strategy phase, Phase 09).

    Input (validated against research/strategy contracts)::

        {"request": ResearchRequest, "brand_name"?: str,
         "brand"?: BrandProfile, "audiences"?: [AudienceProfile]}

    Output::

        {"brand_profile": BrandProfile, "positioning": Positioning}
    """

    def __init__(
        self, context: AgentContext, config: AgentConfig | None = None
    ) -> None:
        super().__init__(
            config if config is not None else build_brand_positioning_config(),
            context,
        )

    def process(
        self,
        *,
        run_id: RunId,
        task_id: TaskId,
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        request, brand_name, existing_brand, audiences = validate_input(input_data)
        now = self.context.now()

        if existing_brand is not None:
            # The supplied brand is authoritative; no model call needed.
            brand = existing_brand
            brand_payload: dict[str, Any] = {}
            positioning_payload: dict[str, Any] = {}
        else:
            payload = self._gather(
                run_id=run_id, task_id=task_id,
                request=request, brand_name=brand_name, audiences=audiences,
            )
            brand_payload, positioning_payload = normalize_payload(payload)
            brand = build_brand_profile(
                run_id=run_id,
                request=request,
                brand_name=brand_name,
                existing=None,
                brand_payload=brand_payload,
                captured_at=now,
            )

        positioning = build_positioning(
            run_id=run_id,
            request=request,
            brand=brand,
            audiences=audiences,
            positioning_payload=positioning_payload,
            generated_at=now,
        )
        return {
            "brand_profile": brand.model_dump(mode="json"),
            "positioning": positioning.model_dump(mode="json"),
        }

    def _gather(
        self,
        *,
        run_id: RunId,
        task_id: TaskId,
        request: ResearchRequest,
        brand_name: str,
        audiences: list[AudienceProfile],
    ) -> dict[str, Any]:
        """Call the router for structured brand/positioning JSON (single hop).

        Fault markers in the prompt (mock mode) or a failing adapter surface
        as normalized agent errors; they never escape as raw provider errors.
        """
        prompt = build_prompt(request, brand_name, audiences)
        llm_request = self.build_llm_request(
            run_id=run_id,
            task_id=task_id,
            prompt=prompt,
            json_schema=BRAND_POSITIONING_OUTPUT_SCHEMA,
        )
        try:
            result = self.context.router.complete_json(
                llm_request, BRAND_POSITIONING_OUTPUT_SCHEMA
            )
        except AgentError:
            raise
        except Exception as exc:  # noqa: BLE001 — normalized below, never swallowed
            raise normalize_router_failure(exc) from exc
        return extract_json_dict(result)
