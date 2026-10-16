"""Typed definition of the brand-positioning agent.

The ``AgentConfig`` is the definition-time contract of this agent: mission,
model role, allowed/forbidden tools, output contracts, and the JSON schema
its LLM calls must satisfy. The router validates model output against this
schema (fail closed) before the agent ever sees it.
"""

from __future__ import annotations

from typing import Any

from agent_framework.config import AgentConfig
from linkgen_contracts.llm.models import LLMRole, ResponseFormat

BRAND_POSITIONING_SLUG = "brand-positioning"

# JSON schema for the model's structured output. Restricted to the router's
# documented schema subset (type/properties/items/required/enum/const) so it
# fails closed rather than being silently ignored. Ids and timestamps are
# NEVER requested from the model — deterministic Python derives them.
BRAND_POSITIONING_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "brand": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "mission": {"type": "string"},
                "tone": {"type": "string"},
                "values": {"type": "array", "items": {"type": "string"}},
                "voice_guidelines": {"type": "array", "items": {"type": "string"}},
                "topics_we_own": {"type": "array", "items": {"type": "string"}},
                "topics_to_avoid": {"type": "array", "items": {"type": "string"}},
            },
        },
        "positioning": {
            "type": "object",
            "properties": {
                "statement": {"type": "string"},
                "differentiation": {"type": "array", "items": {"type": "string"}},
                "justification": {"type": "string"},
            },
        },
    },
}

_SYSTEM_PROMPT = """You are the brand-positioning agent of the LinkGen content system.
Mission: position the content owner as a credible practitioner/expert for the
requested focus and audience.
Write a positioning statement, concrete differentiators, and a justification
grounded in the supplied research and audience segments.
The result must NOT sound like: a generic agency, a motivational speaker, a
corporate social manager, a salesperson, an AI-generated content account, or
a desperate client-seeker.
Never invent credentials, results, statistics, or personal stories that were
not supplied.
Respond only with JSON matching the requested schema."""


def build_brand_positioning_config() -> AgentConfig:
    """Build the definition-time configuration of the positioning agent."""
    return AgentConfig(
        slug=BRAND_POSITIONING_SLUG,
        model_role=LLMRole.REASONING,
        output_contracts=[
            "BrandProfile",
            "Positioning",
        ],
        system_prompt=_SYSTEM_PROMPT,
        temperature=0.0,
        max_output_tokens=2048,
        max_steps=8,
        tools=["web-search"],
        forbidden_tools=["publish", "schedule", "linkedin-post"],
        response_format=ResponseFormat.JSON,
        json_schema=BRAND_POSITIONING_OUTPUT_SCHEMA,
        description="Define expert positioning (strategy phase).",
        version="1.0.0",
    )
