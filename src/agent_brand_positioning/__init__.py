"""brand-positioning — defines expert positioning (Phase 09).

Strategy-phase agent of the LinkGen content-intelligence system. Consumes the
research/audience outputs and produces contract-conforming ``BrandProfile``
and ``Positioning`` artifacts through the shared agent framework and
ModelRouter (mock mode by default).
"""

from __future__ import annotations

from agent_brand_positioning.config import (
    BRAND_POSITIONING_OUTPUT_SCHEMA,
    BRAND_POSITIONING_SLUG,
    build_brand_positioning_config,
)
from agent_brand_positioning.agent import BrandPositioningAgent
from agent_brand_positioning.version import __version__

__all__ = [
    "__version__",
    "BRAND_POSITIONING_OUTPUT_SCHEMA",
    "BRAND_POSITIONING_SLUG",
    "BrandPositioningAgent",
    "build_brand_positioning_config",
]
