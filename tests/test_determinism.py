"""Determinism and id-stability regression tests (golden path).

The same input, clock, and router must produce identical artifacts; different
run ids must produce different derived ids (no cross-run bleed).
"""

from __future__ import annotations

from agent_brand_positioning.agent import BrandPositioningAgent
from agent_framework.ids import deterministic_id
from linkgen_contracts.common.ids import ProfileId, RunId, TaskId

from conftest import RUN_ID, TASK_ID, make_context, make_input, make_request


def _run_agent(payload: dict, *, run_id: str = RUN_ID) -> dict:
    agent = BrandPositioningAgent(context=make_context())
    return agent.process(
        run_id=RunId(run_id), task_id=TaskId(TASK_ID), input_data=payload
    )


def test_same_input_same_output() -> None:
    first = _run_agent(make_input())
    second = _run_agent(make_input())
    assert first == second


def test_brand_id_is_derived_deterministically() -> None:
    output = _run_agent(make_input())
    expected = deterministic_id(ProfileId, RUN_ID, "brand", "jane doe")
    assert output["brand_profile"]["brand_id"] == expected


def test_positioning_id_is_derived_deterministically() -> None:
    from conftest import REQUEST_ID

    output = _run_agent(make_input())
    expected = deterministic_id(ProfileId, RUN_ID, "positioning", REQUEST_ID)
    assert output["positioning"]["positioning_id"] == expected


def test_different_run_id_yields_different_ids() -> None:
    other_run = "run_" + "e4" * 16
    base = _run_agent(make_input())
    other = _run_agent(make_input(run_id=other_run), run_id=other_run)
    assert base["brand_profile"]["brand_id"] != other["brand_profile"]["brand_id"]
    assert (
        base["positioning"]["positioning_id"]
        != other["positioning"]["positioning_id"]
    )


def test_positioning_id_ignores_focus_text() -> None:
    # positioning_id derives from (run_id, request_id), so a re-run under the
    # same run/request is idempotent at the artifact level even if the focus
    # wording changes.
    first = _run_agent(make_input())
    second = _run_agent(make_input(request=make_request(focus="a new focus")))
    assert (
        first["positioning"]["positioning_id"]
        == second["positioning"]["positioning_id"]
    )
