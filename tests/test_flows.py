# SPDX-FileCopyrightText: 2026 Tazlin
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Check the tutorial flows: that the specification loads, and that a flow's request reaches the API intact."""

from __future__ import annotations

from pathlib import Path

import pytest

from param_sweeps.flows import FlowsSpec, load_flows_spec, publishable_request
from param_sweeps.horde_runner import HordeGenerationRequest

FLOWS_SPEC_PATH = Path(__file__).resolve().parents[1] / "flows_spec.json"

QR_FLOW_ID = "qr_code"
"""The flow whose recipe was scan-tested rather than inherited, and so is the one worth asserting."""


@pytest.fixture(scope="module")
def flows_spec() -> FlowsSpec:
    """Return the checked-in flow specification."""
    return load_flows_spec(FLOWS_SPEC_PATH)


def test_control_strength_survives_the_whole_way_to_the_request_body(flows_spec: FlowsSpec) -> None:
    """Assert a flow's control strength reaches the payload rather than being dropped in translation.

    It travels through two hand-written translations, ``JobFeatures`` to ``RenderJob`` and
    ``RenderJob`` to the request body, and a field missing from either one fails silently: the
    request still generates, at whatever strength the backend defaults to.
    """
    flow = flows_spec.flow_for_id(QR_FLOW_ID)
    declared = flow.request.features.control_strength
    assert declared is not None, "the QR flow states a control strength, which is what this checks the plumbing with"

    job = flow.request.to_render_job(flow_id=flow.flow_id, title=flow.title)
    assert job.control_strength == declared

    body = HordeGenerationRequest.from_job(job).as_request_body()
    assert body["params"]["control_strength"] == declared
    assert (
        publishable_request(flow.request, flow_id=flow.flow_id, title=flow.title)["params"]["control_strength"]
        == declared
    )


def test_the_qr_flow_runs_the_recipe_that_was_scan_tested(flows_spec: FlowsSpec) -> None:
    """Assert the QR flow keeps the measured recipe rather than drifting back to the anchor's defaults.

    A QR code that does not scan is not a worked example of anything, so these five values are the
    flow's teaching payload rather than incidental parameters. The strength is stated at the workflow's
    own baseline of 1.0, which is what a worker renders this code at, and the published example is
    the one that decoded through that path.
    """
    request = flows_spec.flow_for_id(QR_FLOW_ID).request

    assert request.params.sampler_name == "k_euler"
    assert request.params.cfg_scale == 7.5
    assert request.params.steps == 25
    assert request.params.karras is False
    assert request.features.control_strength == 1.0
