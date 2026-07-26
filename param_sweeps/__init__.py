# SPDX-FileCopyrightText: 2026 Tazlin
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Render the parameter-sweep illustration library the Learn surfaces teach from.

A single checked-in specification (``sweep_spec.json``) declares every strip: which models,
which scenes, which parameter varies and what each stop teaches. Two interchangeable runners
render that specification, one through a local GPU (``horde-engine``) and one through the
public AI Horde API, and both write the same files.

Public entry points:

- [`SweepSpec`][param_sweeps.spec.SweepSpec]: the parsed specification and its expansion into
  strips and render jobs.
- [`run_campaign`][param_sweeps.campaign.run_campaign]: the resumable driver shared by both runners.
- [`LocalRunner`][param_sweeps.local_runner.LocalRunner] and
  [`HordeApiRunner`][param_sweeps.horde_runner.HordeApiRunner]: the two backends.
- [`build_index`][param_sweeps.outputs.build_index]: the ``index.json`` a web app renders from.
- [`enumerate_library`][param_sweeps.publish.enumerate_library] and
  [`plan_sync`][param_sweeps.publish.plan_sync]: what the library consists of, and what publishing
  it to S3-compatible storage would do.

``param_sweeps.local_runner`` imports ``hordelib`` lazily, so the specification, index and API
runner all work in an environment with no GPU stack installed.
"""

from __future__ import annotations

from param_sweeps.campaign import CampaignResult, run_campaign
from param_sweeps.checkpoint import CampaignCheckpoint
from param_sweeps.outputs import build_index, write_index
from param_sweeps.publish import SyncPlan, enumerate_library, plan_sync
from param_sweeps.runner import SweepRunner
from param_sweeps.sampler_cost import SamplerCostReport, measure_sampler_cost
from param_sweeps.spec import RenderJob, RenderStrip, SweepParameter, SweepSpec, load_spec

__all__ = [
    "CampaignCheckpoint",
    "CampaignResult",
    "RenderJob",
    "RenderStrip",
    "SamplerCostReport",
    "SweepParameter",
    "SweepRunner",
    "SweepSpec",
    "SyncPlan",
    "build_index",
    "enumerate_library",
    "load_spec",
    "measure_sampler_cost",
    "plan_sync",
    "run_campaign",
    "write_index",
]
