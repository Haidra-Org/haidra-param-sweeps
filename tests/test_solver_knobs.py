# SPDX-FileCopyrightText: 2026 Tazlin
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The schedule and solver-knob strips.

These knobs differ from every parameter the library swept before in one way that matters: unset is
not a value. A job that does not state any of them has to reach both backends exactly as it did
before they existed, or every strip already published would be rendering something new.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from param_sweeps.horde_runner import HordeGenerationRequest, unsupported_parameters_in
from param_sweeps.local_runner import HordelibPayload
from param_sweeps.spec import SweepParameter, SweepSpec, load_spec

TOOL_ROOT = Path(__file__).resolve().parents[1]
SWEEP_SPEC_PATH = TOOL_ROOT / "sweep_spec.json"

SOLVER_PARAMETERS = (
    SweepParameter.SCHEDULER,
    SweepParameter.SAMPLER_ETA,
    SweepParameter.SAMPLER_S_CHURN,
    SweepParameter.SAMPLER_SOLVER_TYPE,
    SweepParameter.SCHEDULER_STEPS_GRID,
)


@pytest.fixture(scope="module")
def sweep_spec() -> SweepSpec:
    return load_spec(SWEEP_SPEC_PATH)


class TestUnsetKnobsChangeNothing:
    """A job that never mentions a solver knob reaches both backends exactly as it did before."""

    def test_a_job_that_states_no_knob_carries_none_into_the_local_payload(self, sweep_spec: SweepSpec) -> None:
        """Assert an unset solver knob is absent from the local runner payload, not sent as a default."""
        job = next(job for job in sweep_spec.expand_jobs() if job.parameter is SweepParameter.STEPS)
        payload = HordelibPayload.from_job(job).as_hordelib_mapping()

        for field_name in ("scheduler", "sampler_eta", "sampler_s_noise", "sampler_s_churn", "flow_shift"):
            assert field_name not in payload, f"{field_name} reached a payload that never asked for it"

    def test_a_job_that_states_no_knob_carries_none_into_the_request_body(self, sweep_spec: SweepSpec) -> None:
        """Assert an unset solver knob is absent from the horde request body, not sent as a default."""
        job = next(job for job in sweep_spec.expand_jobs() if job.parameter is SweepParameter.STEPS)
        params = HordeGenerationRequest.from_job(job).as_request_body()["params"]

        for field_name in ("scheduler", "sampler_eta", "sampler_s_noise", "sampler_s_churn", "flow_shift"):
            assert field_name not in params, f"{field_name} reached a request that never asked for it"

    def test_nothing_reports_an_unsupported_parameter_it_did_not_ask_for(self, sweep_spec: SweepSpec) -> None:
        """Assert a job that sets no solver knob reports no unsupported parameters at all."""
        job = next(job for job in sweep_spec.expand_jobs() if job.parameter is SweepParameter.STEPS)
        assert unsupported_parameters_in(job) == []


class TestSolverStrips:
    """The strips that teach the new schedule and solver-knob parameters, and what each one varies."""

    def test_every_new_parameter_has_a_strip(self, sweep_spec: SweepSpec) -> None:
        """Assert every solver parameter has at least one strip demonstrating it."""
        swept = {strip.parameter for strip in sweep_spec.expand_strips()}
        missing = [str(parameter) for parameter in SOLVER_PARAMETERS if parameter not in swept]
        assert missing == [], f"parameters with nothing to show for them: {missing}"

    def test_the_schedule_strip_covers_every_schedule_the_backend_offers(self, sweep_spec: SweepSpec) -> None:
        """Assert the schedule strip's jobs cover every schedule the backend declares as known."""
        from horde_sdk.generation_parameters.image.consts import KNOWN_IMAGE_SCHEDULERS

        swept = {
            str(job.scheduler)
            for job in sweep_spec.expand_jobs()
            if job.parameter is SweepParameter.SCHEDULER and job.scheduler is not None
        }
        assert swept == {str(schedule) for schedule in KNOWN_IMAGE_SCHEDULERS}

    def test_the_eta_ladder_includes_the_value_the_claim_is_about(self, sweep_spec: SweepSpec) -> None:
        """Assert the eta strip includes zero, the value the deterministic-twin claim turns on."""
        # eta 0 is the stop that can falsify the collapse-onto-the-deterministic-twin claim.
        eta_values = {
            job.sampler_eta for job in sweep_spec.expand_jobs() if job.parameter is SweepParameter.SAMPLER_ETA
        }
        assert 0.0 in eta_values

    def test_a_strip_varies_only_the_knob_it_teaches(self, sweep_spec: SweepSpec) -> None:
        """Assert an eta or s_churn strip holds seed, sampler and steps constant across its jobs."""
        for parameter in (SweepParameter.SAMPLER_ETA, SweepParameter.SAMPLER_S_CHURN):
            jobs = [job for job in sweep_spec.expand_jobs() if job.parameter is parameter]
            assert len({job.seed for job in jobs}) == 1
            assert len({job.sampler_name for job in jobs}) == 1
            assert len({job.steps for job in jobs}) == 1

    def test_the_schedule_grid_crosses_schedules_with_step_budgets(self, sweep_spec: SweepSpec) -> None:
        """Assert the schedule-steps grid renders every schedule at every step budget, not a sample of pairs."""
        jobs = [job for job in sweep_spec.expand_jobs() if job.parameter is SweepParameter.SCHEDULER_STEPS_GRID]
        schedules = {str(job.scheduler) for job in jobs}
        budgets = {job.steps for job in jobs}
        assert len(jobs) == len(schedules) * len(budgets)
        assert {"align_your_steps", "gits"} <= schedules

    def test_a_stated_knob_reaches_both_backends(self, sweep_spec: SweepSpec) -> None:
        """Assert a job that states sampler_eta passes it into both payloads and into the unsupported list."""
        job = next(job for job in sweep_spec.expand_jobs() if job.parameter is SweepParameter.SAMPLER_ETA)

        payload = HordelibPayload.from_job(job).as_hordelib_mapping()
        assert payload["sampler_eta"] == job.sampler_eta

        params = HordeGenerationRequest.from_job(job).as_request_body()["params"]
        assert params["sampler_eta"] == job.sampler_eta

        assert "sampler_eta" in unsupported_parameters_in(job)
