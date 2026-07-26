# SPDX-FileCopyrightText: 2026 Tazlin
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The schedule and solver-knob strips, and the signoff batch that puts the claims to a ruling.

These knobs differ from every parameter the library swept before in one way that matters: unset is
not a value. A job that states none of them has to reach both backends exactly as it did before they
existed, or every strip already published would be rendering something new.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from param_sweeps.horde_runner import HordeGenerationRequest, unsupported_parameters_in
from param_sweeps.local_runner import HordelibPayload
from param_sweeps.spec import (
    RULING_OTHER_OPTION_ID,
    BatchSection,
    RulingControl,
    RulingItem,
    RulingSet,
    SweepParameter,
    SweepSpec,
    load_spec,
)

TOOL_ROOT = Path(__file__).resolve().parents[1]
SWEEP_SPEC_PATH = TOOL_ROOT / "sweep_spec.json"
SIGNOFF_SPEC_PATH = TOOL_ROOT / "signoff_solver_knobs_spec.json"

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


@pytest.fixture(scope="module")
def signoff_spec() -> SweepSpec:
    return load_spec(SIGNOFF_SPEC_PATH)


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
        """Assert every solver parameter this wave adds has at least one strip demonstrating it."""
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
        """Assert a job that states sampler_eta carries it into both payloads and into the unsupported list."""
        job = next(job for job in sweep_spec.expand_jobs() if job.parameter is SweepParameter.SAMPLER_ETA)

        payload = HordelibPayload.from_job(job).as_hordelib_mapping()
        assert payload["sampler_eta"] == job.sampler_eta

        params = HordeGenerationRequest.from_job(job).as_request_body()["params"]
        assert params["sampler_eta"] == job.sampler_eta

        assert "sampler_eta" in unsupported_parameters_in(job)


class TestSignoffBatch:
    """The batch that renders the images this wave's rulings are judged against."""

    def test_it_covers_the_three_sections_it_was_written_for(self, signoff_spec: SweepSpec) -> None:
        """Assert the batch's strips cover all three sections: suspect claims, goldens and raw backend."""
        sections = {strip.batch_section for strip in signoff_spec.expand_strips() if strip.batch_section is not None}
        assert sections == {
            BatchSection.SUSPECT_CLAIMS,
            BatchSection.MINTED_GOLDENS,
            BatchSection.RAW_BACKEND,
        }

    def test_the_raw_backend_section_covers_both_baselines_at_both_step_counts(
        self,
        signoff_spec: SweepSpec,
    ) -> None:
        """Assert the raw-backend section renders both schedules, both step counts and both baselines."""
        # These images cannot be produced by either runner, so the section stands or falls on
        # covering the pairing completely: two schedules, two step counts, both baselines.
        raw_jobs = [
            job
            for strip in signoff_spec.expand_strips()
            if strip.batch_section is BatchSection.RAW_BACKEND
            for job in strip.jobs
        ]
        assert len(raw_jobs) == 8
        assert {job.sampler_name for job in raw_jobs} == {"dpmpp_3m_sde"}
        assert {str(job.scheduler) for job in raw_jobs} == {"normal", "simple"}
        assert {job.steps for job in raw_jobs} == {8, 25}
        assert {job.model_slug for job in raw_jobs} == {"deliberate", "sdxl-1-0"}

    def test_the_divergent_pairing_is_rendered_against_the_schedules_it_is_measured_against(
        self,
        signoff_spec: SweepSpec,
    ) -> None:
        """Assert the disputed sampler is rendered on every schedule the ruling weighs, at both step counts."""
        # The one hard rejection resting on measurement rather than on code, at both step counts the
        # original measurement used.
        jobs = [
            job
            for strip in signoff_spec.expand_strips()
            if strip.batch_section is BatchSection.SUSPECT_CLAIMS
            for job in strip.jobs
            if job.sampler_name == "dpmpp_3m_sde"
        ]
        assert {str(job.scheduler) for job in jobs} == {"normal", "karras", "simple", "sgm_uniform"}
        assert {job.steps for job in jobs} == {8, 25}

    def test_every_golden_minted_this_wave_is_presented(self, signoff_spec: SweepSpec) -> None:
        """Assert every reference image this wave mints, sampler and schedule, is presented for approval."""
        golden_jobs = [
            job
            for strip in signoff_spec.expand_strips()
            if strip.batch_section is BatchSection.MINTED_GOLDENS
            for job in strip.jobs
        ]
        sampler_goldens = [job for job in golden_jobs if job.scene_id.startswith("golden-sampler-")]
        schedule_goldens = [job for job in golden_jobs if job.scene_id.startswith("golden-schedule-")]

        assert len(sampler_goldens) == 15, "one per sampler whose reference image this wave minted"
        assert len(schedule_goldens) == 4, "two schedules on each of the two baselines"

    def test_the_golden_re_renders_use_the_reference_suite_parameters(self, signoff_spec: SweepSpec) -> None:
        """Assert every golden re-render uses the reference suite's fixed seed, steps, cfg and size."""
        # A re-render on different parameters would review a different image than the one on disk.
        sampler_goldens = [job for job in signoff_spec.expand_jobs() if job.scene_id.startswith("golden-sampler-")]
        for job in sampler_goldens:
            assert job.seed == 3688490319
            assert job.steps == 30
            assert job.cfg_scale == 6.5
            assert (job.width, job.height) == (512, 512)
            assert str(job.scheduler) == "normal", "the reference suite renders with karras off"


SOLVER_KNOB_RULINGS = "signoff-solver-knobs"
SCHEDULE_CHARACTER_RULINGS = "signoff-schedule-characters"


def _solver_knob_rulings(spec: SweepSpec) -> RulingSet:
    return spec.ruling_set_for_id(SOLVER_KNOB_RULINGS)


def _ruling(spec: SweepSpec, ruling_id: str) -> RulingItem:
    return next(item for item in _solver_knob_rulings(spec).items if item.ruling_id == ruling_id)


class TestRulings:
    """The questions this wave cannot ship without answers to, as the specification states them."""

    def test_every_claim_this_batch_renders_for_has_a_ruling_to_answer_it(self, signoff_spec: SweepSpec) -> None:
        """Assert the solver-knob ruling set lists exactly the ruling ids this batch was built to answer."""
        assert [item.ruling_id for item in _solver_knob_rulings(signoff_spec).items] == [
            "dpmpp-3m-sde-normal",
            "karras-at-low-steps",
            "schedule-character-claims",
            "per-sampler-eta-ranges",
            "cfg-pp-guidance",
            "ays-gits-at-low-steps",
            "minted-golden-correctness",
            "sampler-presentation-curation",
            "cost-followup",
            "learn-publishing",
            "administrative",
        ]

    def test_every_ruling_states_the_question_it_is_answering(self, signoff_spec: SweepSpec) -> None:
        """Assert every ruling item carries a non-empty title and a non-empty question."""
        for item in _solver_knob_rulings(signoff_spec).items:
            assert item.title.strip(), f"{item.ruling_id} has no title"
            assert item.question.strip(), f"{item.ruling_id} states no question"

    def test_no_ruling_forces_an_answer_that_does_not_fit(self, signoff_spec: SweepSpec) -> None:
        """Assert every ruling, and every row of a row-select ruling, offers the other-option escape."""
        for item in _solver_knob_rulings(signoff_spec).items:
            if item.control is RulingControl.ROW_SELECT:
                for row in item.rows:
                    offered = [option.choice_id for option in item.presented_options(row=row)]
                    assert RULING_OTHER_OPTION_ID in offered, f"{item.ruling_id}/{row.row_id} offers no escape"
            else:
                offered = [option.choice_id for option in item.presented_options()]
                assert RULING_OTHER_OPTION_ID in offered, f"{item.ruling_id} offers no escape"

    def test_the_divergence_ruling_points_at_the_renders_that_decide_it(self, signoff_spec: SweepSpec) -> None:
        """Assert the disputed-pairing ruling cites the raw-backend section and offers a single choice."""
        # The raw-backend section is the whole reason that section exists: it is the only evidence
        # with nothing in the path that could have altered the pairing before it was judged.
        item = _ruling(signoff_spec, "dpmpp-3m-sde-normal")

        assert BatchSection.RAW_BACKEND in item.evidence_sections
        assert item.control is RulingControl.RADIO

    def test_the_golden_checklist_covers_every_golden_the_batch_re_renders(self, signoff_spec: SweepSpec) -> None:
        """Assert no minted reference image can ship without having been approved one by one."""
        item = _ruling(signoff_spec, "minted-golden-correctness")
        golden_job_ids = {
            job.scene_id
            for strip in signoff_spec.expand_strips()
            if strip.batch_section is BatchSection.MINTED_GOLDENS
            for job in strip.jobs
        }

        assert item.control is RulingControl.CHECKBOX_SET
        assert item.master_toggle_label is not None, "nineteen boxes need a way to tick them at once"
        assert {option.evidence_job_id for option in item.options} == golden_job_ids

    def test_the_follow_up_set_is_keyed_apart_from_the_wave_that_was_already_ruled(
        self,
        signoff_spec: SweepSpec,
    ) -> None:
        """Assert the supplemental questions are their own set, so ruled answers are not reopened."""
        assert [ruling_set.ruling_set_id for ruling_set in signoff_spec.rulings] == [
            SOLVER_KNOB_RULINGS,
            SCHEDULE_CHARACTER_RULINGS,
        ]

    def test_the_administrative_ruling_answers_its_two_decisions_separately(self, signoff_spec: SweepSpec) -> None:
        """Assert the administrative ruling has one row per decision, each with its own options."""
        item = _ruling(signoff_spec, "administrative")

        assert [row.row_id for row in item.rows] == ["boto3-in-the-hordelib-venv", "worker-version-meta"]
        for row in item.rows:
            assert row.options, f"{row.row_id} would otherwise be answered from another decision's options"


class TestScheduleCharacterEvidence:
    """The supplemental renders, and the set that rules on what they show.

    The three claims here were left unsure because the sheet carried no example of them. What makes
    an example answer a claim is the reference beside it: a schedule looks washed out or blurred
    only against something, so each render has a ``simple`` twin that differs in the schedule and
    in nothing else.
    """

    def test_it_rules_on_exactly_the_three_schedules_the_first_pass_left_unsure(
        self,
        signoff_spec: SweepSpec,
    ) -> None:
        """Assert the follow-up set rules on exactly the three schedules and offers accurate/inaccurate/unsure."""
        ruling_set = signoff_spec.ruling_set_for_id(SCHEDULE_CHARACTER_RULINGS)

        assert [item.ruling_id for item in ruling_set.items] == [
            "beta-cast-and-posterisation",
            "ddim-uniform-washed-out",
            "linear-quadratic-blur",
        ]
        for item in ruling_set.items:
            assert item.control is RulingControl.ROW_SELECT
            offered = {option.choice_id for option in item.presented_options(row=item.rows[0])}
            assert offered == {"accurate", "inaccurate", "unsure", RULING_OTHER_OPTION_ID}

    def test_every_row_is_a_render_that_has_a_reference_differing_only_in_the_schedule(
        self,
        signoff_spec: SweepSpec,
    ) -> None:
        """Assert each cited render has a ``simple`` twin at its own model, sampler, steps and seed."""
        jobs_by_id = {job.scene_id: job for job in signoff_spec.expand_jobs()}
        references = [job for job in jobs_by_id.values() if str(job.scheduler) == "simple"]
        ruling_set = signoff_spec.ruling_set_for_id(SCHEDULE_CHARACTER_RULINGS)

        for item in ruling_set.items:
            for row in item.rows:
                assert row.evidence_job_id is not None, f"{item.ruling_id}/{row.row_id} cites no render"
                job = jobs_by_id[row.evidence_job_id]
                assert str(job.scheduler) != "simple"
                assert any(
                    reference.model_slug == job.model_slug
                    and reference.sampler_name == job.sampler_name
                    and reference.steps == job.steps
                    and reference.seed == job.seed
                    and reference.prompt == job.prompt
                    for reference in references
                ), f"{row.evidence_job_id} has no simple render to be read against"

    def test_the_claims_are_rendered_at_the_budgets_and_models_they_are_stated_at(
        self,
        signoff_spec: SweepSpec,
    ) -> None:
        """Assert each supplemental schedule is rendered at both step budgets and on the models its claim names."""
        supplemental = {
            job.scene_id: job
            for job in signoff_spec.expand_jobs()
            if str(job.scheduler) in {"beta", "ddim_uniform", "linear_quadratic"}
        }

        for schedule in ("beta", "ddim_uniform", "linear_quadratic"):
            budgets = {job.steps for job in supplemental.values() if str(job.scheduler) == schedule}
            assert {8, 25} <= budgets, f"{schedule} is not rendered at both a short and a long budget"

        # The blur claim is stated about models that are not flow models, so it is put to both.
        linear_quadratic_models = {
            job.model_slug for job in supplemental.values() if str(job.scheduler) == "linear_quadratic"
        }
        assert linear_quadratic_models == {"deliberate", "sdxl-1-0"}

        # The degradation claim is stated about higher-order SDE samplers specifically.
        sde_pairing = supplemental["sde-schedule-ddim_uniform"]
        assert sde_pairing.sampler_name == "dpmpp_2m_sde"
        assert sde_pairing.steps == 25
