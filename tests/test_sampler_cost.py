# SPDX-FileCopyrightText: 2026 Tazlin
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The sampler cost measurement: its fit, its schedule pairings and the artifact it writes.

Nothing here renders. The measurement's whole value is that the numbers it reports can be traced,
so the arithmetic, the pairing rules and the shape of the artifact are checked against stated
inputs rather than against whatever a GPU happened to do.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from param_sweeps.sampler_cost import (
    MEASUREMENT_SCHEMA_VERSION,
    MeasuredModel,
    MeasurementSettings,
    SamplerCostReport,
    ScheduleChoiceReason,
    SchedulePolicy,
    build_records,
    derive_pairings,
    fit_line,
    format_summary_table,
    measurement_job,
    write_report,
)
from param_sweeps.spec import ModelSpec

SD15_BASELINE = "stable_diffusion_1"
SDXL_BASELINE = "stable_diffusion_xl"

POLICY = SchedulePolicy(
    rejected_pairings=frozenset({("moody_sde", "karras"), ("moody_sde", "normal")}),
    recommended_schedulers={"moody_sde": ("normal", "simple"), "restricted_only": ("align_your_steps",)},
    scheduler_baselines={"align_your_steps": frozenset({SD15_BASELINE})},
    evaluations_per_step={
        "k_euler": 1,
        "k_heun": 2,
        "heunpp2": 3,
        "moody_sde": 1,
        "restricted_only": 1,
        "unadvised": 1,
    },
)

MODEL = ModelSpec(
    slug="deliberate",
    horde_name="Deliberate",
    baseline=SD15_BASELINE,
    native_width=512,
    native_height=512,
    anchor_clip_skip=1,
    approximate_download_size_gb=2.1,
    note="the SD1.5 model the sampler sign-off used",
)


class TestFitLine:
    """The least-squares fit that turns timing points into a slope, an intercept and a fit quality."""

    def test_a_clean_line_returns_its_own_slope_and_intercept(self) -> None:
        """Assert points lying exactly on a line recover that line's slope and intercept."""
        fit = fit_line([(10, 6.0), (20, 11.0), (30, 16.0), (40, 21.0)])

        assert fit is not None
        assert fit.slope_seconds_per_step == pytest.approx(0.5)
        assert fit.intercept_seconds == pytest.approx(1.0)
        assert fit.r_squared == pytest.approx(1.0)

    def test_scattered_points_lower_the_fit_quality_without_moving_the_slope_much(self) -> None:
        """Assert noise around a line degrades r_squared while leaving the recovered slope close."""
        fit = fit_line([(10, 6.2), (20, 10.8), (30, 16.1), (40, 20.9)])

        assert fit is not None
        assert fit.slope_seconds_per_step == pytest.approx(0.5, abs=0.02)
        assert 0.99 < fit.r_squared < 1.0

    def test_one_step_count_is_not_enough_to_separate_the_slope_from_the_overhead(self) -> None:
        """Assert a fit is refused when every point shares one step count, since no slope is derivable."""
        assert fit_line([(20, 11.0), (20, 11.4)]) is None

    def test_timings_that_do_not_move_with_steps_report_a_flat_line(self) -> None:
        """Assert constant timings fit a zero slope rather than a spuriously nonzero one."""
        fit = fit_line([(10, 5.0), (20, 5.0), (30, 5.0)])

        assert fit is not None
        assert fit.slope_seconds_per_step == pytest.approx(0.0)
        assert fit.r_squared == pytest.approx(1.0)


class TestDerivePairings:
    """How samplers are matched to a schedule to measure against, and where the policy refuses to guess."""

    def test_every_unconstrained_sampler_is_measured_on_the_default_schedule(self) -> None:
        """Assert a sampler with no rejected pairing is measured on the run's default schedule."""
        pairings = derive_pairings(samplers=["k_euler", "k_heun"], baseline=SD15_BASELINE, policy=POLICY)

        assert [pairing.scheduler for pairing in pairings] == ["karras", "karras"]
        assert {pairing.schedule_reason for pairing in pairings} == {ScheduleChoiceReason.DEFAULT}
        assert [pairing.evaluations_per_step for pairing in pairings] == [1, 2]

    def test_a_refused_default_falls_to_the_first_advised_schedule_the_policy_allows(self) -> None:
        """Assert a rejected default falls back to the sampler's first recommended, unrejected schedule."""
        (pairing,) = derive_pairings(samplers=["moody_sde"], baseline=SD15_BASELINE, policy=POLICY)

        assert pairing.scheduler == "simple"
        assert pairing.schedule_reason is ScheduleChoiceReason.SUBSTITUTED

    def test_the_order_the_samplers_were_asked_for_is_the_order_they_are_measured_in(self) -> None:
        """Assert derived pairings preserve the order the samplers were requested in."""
        pairings = derive_pairings(samplers=["k_heun", "k_euler"], baseline=SD15_BASELINE, policy=POLICY)

        assert [pairing.sampler_name for pairing in pairings] == ["k_heun", "k_euler"]

    def test_a_sampler_with_no_stated_cost_is_refused_rather_than_measured_as_free(self) -> None:
        """Assert a sampler missing from the evaluations-per-step table raises rather than defaulting."""
        with pytest.raises(ValueError, match="evaluations per step"):
            derive_pairings(samplers=["not_in_the_table"], baseline=SD15_BASELINE, policy=POLICY)

    def test_a_schedule_undefined_for_the_baseline_is_not_substituted_in(self) -> None:
        """Assert a schedule the baseline has no entry for is skipped rather than substituted."""
        policy = POLICY.model_copy(
            update={"rejected_pairings": frozenset({("restricted_only", "karras")})},
        )

        assert derive_pairings(samplers=["restricted_only"], baseline=SD15_BASELINE, policy=policy)[0].scheduler == (
            "align_your_steps"
        )
        with pytest.raises(ValueError, match="advises no schedule"):
            derive_pairings(samplers=["restricted_only"], baseline=SDXL_BASELINE, policy=policy)


class TestBuildRecords:
    """Turning raw timings into per-sampler records: their fits, their ratios and their discrepancy flag."""

    def test_a_ratio_is_the_slope_against_the_reference_samplers_slope(self) -> None:
        """Assert a sampler's ratio is its fitted slope divided by the reference sampler's slope."""
        records = build_records(
            timings={
                ("deliberate", "k_euler"): {10: [6.0], 20: [11.0], 30: [16.0]},
                ("deliberate", "k_heun"): {10: [11.0], 20: [21.0], 30: [31.0]},
            },
            pairings_by_model={
                "deliberate": derive_pairings(
                    samplers=["k_euler", "k_heun"],
                    baseline=SD15_BASELINE,
                    policy=POLICY,
                ),
            },
        )

        by_sampler = {record.sampler_name: record for record in records}
        assert by_sampler["k_euler"].ratio_vs_k_euler == pytest.approx(1.0)
        assert by_sampler["k_heun"].ratio_vs_k_euler == pytest.approx(2.0)
        assert by_sampler["k_heun"].expected_ratio == pytest.approx(2.0)
        assert not any(record.is_discrepant for record in records)

    def test_a_ratio_far_from_its_evaluation_family_is_flagged_and_nothing_is_adjusted(self) -> None:
        """Assert a measured ratio far from its expected ratio is flagged discrepant, not corrected."""
        records = build_records(
            timings={
                ("deliberate", "k_euler"): {10: [6.0], 20: [11.0]},
                ("deliberate", "k_heun"): {10: [8.0], 20: [13.0]},
            },
            pairings_by_model={
                "deliberate": derive_pairings(samplers=["k_euler", "k_heun"], baseline=SD15_BASELINE, policy=POLICY),
            },
        )

        heun = next(record for record in records if record.sampler_name == "k_heun")
        assert heun.is_discrepant
        assert heun.ratio_vs_k_euler == pytest.approx(1.0)
        assert heun.expected_ratio == pytest.approx(2.0)
        assert heun.fit is not None
        assert heun.fit.slope_seconds_per_step == pytest.approx(0.5)

    def test_a_point_is_the_median_of_its_repeats_so_one_slow_render_cannot_move_it(self) -> None:
        """Assert each fitted point is the median of its repeat timings, with the raw repeats kept too."""
        records = build_records(
            timings={("deliberate", "k_euler"): {10: [6.0, 6.1, 30.0], 20: [11.0, 11.0, 11.2]}},
            pairings_by_model={
                "deliberate": derive_pairings(samplers=["k_euler"], baseline=SD15_BASELINE, policy=POLICY),
            },
        )

        (record,) = records
        assert [point.median_seconds for point in record.points] == pytest.approx([6.1, 11.0])
        assert [point.seconds for point in record.points] == [(6.0, 6.1, 30.0), (11.0, 11.0, 11.2)]

    def test_a_sampler_with_too_few_points_reports_no_fit_rather_than_a_guess(self) -> None:
        """Assert a sampler with too few timed points reports no fit and no ratio, plus its failures."""
        records = build_records(
            timings={("deliberate", "k_euler"): {10: [6.0]}},
            pairings_by_model={
                "deliberate": derive_pairings(samplers=["k_euler"], baseline=SD15_BASELINE, policy=POLICY),
            },
            failures={("deliberate", "k_euler"): ["20 steps: the backend returned no image"]},
        )

        (record,) = records
        assert record.fit is None
        assert record.ratio_vs_k_euler is None
        assert record.is_discrepant is False
        assert record.failed_renders == ("20 steps: the backend returned no image",)


class TestMeasurementJob:
    """The render job a measurement pairing turns into, and how little of it a new pairing changes."""

    def test_only_the_sampler_the_schedule_and_the_steps_differ_between_two_jobs(self) -> None:
        """Assert two jobs for different pairings differ only in sampler name and steps."""
        pairings = derive_pairings(samplers=["k_euler", "k_heun"], baseline=SD15_BASELINE, policy=POLICY)
        first = measurement_job(
            image_model=MODEL,
            pairing=pairings[0],
            steps=10,
            seed=210000,
            cfg_scale=7.0,
            stop_index=1,
        )
        second = measurement_job(
            image_model=MODEL,
            pairing=pairings[1],
            steps=40,
            seed=210000,
            cfg_scale=7.0,
            stop_index=2,
        )

        differing = first.generation_parameters().differing_field_names(second.generation_parameters())
        assert differing == {"sampler_name", "steps"}
        assert first.width == 512
        assert first.height == 512
        assert first.karras is True
        assert first.features().is_plain


class TestReportArtifact:
    """The report as it is written to disk, and what a reader of that file can rely on finding in it."""

    def test_the_written_artifact_carries_its_schema_version_and_reloads(self, tmp_path: Path) -> None:
        """Assert a written report lands under measurements/, states its schema version and round-trips."""
        report = _report()

        report_path = write_report(report=report, output_root=tmp_path)

        assert report_path.parent == tmp_path / "measurements"
        written = json.loads(report_path.read_text(encoding="utf-8"))
        assert written["schema_version"] == MEASUREMENT_SCHEMA_VERSION
        assert SamplerCostReport.model_validate(written) == report

    def test_every_record_states_the_schedule_it_was_measured_on_and_why(self, tmp_path: Path) -> None:
        """Assert every written record states its scheduler, schedule reason, cost and ratio fields."""
        written = json.loads(write_report(report=_report(), output_root=tmp_path).read_text(encoding="utf-8"))

        for record in written["records"]:
            assert record["scheduler"]
            assert record["schedule_reason"] in {"default", "substituted"}
            assert record["evaluations_per_step"] >= 1
            assert "ratio_vs_k_euler" in record
            assert "is_discrepant" in record

    def test_the_summary_table_names_every_record_and_counts_the_discrepancies(self) -> None:
        """Assert the summary table names every sampler and states how many records are discrepant."""
        table = format_summary_table(_report())

        assert "k_euler" in table
        assert "k_heun" in table
        assert "1 discrepant" in table


def _report() -> SamplerCostReport:
    """Return a report built from stated timings, with one honest discrepancy in it."""
    pairings = derive_pairings(samplers=["k_euler", "k_heun"], baseline=SD15_BASELINE, policy=POLICY)
    records = build_records(
        timings={
            ("deliberate", "k_euler"): {10: [6.0], 20: [11.0]},
            ("deliberate", "k_heun"): {10: [8.0], 20: [13.0]},
        },
        pairings_by_model={"deliberate": pairings},
    )
    return SamplerCostReport(
        generated_at="2026-08-03T12-00-00Z",
        runner_name="local:cuda",
        settings=MeasurementSettings(
            seed=210000,
            cfg_scale=7.0,
            prompt="a prompt held constant",
            negative_prompt="a negative prompt held constant",
            step_counts=(10, 20),
            repeats=1,
            warmup_renders=1,
            warmup_steps=10,
            reference_sampler="k_euler",
            default_scheduler="karras",
            discrepancy_tolerance=0.2,
        ),
        models=(
            MeasuredModel(
                slug=MODEL.slug,
                horde_name=MODEL.horde_name,
                baseline=MODEL.baseline,
                width=MODEL.native_width,
                height=MODEL.native_height,
                clip_skip=MODEL.anchor_clip_skip,
            ),
        ),
        records=tuple(records),
    )
