# SPDX-FileCopyrightText: 2026 Tazlin
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Measure what a step of each sampler costs, through the same pipeline the sweeps render on.

A cost figure taken from a single render is three different numbers added together: the model
load, the fixed per-job overhead every render pays whatever the sampler, and the sampling itself.
This mode separates them by rendering the same job at several step counts and fitting a line
through the results. The slope is what a step costs, the intercept is everything that does not
scale with steps, and the ratio of one sampler's slope to `k_euler`'s is the figure a cost model
wants.

The measurement is arranged so nothing but the sampler differs: one seed, one prompt, one
resolution and one cfg scale throughout, warmup renders discarded after every model load, and
repeats reduced by median rather than mean so a single interrupted render cannot move the point.

Critical public members:

- [`measure_sampler_cost`][param_sweeps.sampler_cost.measure_sampler_cost]: run the measurement.
- [`SamplerCostReport`][param_sweeps.sampler_cost.SamplerCostReport]: the artifact it writes.
- [`derive_pairings`][param_sweeps.sampler_cost.derive_pairings]: which schedule each sampler is
  measured on, and why that one.
- [`fit_line`][param_sweeps.sampler_cost.fit_line]: the ordinary least squares fit the slope and
  intercept come from.
- [`SchedulePolicy`][param_sweeps.sampler_cost.SchedulePolicy]: the sampler facts the derivation
  reads, kept as plain data so it can be stated in a test as well as read from the shared table.
"""

from __future__ import annotations

import logging
import statistics
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from strenum import StrEnum

from param_sweeps.runner import SweepRunner
from param_sweeps.spec import ModelSpec, RenderJob, SweepParameter

logger = logging.getLogger(__name__)

MEASUREMENT_SCHEMA_VERSION = 1
"""The layout of the JSON artifact this module writes."""

MEASUREMENT_DIRECTORY = "measurements"
"""Where under the output root a measurement lands, beside the rendered library rather than in it."""

REFERENCE_SAMPLER = "k_euler"
"""The sampler every ratio is expressed against: one model evaluation per step, no correction."""

DEFAULT_SCHEDULER = "karras"
"""The schedule every sampler is measured on unless it is constrained away from it."""

DEFAULT_STEP_COUNTS: tuple[int, ...] = (10, 20, 30, 40)
"""The step counts the line is fitted through."""

DEFAULT_REPEATS = 3
"""How many times each point is rendered before its median is taken."""

DEFAULT_WARMUP_RENDERS = 1
"""How many discarded renders each model gets before anything is timed."""

WARMUP_STEPS = 10
"""The step count warmup renders use: enough to fault the whole path in, cheap enough to waste."""

DISCREPANCY_TOLERANCE = 0.20
"""How far a measured ratio may sit from its evaluations-per-step family before it is flagged."""

MEASUREMENT_PROMPT = (
    "a woman in her 30s at the wooden railing of an airship gondola at sunset, brass fittings,"
    " taut rigging, a sea of clouds below, photograph, 35mm, natural light, sharp focus, highly"
    " detailed, realistic skin texture"
)
"""The prompt every timed render uses, carried over from the sampler sign-off batch.

Held identical across the whole measurement. What it depicts does not matter to a timing; that it
never changes does, because prompt length moves the conditioning cost that lands in the intercept.
"""

MEASUREMENT_NEGATIVE_PROMPT = "cartoon, illustration, painting, lowres, blurry, deformed hands, extra fingers"
"""The negative prompt every timed render uses, from the same batch and held identical for the same reason."""


class ScheduleChoiceReason(StrEnum):
    """Why a sampler is measured on the schedule it is measured on."""

    DEFAULT = "default"
    """The default schedule, which is what nearly every sampler is measured on."""

    SUBSTITUTED = "substituted"
    """The default pairing is one the shared table refuses, so a compatible schedule stands in."""


class SchedulePolicy(BaseModel):
    """Represents the sampler facts a measurement plan is derived from.

    This is plain data rather than a call into the shared constraints table so that the derivation
    can be exercised against a stated policy, and so a table that grows a sampler cannot change
    what an already-written test asserts.

    Attributes:
        rejected_pairings: Sampler and schedule pairs that produce no usable image.
        recommended_schedulers: The schedules advised for a sampler, best first, for the samplers
            an advisory names. A sampler absent from the mapping has no advice.
        scheduler_baselines: The baselines a restricted schedule is defined for. A schedule absent
            from the mapping works on every baseline.
        evaluations_per_step: How many model evaluations one step of each sampler performs.
    """

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    rejected_pairings: frozenset[tuple[str, str]] = frozenset()
    recommended_schedulers: Mapping[str, tuple[str, ...]] = Field(default_factory=dict)
    scheduler_baselines: Mapping[str, frozenset[str]] = Field(default_factory=dict)
    evaluations_per_step: Mapping[str, int] = Field(default_factory=dict)

    def allows(self, *, sampler: str, scheduler: str, baseline: str) -> bool:
        """Return whether this sampler may be measured on this schedule against this baseline."""
        if (sampler, scheduler) in self.rejected_pairings:
            return False

        defined_baselines = self.scheduler_baselines.get(scheduler)
        return defined_baselines is None or baseline in defined_baselines


class SamplerSchedulePairing(BaseModel):
    """Represents one sampler, the schedule it is measured on, and what a step of it should cost."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    sampler_name: str
    scheduler: str
    schedule_reason: ScheduleChoiceReason
    evaluations_per_step: int


class LinearFit(BaseModel):
    """Represents an ordinary least squares fit of render time against step count."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    slope_seconds_per_step: float
    intercept_seconds: float
    r_squared: float


class StepPoint(BaseModel):
    """Represents every timing taken at one step count, and the median the fit uses."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    steps: int
    seconds: tuple[float, ...] = Field(min_length=1)
    median_seconds: float


class SamplerCostRecord(BaseModel):
    """Represents what one sampler cost on one model, and how that compares to its family."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    sampler_name: str
    model_slug: str
    scheduler: str
    schedule_reason: ScheduleChoiceReason
    evaluations_per_step: int
    points: tuple[StepPoint, ...] = ()
    fit: LinearFit | None = None
    ratio_vs_k_euler: float | None = None
    expected_ratio: float | None = None
    """What the ratio should be if a step costs its evaluations, relative to the reference sampler."""

    is_discrepant: bool = False
    """Whether the measured ratio sits further from the expected one than the tolerance allows."""

    failed_renders: tuple[str, ...] = ()
    """One entry per render that raised, so a thin point is visible rather than merely narrow."""


class MeasurementSettings(BaseModel):
    """Represents everything held constant across a measurement, as it was actually run."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    seed: int
    cfg_scale: float
    prompt: str
    negative_prompt: str
    step_counts: tuple[int, ...]
    repeats: int
    warmup_renders: int
    warmup_steps: int
    reference_sampler: str
    default_scheduler: str
    discrepancy_tolerance: float


class MeasuredModel(BaseModel):
    """Represents one model a measurement covered, at the geometry it was measured at."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    slug: str
    horde_name: str
    baseline: str
    width: int
    height: int
    clip_skip: int


class SamplerCostReport(BaseModel):
    """Represents a whole measurement: what was held constant, what was run, and what it cost."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    schema_version: int = MEASUREMENT_SCHEMA_VERSION
    generated_at: str
    runner_name: str
    settings: MeasurementSettings
    models: tuple[MeasuredModel, ...]
    records: tuple[SamplerCostRecord, ...]

    @property
    def discrepancies(self) -> tuple[SamplerCostRecord, ...]:
        """Return the records whose measured ratio disagrees with their evaluations per step."""
        return tuple(record for record in self.records if record.is_discrepant)


def horde_sdk_schedule_policy() -> SchedulePolicy:
    """Return the schedule policy the shared constraints table states.

    The vocabulary and the constraints are read from `horde_sdk` rather than restated here, so a
    sampler added there is measured here without this module being touched.

    Returns:
        The policy, covering every sampler the installed vocabulary knows.

    Raises:
        RuntimeError: If the installed `horde_sdk` predates the constraints table, which is the
            one thing that cannot be worked around from here.
    """
    try:
        from horde_sdk.generation_parameters.image.constraints import (
            REJECTED_SAMPLER_SCHEDULER_PAIRINGS,
            SAMPLER_CONSTRAINTS,
            SAMPLER_RECOMMENDATIONS,
            SCHEDULER_BASELINE_APPLICABILITY,
        )
    except ImportError as missing_table:
        raise RuntimeError(
            "the installed horde_sdk has no image sampler constraints table, so the sampler"
            " vocabulary and the schedule pairings cannot be read from it; install the horde_sdk"
            " that carries horde_sdk.generation_parameters.image.constraints",
        ) from missing_table

    recommended: dict[str, list[str]] = {}
    for recommendation in SAMPLER_RECOMMENDATIONS:
        for sampler in recommendation.samplers:
            for scheduler in recommendation.schedulers:
                advised = recommended.setdefault(str(sampler), [])
                if str(scheduler) not in advised:
                    advised.append(str(scheduler))

    return SchedulePolicy(
        rejected_pairings=frozenset(
            (str(sampler), str(scheduler)) for sampler, scheduler in REJECTED_SAMPLER_SCHEDULER_PAIRINGS
        ),
        recommended_schedulers={sampler: tuple(schedulers) for sampler, schedulers in recommended.items()},
        scheduler_baselines={
            str(scheduler): frozenset(str(baseline) for baseline in baselines)
            for scheduler, baselines in SCHEDULER_BASELINE_APPLICABILITY.items()
        },
        evaluations_per_step={
            str(sampler): constraints.evaluations_per_step for sampler, constraints in SAMPLER_CONSTRAINTS.items()
        },
    )


def horde_sdk_samplers() -> tuple[str, ...]:
    """Return every sampler the installed vocabulary knows, in its own order.

    Raises:
        RuntimeError: If the installed `horde_sdk` has no image sampler vocabulary.
    """
    try:
        from horde_sdk.generation_parameters.image.consts import KNOWN_IMAGE_SAMPLERS
    except ImportError as missing_vocabulary:
        raise RuntimeError(
            "the installed horde_sdk has no image sampler vocabulary to measure",
        ) from missing_vocabulary

    return tuple(str(sampler) for sampler in KNOWN_IMAGE_SAMPLERS)


def derive_pairings(
    *,
    samplers: Sequence[str],
    baseline: str,
    policy: SchedulePolicy,
    default_scheduler: str = DEFAULT_SCHEDULER,
) -> list[SamplerSchedulePairing]:
    """Return the schedule each sampler is measured on against this baseline, and why.

    Every sampler is measured on the default schedule, because a measurement whose schedule moves
    with the sampler measures two things at once. A sampler the policy refuses that pairing for
    takes the first schedule it advises that the policy does allow, and says so in the pairing, so
    a substitution is visible in the artifact rather than folded into the number.

    Args:
        samplers: The samplers to measure, in the order they should be run.
        baseline: The baseline of the model they will be measured on.
        policy: The sampler facts to derive from.
        default_scheduler: The schedule to use wherever it is allowed.

    Returns:
        One pairing per sampler, in the order given.

    Raises:
        ValueError: If a sampler is allowed no schedule at all on this baseline, or the policy
            states no evaluations per step for it.
    """
    pairings: list[SamplerSchedulePairing] = []
    for sampler in samplers:
        evaluations = policy.evaluations_per_step.get(sampler)
        if evaluations is None:
            raise ValueError(f"the policy states no evaluations per step for {sampler!r}")

        if policy.allows(sampler=sampler, scheduler=default_scheduler, baseline=baseline):
            scheduler = default_scheduler
            reason = ScheduleChoiceReason.DEFAULT
        else:
            scheduler = _substitute_scheduler(sampler=sampler, baseline=baseline, policy=policy)
            reason = ScheduleChoiceReason.SUBSTITUTED

        pairings.append(
            SamplerSchedulePairing(
                sampler_name=sampler,
                scheduler=scheduler,
                schedule_reason=reason,
                evaluations_per_step=evaluations,
            ),
        )

    return pairings


def _substitute_scheduler(*, sampler: str, baseline: str, policy: SchedulePolicy) -> str:
    """Return the schedule to measure a sampler on when the default is refused.

    Raises:
        ValueError: If the policy advises nothing this sampler and baseline can use.
    """
    for candidate in policy.recommended_schedulers.get(sampler, ()):
        if policy.allows(sampler=sampler, scheduler=candidate, baseline=baseline):
            return candidate

    raise ValueError(
        f"{sampler!r} is refused the {DEFAULT_SCHEDULER} schedule on {baseline} and the policy advises"
        " no schedule it can be measured on instead",
    )


def fit_line(points: Sequence[tuple[int, float]]) -> LinearFit | None:
    """Return the ordinary least squares fit of seconds against steps, or None when it has no meaning.

    Args:
        points: One (steps, seconds) pair per step count, in any order.

    Returns:
        The fit, or None when fewer than two distinct step counts were measured, which is the case
        the slope is undefined in.
    """
    distinct_step_counts = {steps for steps, _ in points}
    if len(distinct_step_counts) < 2:
        return None

    step_values = [float(steps) for steps, _ in points]
    second_values = [seconds for _, seconds in points]
    mean_steps = statistics.fmean(step_values)
    mean_seconds = statistics.fmean(second_values)

    covariance = sum(
        (steps - mean_steps) * (seconds - mean_seconds)
        for steps, seconds in zip(step_values, second_values, strict=True)
    )
    step_variance = sum((steps - mean_steps) ** 2 for steps in step_values)
    slope = covariance / step_variance
    intercept = mean_seconds - slope * mean_steps

    residual_sum = sum(
        (seconds - (slope * steps + intercept)) ** 2 for steps, seconds in zip(step_values, second_values, strict=True)
    )
    total_sum = sum((seconds - mean_seconds) ** 2 for seconds in second_values)
    # A perfectly flat set of timings has nothing for the line to explain: the fit is exact when it
    # reproduces them and worthless otherwise, and the usual ratio would divide by zero either way.
    r_squared = (1.0 if residual_sum == 0 else 0.0) if total_sum == 0 else 1.0 - residual_sum / total_sum

    return LinearFit(slope_seconds_per_step=slope, intercept_seconds=intercept, r_squared=r_squared)


def build_records(
    *,
    timings: Mapping[tuple[str, str], Mapping[int, Sequence[float]]],
    pairings_by_model: Mapping[str, Sequence[SamplerSchedulePairing]],
    failures: Mapping[tuple[str, str], Sequence[str]] | None = None,
    reference_sampler: str = REFERENCE_SAMPLER,
    tolerance: float = DISCREPANCY_TOLERANCE,
) -> list[SamplerCostRecord]:
    """Return one record per sampler and model, fitted, compared and flagged.

    Args:
        timings: Every timing taken, keyed by (model slug, sampler) and then by step count.
        pairings_by_model: The pairings each model was measured with, in run order.
        failures: The renders that raised, keyed the same way as the timings.
        reference_sampler: The sampler every ratio is expressed against.
        tolerance: How far a ratio may sit from its evaluations-per-step family before it is flagged.

    Returns:
        The records, model by model in the order the pairings were given.
    """
    resolved_failures = failures if failures is not None else {}
    records: list[SamplerCostRecord] = []

    for model_slug, pairings in pairings_by_model.items():
        fits = {
            pairing.sampler_name: fit_line(_median_points(timings.get((model_slug, pairing.sampler_name), {})))
            for pairing in pairings
        }
        reference_fit = fits.get(reference_sampler)
        reference_evaluations = next(
            (pairing.evaluations_per_step for pairing in pairings if pairing.sampler_name == reference_sampler),
            None,
        )

        for pairing in pairings:
            measured = timings.get((model_slug, pairing.sampler_name), {})
            fit = fits[pairing.sampler_name]
            ratio = _ratio(fit=fit, reference_fit=reference_fit)
            expected = (
                pairing.evaluations_per_step / reference_evaluations
                if reference_evaluations is not None and reference_evaluations > 0
                else None
            )
            records.append(
                SamplerCostRecord(
                    sampler_name=pairing.sampler_name,
                    model_slug=model_slug,
                    scheduler=pairing.scheduler,
                    schedule_reason=pairing.schedule_reason,
                    evaluations_per_step=pairing.evaluations_per_step,
                    points=tuple(
                        StepPoint(
                            steps=steps,
                            seconds=tuple(seconds),
                            median_seconds=statistics.median(seconds),
                        )
                        for steps, seconds in sorted(measured.items())
                        if seconds
                    ),
                    fit=fit,
                    ratio_vs_k_euler=ratio,
                    expected_ratio=expected,
                    is_discrepant=_is_discrepant(ratio=ratio, expected=expected, tolerance=tolerance),
                    failed_renders=tuple(resolved_failures.get((model_slug, pairing.sampler_name), ())),
                ),
            )

    return records


def _median_points(measured: Mapping[int, Sequence[float]]) -> list[tuple[int, float]]:
    """Return one (steps, median seconds) pair per step count that produced a timing."""
    return [(steps, statistics.median(seconds)) for steps, seconds in sorted(measured.items()) if seconds]


def _ratio(*, fit: LinearFit | None, reference_fit: LinearFit | None) -> float | None:
    """Return a slope as a multiple of the reference slope, or None when either is unusable."""
    if fit is None or reference_fit is None or reference_fit.slope_seconds_per_step <= 0:
        return None

    return fit.slope_seconds_per_step / reference_fit.slope_seconds_per_step


def _is_discrepant(*, ratio: float | None, expected: float | None, tolerance: float) -> bool:
    """Return whether a measured ratio sits further from its expected one than the tolerance allows."""
    if ratio is None or expected is None or expected <= 0:
        return False

    return abs(ratio / expected - 1.0) > tolerance


def format_summary_table(report: SamplerCostReport) -> str:
    """Return the compact table printed when a measurement finishes.

    One line per sampler and model, in the order the records were produced, plus a closing count of
    the ratios that disagree with their evaluations-per-step family.
    """
    header = (
        f"{'sampler':<32}{'model':<16}{'sched':<10}{'sec/step':>10}{'fixed':>9}"
        f"{'r2':>7}{'ratio':>8}{'expect':>8}{'eval':>6}  flag"
    )
    lines = [header, "-" * len(header)]

    for record in report.records:
        fit = record.fit
        lines.append(
            f"{record.sampler_name:<32}{record.model_slug:<16}{record.scheduler:<10}"
            f"{_number(fit.slope_seconds_per_step if fit else None, 4):>10}"
            f"{_number(fit.intercept_seconds if fit else None, 2):>9}"
            f"{_number(fit.r_squared if fit else None, 3):>7}"
            f"{_number(record.ratio_vs_k_euler, 2):>8}"
            f"{_number(record.expected_ratio, 2):>8}"
            f"{record.evaluations_per_step:>6}  "
            f"{'DISCREPANT' if record.is_discrepant else ''}",
        )

    failed_render_count = sum(len(record.failed_renders) for record in report.records)
    lines.append("")
    lines.append(
        f"{len(report.records)} record(s), {len(report.discrepancies)} discrepant,"
        f" {failed_render_count} failed render(s).",
    )
    return "\n".join(lines)


def _number(value: float | None, places: int) -> str:
    """Return a number for the table, or a dash when there is none."""
    return "-" if value is None else f"{value:.{places}f}"


def measurement_job(
    *,
    image_model: ModelSpec,
    pairing: SamplerSchedulePairing,
    steps: int,
    seed: int,
    cfg_scale: float,
    stop_index: int,
) -> RenderJob:
    """Return the render job one timing is taken from.

    Every field except the sampler, the schedule and the step count is the same for every job in a
    measurement, which is what makes the timings comparable. The job renders at the model's native
    size, because that is the size the horde prices these models at.
    """
    return RenderJob(
        parameter=SweepParameter.SAMPLER,
        model_slug=image_model.slug,
        horde_model_name=image_model.horde_name,
        scene_id="sampler-cost",
        prompt=MEASUREMENT_PROMPT,
        negative_prompt=MEASUREMENT_NEGATIVE_PROMPT,
        stop_index=stop_index,
        stop_label=f"{pairing.sampler_name}_{steps:03d}",
        stop_value=pairing.sampler_name,
        caption=f"{pairing.sampler_name} on {pairing.scheduler} at {steps} steps",
        width=image_model.native_width,
        height=image_model.native_height,
        steps=steps,
        cfg_scale=cfg_scale,
        sampler_name=pairing.sampler_name,
        clip_skip=image_model.anchor_clip_skip,
        hires_fix=False,
        tiling=False,
        karras=pairing.scheduler == DEFAULT_SCHEDULER,
        denoising_strength=1.0,
        seed=seed,
        scheduler=pairing.scheduler,
        directory_override=f"{MEASUREMENT_DIRECTORY}/{image_model.slug}",
    )


def measure_sampler_cost(
    *,
    runner: SweepRunner,
    models: Sequence[ModelSpec],
    policy: SchedulePolicy,
    samplers: Sequence[str],
    seed: int,
    cfg_scale: float,
    step_counts: Sequence[int] = DEFAULT_STEP_COUNTS,
    repeats: int = DEFAULT_REPEATS,
    warmup_renders: int = DEFAULT_WARMUP_RENDERS,
) -> SamplerCostReport:
    """Render the measurement and return what it cost, without writing an image anywhere.

    The runner is prepared once with every model, then the models are measured one after another so
    the weights are loaded as few times as possible. Each model's timed renders are preceded by
    discarded warmup renders, because the first render after a model reaches the device pays for
    the graph being built and would otherwise land in the intercept.

    Args:
        runner: The backend to render through, unprepared.
        models: The models to measure, each at its own native size.
        policy: The sampler facts the schedule pairings are derived from.
        samplers: The samplers to measure, in run order.
        seed: The seed every render uses.
        cfg_scale: The cfg scale every render uses.
        step_counts: The step counts the line is fitted through.
        repeats: How many timings each point is the median of.
        warmup_renders: How many discarded renders each model gets first.

    Returns:
        The finished report, including any render that failed.

    Raises:
        ValueError: If no model or no sampler was given, or a sampler can be measured on no
            schedule at all.

    Side Effects:
        Renders on the accelerator, and on the local runner may download model weights.
    """
    if not models:
        raise ValueError("a measurement needs at least one model")
    if not samplers:
        raise ValueError("a measurement needs at least one sampler")

    pairings_by_model = {
        image_model.slug: derive_pairings(samplers=samplers, baseline=image_model.baseline, policy=policy)
        for image_model in models
    }
    timings: dict[tuple[str, str], dict[int, list[float]]] = {}
    failures: dict[tuple[str, str], list[str]] = {}

    started_at = datetime.now(UTC)
    try:
        runner.prepare(
            [
                measurement_job(
                    image_model=image_model,
                    pairing=pairings_by_model[image_model.slug][0],
                    steps=min(step_counts),
                    seed=seed,
                    cfg_scale=cfg_scale,
                    stop_index=0,
                )
                for image_model in models
            ],
            models,
        )

        for image_model in models:
            _run_warmups(
                runner=runner,
                image_model=image_model,
                pairing=pairings_by_model[image_model.slug][0],
                seed=seed,
                cfg_scale=cfg_scale,
                warmup_renders=warmup_renders,
            )
            _measure_model(
                runner=runner,
                image_model=image_model,
                pairings=pairings_by_model[image_model.slug],
                step_counts=step_counts,
                repeats=repeats,
                seed=seed,
                cfg_scale=cfg_scale,
                timings=timings,
                failures=failures,
            )
    finally:
        runner.close()

    return SamplerCostReport(
        generated_at=started_at.isoformat(),
        runner_name=runner.name,
        settings=MeasurementSettings(
            seed=seed,
            cfg_scale=cfg_scale,
            prompt=MEASUREMENT_PROMPT,
            negative_prompt=MEASUREMENT_NEGATIVE_PROMPT,
            step_counts=tuple(step_counts),
            repeats=repeats,
            warmup_renders=warmup_renders,
            warmup_steps=WARMUP_STEPS,
            reference_sampler=REFERENCE_SAMPLER,
            default_scheduler=DEFAULT_SCHEDULER,
            discrepancy_tolerance=DISCREPANCY_TOLERANCE,
        ),
        models=tuple(
            MeasuredModel(
                slug=image_model.slug,
                horde_name=image_model.horde_name,
                baseline=image_model.baseline,
                width=image_model.native_width,
                height=image_model.native_height,
                clip_skip=image_model.anchor_clip_skip,
            )
            for image_model in models
        ),
        records=tuple(
            build_records(timings=timings, pairings_by_model=pairings_by_model, failures=failures),
        ),
    )


def _run_warmups(
    *,
    runner: SweepRunner,
    image_model: ModelSpec,
    pairing: SamplerSchedulePairing,
    seed: int,
    cfg_scale: float,
    warmup_renders: int,
) -> None:
    """Render and discard this model's warmups, so the first timed render is not the first render."""
    for warmup_number in range(warmup_renders):
        job = measurement_job(
            image_model=image_model,
            pairing=pairing,
            steps=WARMUP_STEPS,
            seed=seed,
            cfg_scale=cfg_scale,
            stop_index=0,
        )
        logger.info("Warmup render: model=%s number=%s", image_model.slug, warmup_number + 1)
        started = time.perf_counter()
        runner.render(job)
        logger.info("Warmup done: model=%s seconds=%.2f", image_model.slug, time.perf_counter() - started)


def _measure_model(
    *,
    runner: SweepRunner,
    image_model: ModelSpec,
    pairings: Sequence[SamplerSchedulePairing],
    step_counts: Sequence[int],
    repeats: int,
    seed: int,
    cfg_scale: float,
    timings: dict[tuple[str, str], dict[int, list[float]]],
    failures: dict[tuple[str, str], list[str]],
) -> None:
    """Time every sampler, step count and repeat for one model, recording failures as failures.

    A render that raises costs its repeat and nothing else: the point keeps the timings it did get,
    the failure is carried into the record, and the rest of the measurement goes on.
    """
    total = len(pairings) * len(step_counts) * repeats
    render_number = 0

    for stop_index, pairing in enumerate(pairings, start=1):
        for steps in step_counts:
            for repeat_number in range(repeats):
                render_number += 1
                job = measurement_job(
                    image_model=image_model,
                    pairing=pairing,
                    steps=steps,
                    seed=seed,
                    cfg_scale=cfg_scale,
                    stop_index=stop_index,
                )
                print(
                    f"[{image_model.slug} {render_number}/{total}] {pairing.sampler_name}"
                    f" on {pairing.scheduler} at {steps} steps, repeat {repeat_number + 1}/{repeats}",
                )
                started = time.perf_counter()
                try:
                    runner.render(job)
                except RuntimeError as render_error:
                    logger.error(
                        "Timed render failed: model=%s sampler=%s steps=%s error=%s",
                        image_model.slug,
                        pairing.sampler_name,
                        steps,
                        render_error,
                    )
                    failures.setdefault((image_model.slug, pairing.sampler_name), []).append(
                        f"{steps} steps: {render_error}",
                    )
                    continue

                elapsed = time.perf_counter() - started
                timings.setdefault((image_model.slug, pairing.sampler_name), {}).setdefault(steps, []).append(elapsed)
                print(f"    {elapsed:.2f}s")


def write_report(*, report: SamplerCostReport, output_root: Path) -> Path:
    """Write the measurement artifact under the output root, and return where it landed.

    The filename carries the run's start time, so a re-measurement never overwrites the one it is
    meant to be compared against.

    Side Effects:
        Creates the measurement directory and writes one JSON file into it.
    """
    directory = output_root / MEASUREMENT_DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    stamp = report.generated_at.replace(":", "-").replace("+00-00", "Z")
    report_path = directory / f"sampler-cost-{stamp}.json"
    report_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return report_path
