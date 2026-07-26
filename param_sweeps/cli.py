# SPDX-FileCopyrightText: 2026 Tazlin
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The command line: plan a campaign, render it, rebuild the index, and publish the result.

Every rendering subcommand takes the same specification and the same filters, so "render the
whole library" and "regenerate one strip" are the same command with the filters narrowed. The
publishing subcommands (``manifest``, ``upload``, ``verify``) work from ``index.json`` alone and
need no specification at all; the two of them that reach a bucket also need the tool's optional
publish extra, and say so plainly when it is absent rather than failing on an import. ``qr-check``
works from the rendered tree in the same way, and is the gate that every QR image the library and
the tutorials publish still decodes to the payload it was asked to carry. ``flows`` is
the odd one out: it reads its own specification and writes the frontpage's tutorial asset tree
rather than the sweep library. ``alchemy`` reads its own specification too, and writes both: the
evidence tree the alchemy pages read, and the one strip in the sweep library whose stops are
upscalers rather than a generation parameter. ``review`` reads
nothing but the specification and what is already on disk; ``choose`` closes that loop by
writing the picks a reviewer saved from the sheet back into the specification. Nothing here
ever prints the API key or the bucket's secret.

Critical public member: [`main`][param_sweeps.cli.main], the console entry point.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

from strenum import StrEnum

from param_sweeps.alchemy import (
    ANNOTATION_FORM,
    BETA_MODEL_CATEGORIES,
    AlchemyBackend,
    AlchemyResult,
    AlchemyUseCase,
    HordelibAlchemyBackend,
    LiveHordeAlchemyBackend,
    known_alchemy_forms,
    known_annotation_control_types,
    load_alchemy_spec,
    run_alchemy,
    use_case_counts,
)
from param_sweeps.campaign import CampaignResult, JobFilter, run_campaign
from param_sweeps.flows import (
    FlowResult,
    FlowsSpec,
    FlowStatus,
    HordeFlowGenerator,
    LocalFlowGenerator,
    load_flows_spec,
    run_flows,
    write_manifest,
)
from param_sweeps.horde_client import HordeApiClient
from param_sweeps.horde_runner import HordeApiRunner, MissingApiKeyError, default_env_file, read_api_key
from param_sweeps.image_utilities import DEFAULT_SERVICE_URL, HordeImageUtilitiesBackend, resolve_service_url
from param_sweeps.local_runner import GpuBackend, LocalRunner, resolve_gpu_backend
from param_sweeps.outputs import write_index
from param_sweeps.publish import (
    IncompleteLibraryError,
    MissingPublishDependencyError,
    MissingS3ConfigurationError,
    S3Settings,
    SweepLibraryBucket,
    SyncPlan,
    apply_sync_plan,
    bucket_transport_errors,
    enumerate_library,
    format_byte_size,
    load_s3_settings,
    object_key,
    plan_sync,
    verify_local,
    verify_remote,
)
from param_sweeps.qr_decode import check_library
from param_sweeps.review import apply_choices, write_contact_sheet
from param_sweeps.runner import SweepRunner
from param_sweeps.sampler_cost import (
    DEFAULT_REPEATS,
    DEFAULT_STEP_COUNTS,
    DEFAULT_WARMUP_RENDERS,
    derive_pairings,
    format_summary_table,
    horde_sdk_samplers,
    horde_sdk_schedule_policy,
    measure_sampler_cost,
    write_report,
)
from param_sweeps.spec import TOOL_ROOT, ModelSpec, SweepParameter, SweepSpec, load_spec
from param_sweeps.step_preview import run_step_preview

logger = logging.getLogger(__name__)

DEFAULT_SPEC_PATH = TOOL_ROOT / "sweep_spec.json"
DEFAULT_OUTPUT_ROOT = TOOL_ROOT.parents[1] / "assets" / "param-sweeps"
"""``<haidra-assets>/assets/param-sweeps``: the library's home in this repository."""

DEFAULT_SAMPLER_COST_SPEC_PATH = TOOL_ROOT / "signoff_solver_knobs_spec.json"
"""The specification the sampler measurement takes its models, geometry, seed and cfg scale from."""

DEFAULT_SAMPLER_COST_MODEL_SLUGS = ("deliberate", "sdxl-1-0")
"""One SD1.5 model and one SDXL model, the pair the sampler sign-off batch was rendered with."""

DEFAULT_ALCHEMY_SPEC_PATH = TOOL_ROOT / "alchemy_spec.json"
"""The specification placing every alchemy form the shared vocabulary names into a use case."""

DEFAULT_FLOWS_SPEC_PATH = TOOL_ROOT / "flows_spec.json"
DEFAULT_FLOWS_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT / "tutorials"
"""The evidence library's tutorial tree, which is the consumer of the flows this tool generates."""

PUBLISHING_COMMANDS = frozenset({"manifest", "upload", "verify"})
"""The subcommands that work from the rendered library and never read the specification."""


class AlchemyBackendChoice(StrEnum):
    """Where the ``alchemy`` subcommand applies its forms.

    The three are not interchangeable: the post-processors run locally, the describe family only
    runs on the horde, and the control maps only run on a horde-image-utilities service, so a full
    campaign is three runs restricted to the forms each can answer.
    """

    HORDELIB = "hordelib"
    API = "api"
    IMAGE_UTILITIES = "hiu"


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for every subcommand."""
    parser = argparse.ArgumentParser(
        prog="param-sweeps",
        description="Render the parameter-sweep illustration library from its specification.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="print the fetch plan and image count, and generate nothing")
    _add_common_arguments(plan_parser)
    _add_local_arguments(plan_parser)

    local_parser = subparsers.add_parser("local", help="render through horde-engine on this machine's GPU")
    _add_common_arguments(local_parser)
    _add_local_arguments(local_parser)

    horde_parser = subparsers.add_parser("horde", help="render through the public AI Horde API")
    _add_common_arguments(horde_parser)
    horde_parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="the .env holding AIHORDE_DEV_KEY (default: the frontpage repository's, if present)",
    )
    horde_parser.add_argument("--api-url", default=None, help="the API root to use instead of the public horde")

    sampler_cost_parser = subparsers.add_parser(
        "sampler-cost",
        help="measure what a step of each sampler costs, and write the measurement artifact",
    )
    sampler_cost_parser.add_argument(
        "--spec",
        type=Path,
        default=DEFAULT_SAMPLER_COST_SPEC_PATH,
        help="the specification the models, geometry, seed and cfg scale come from",
    )
    sampler_cost_parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="the output tree's root; the artifact lands in its measurements directory",
    )
    sampler_cost_parser.add_argument(
        "--model",
        action="append",
        default=[],
        help=(
            f"restrict the run to this model slug (repeatable; default: {', '.join(DEFAULT_SAMPLER_COST_MODEL_SLUGS)})"
        ),
    )
    sampler_cost_parser.add_argument(
        "--sampler",
        action="append",
        default=[],
        help="restrict the run to this sampler (repeatable; default: the whole known vocabulary)",
    )
    sampler_cost_parser.add_argument(
        "--steps",
        action="append",
        type=int,
        default=[],
        help=f"a step count to fit through (repeatable; default: {', '.join(str(x) for x in DEFAULT_STEP_COUNTS)})",
    )
    sampler_cost_parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help="how many renders each point is the median of",
    )
    sampler_cost_parser.add_argument(
        "--warmups",
        type=int,
        default=DEFAULT_WARMUP_RENDERS,
        help="how many discarded renders each model gets before anything is timed",
    )
    sampler_cost_parser.add_argument(
        "--plan-only",
        action="store_true",
        help="print the samplers, schedules and render count, and measure nothing",
    )
    _add_local_arguments(sampler_cost_parser)
    sampler_cost_parser.add_argument("--verbose", action="store_true", help="log at debug level")

    step_preview_parser = subparsers.add_parser(
        "step-preview",
        help="capture one render step by step and publish it as the two step-preview strips",
    )
    step_preview_parser.add_argument(
        "--spec",
        type=Path,
        default=DEFAULT_SPEC_PATH,
        help="the specification carrying the two step-preview strips",
    )
    step_preview_parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="the output tree's root",
    )
    _add_local_arguments(step_preview_parser)
    step_preview_parser.add_argument("--verbose", action="store_true", help="log at debug level")

    alchemy_parser = subparsers.add_parser(
        "alchemy",
        help="apply every alchemy form to its source, and write the evidence manifest and strip",
    )
    alchemy_parser.add_argument(
        "--spec",
        type=Path,
        default=DEFAULT_ALCHEMY_SPEC_PATH,
        help="the alchemy specification to read",
    )
    alchemy_parser.add_argument(
        "--sweep-spec",
        type=Path,
        default=DEFAULT_SPEC_PATH,
        help="the sweep specification carrying the upscaler-form strip",
    )
    alchemy_parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="the output tree's root; the evidence lands in its alchemy directory",
    )
    alchemy_parser.add_argument(
        "--form",
        action="append",
        default=[],
        choices=list(known_alchemy_forms()),
        metavar="FORM",
        help="restrict the run to this form (repeatable; default: every form the specification places)",
    )
    alchemy_parser.add_argument(
        "--control-type",
        action="append",
        default=[],
        choices=list(known_annotation_control_types()),
        metavar="CONTROL_TYPE",
        help="restrict the run to this control map (repeatable; default: every one the SDK names)",
    )
    alchemy_parser.add_argument(
        "--force",
        action="store_true",
        help="apply selected forms even when their result is already on disk",
    )
    alchemy_parser.add_argument(
        "--plan-only",
        action="store_true",
        help="print the forms and their use cases, and apply nothing",
    )
    alchemy_parser.add_argument(
        "--backend",
        choices=[str(backend) for backend in AlchemyBackendChoice],
        default=str(AlchemyBackendChoice.HORDELIB),
        help=(
            "apply the forms locally through horde-engine, by asking the live horde, or by asking a"
            " horde-image-utilities service for the control maps"
        ),
    )
    alchemy_parser.add_argument(
        "--hiu-url",
        default=None,
        help=(
            "the horde-image-utilities service to draw the control maps on"
            f" (default: {DEFAULT_SERVICE_URL}, or PARAM_SWEEPS_HIU_URL)"
        ),
    )
    # Not `_add_local_arguments`: a post-processing model is small and is fetched by the backend
    # as it resolves the form, so there is no fetch to decline and no `--no-download` to offer.
    alchemy_parser.add_argument(
        "--gpu-backend",
        default=None,
        choices=[str(backend) for backend in GpuBackend],
        help="the compute backend to use (default: cuda, or PARAM_SWEEPS_GPU_BACKEND)",
    )
    alchemy_parser.add_argument("--device-index", type=int, default=0, help="the accelerator index to pin to")
    alchemy_parser.add_argument(
        "--beta",
        action="store_true",
        help=(
            "resolve the "
            f"{', '.join(BETA_MODEL_CATEGORIES)} weights from the pending model reference as well as"
            " the published one, which is what the beta face restorers are carried by"
        ),
    )
    alchemy_parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="the .env holding AIHORDE_DEV_KEY, for --backend api (default: the frontpage repository's)",
    )
    alchemy_parser.add_argument("--api-url", default=None, help="the API root to use instead of the public horde")
    alchemy_parser.add_argument("--verbose", action="store_true", help="log at debug level")

    flows_parser = subparsers.add_parser(
        "flows",
        help="generate the tutorial worked examples through the API and write their manifest",
    )
    flows_parser.add_argument(
        "--spec",
        type=Path,
        default=DEFAULT_FLOWS_SPEC_PATH,
        help="the flow specification to read",
    )
    flows_parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_FLOWS_OUTPUT_ROOT,
        help="the tutorial asset tree to write into (default: the frontpage's)",
    )
    flows_parser.add_argument("--flow", action="append", default=[], help="restrict the run to this flow id")
    flows_parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="the .env holding AIHORDE_DEV_KEY (default: the frontpage repository's, if present)",
    )
    flows_parser.add_argument("--api-url", default=None, help="the API root to use instead of the public horde")
    flows_parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="rewrite tutorials/manifest.json from what is on disk and generate nothing",
    )
    flows_parser.add_argument(
        "--local",
        action="store_true",
        help="render through the worker software on this machine's GPU instead of the public horde",
    )
    flows_parser.add_argument(
        "--gpu-backend",
        default=None,
        choices=[str(backend) for backend in GpuBackend],
        help="with --local: the compute backend to use (default: cuda, or PARAM_SWEEPS_GPU_BACKEND)",
    )
    flows_parser.add_argument(
        "--device-index", type=int, default=0, help="with --local: the accelerator index to pin this process to"
    )
    flows_parser.add_argument(
        "--no-download",
        action="store_true",
        help="with --local: fail instead of downloading a model that is not already in the worker cache",
    )
    flows_parser.add_argument(
        "--dry-run", action="store_true", help="print what would be submitted and generate nothing"
    )
    flows_parser.add_argument("--verbose", action="store_true", help="log at debug level")

    review_parser = subparsers.add_parser(
        "review",
        help="write a static contact sheet of what a batch produced, for visual sign-off",
    )
    review_parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC_PATH, help="the sweep specification to read")
    review_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="the output tree's root")
    review_parser.add_argument("--title", default="Sign-off contact sheet", help="the heading the sheet carries")
    review_parser.add_argument("--verbose", action="store_true", help="log at debug level")

    choose_parser = subparsers.add_parser(
        "choose",
        help="write the choices saved from the contact sheet into the specification, and refresh the sheet",
    )
    choose_parser.add_argument("choices", type=Path, help="the file the sheet's Save choices button downloaded")
    choose_parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC_PATH, help="the specification to update")
    choose_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="the output tree's root")
    choose_parser.add_argument("--title", default="Sign-off contact sheet", help="the refreshed sheet's heading")
    choose_parser.add_argument("--verbose", action="store_true", help="log at debug level")

    index_parser = subparsers.add_parser("index", help="rebuild index.json from the images already on disk")
    index_parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC_PATH, help="the sweep specification to read")
    index_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="the output tree's root")
    index_parser.add_argument("--verbose", action="store_true", help="log at debug level")

    qr_check_parser = subparsers.add_parser(
        "qr-check",
        help="scan every published QR image and check it decodes to the payload it was asked to carry",
    )
    _add_library_arguments(qr_check_parser)

    manifest_parser = subparsers.add_parser(
        "manifest",
        help="list every file index.json references, with its size and total",
    )
    _add_library_arguments(manifest_parser)
    manifest_parser.add_argument(
        "--keys",
        action="store_true",
        help="print bucket keys under the configured prefix instead of index-relative paths",
    )

    upload_parser = subparsers.add_parser("upload", help="sync the library to S3-compatible storage")
    _add_library_arguments(upload_parser)
    _add_bucket_arguments(upload_parser)
    upload_parser.add_argument("--dry-run", action="store_true", help="print the plan and touch nothing")
    upload_parser.add_argument(
        "--delete",
        action="store_true",
        help="also remove objects under the prefix that neither the index nor a manifest references",
    )

    verify_parser = subparsers.add_parser("verify", help="check a published library against index.json")
    _add_library_arguments(verify_parser)
    _add_bucket_arguments(verify_parser)
    verify_parser.add_argument(
        "--local",
        action="store_true",
        help="check the files on disk instead of the bucket, and need no credentials",
    )

    return parser


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the specification, output and filter arguments every rendering subcommand shares."""
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC_PATH, help="the sweep specification to read")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="the output tree's root")
    parser.add_argument(
        "--parameter",
        action="append",
        default=[],
        choices=[str(parameter) for parameter in SweepParameter],
        help="restrict the run to this parameter (repeatable)",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="restrict the run to this model slug (repeatable)",
    )
    parser.add_argument("--scene", action="append", default=[], help="restrict the run to this scene id (repeatable)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="render selected images even when the checkpoint already accounts for them",
    )
    parser.add_argument("--verbose", action="store_true", help="log at debug level")


def _add_library_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the arguments a subcommand that works from the rendered library alone needs."""
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="the output tree's root")
    parser.add_argument("--verbose", action="store_true", help="log at debug level")


def _add_bucket_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the arguments that point a subcommand at the bucket the library is published to."""
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="the .env holding the PARAM_SWEEPS_S3_* variables (default: the frontpage repository's)",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="the key prefix to publish under, overriding PARAM_SWEEPS_S3_PREFIX",
    )


def _add_local_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the arguments that select the compute backend for a local run."""
    parser.add_argument(
        "--gpu-backend",
        default=None,
        choices=[str(backend) for backend in GpuBackend],
        help="the compute backend to use (default: cuda, or PARAM_SWEEPS_GPU_BACKEND)",
    )
    parser.add_argument("--device-index", type=int, default=0, help="the accelerator index to pin this process to")
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="fail instead of downloading a model that is not already in the worker cache",
    )


def main(argv: list[str] | None = None) -> int:
    """Run the command line, and return the process exit code.

    Args:
        argv: The arguments to parse, or None to read ``sys.argv``.

    Returns:
        0 when everything asked for succeeded, 1 otherwise.
    """
    arguments = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if arguments.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if arguments.command in PUBLISHING_COMMANDS:
        return _run_publishing_command(arguments)

    if arguments.command == "qr-check":
        return _run_qr_check(arguments)

    if arguments.command == "alchemy":
        return _run_alchemy(arguments)

    if arguments.command == "flows":
        return _run_flows(arguments)

    if arguments.command == "choose":
        return _run_choose(arguments)

    spec = load_spec(arguments.spec)
    output_root: Path = arguments.output_root

    if arguments.command == "sampler-cost":
        return _run_sampler_cost(arguments, spec=spec, output_root=output_root)

    if arguments.command == "step-preview":
        return _run_step_preview(arguments, spec=spec, output_root=output_root)

    if arguments.command == "index":
        index_path = write_index(spec=spec, output_root=output_root)
        print(f"Wrote {index_path}")
        return 0

    if arguments.command == "review":
        sheet_path = write_contact_sheet(spec=spec, output_root=output_root, title=arguments.title)
        print(f"Wrote {sheet_path}")
        return 0

    job_filter = JobFilter(
        parameters=tuple(arguments.parameter),
        model_slugs=tuple(arguments.model),
        scene_ids=tuple(arguments.scene),
    )

    try:
        runner = _build_runner(arguments)
    except (ValueError, MissingApiKeyError) as configuration_error:
        print(f"error: {configuration_error}", file=sys.stderr)
        return 1

    result = run_campaign(
        spec=spec,
        runner=runner,
        output_root=output_root,
        job_filter=job_filter,
        plan_only=arguments.command == "plan",
        force=arguments.force,
    )
    return _report(result)


def _build_runner(arguments: argparse.Namespace) -> SweepRunner:
    """Create the runner the chosen subcommand needs.

    ``plan`` uses the local runner because the plan it prints is about this machine's cache.

    Raises:
        ValueError: If an unsupported GPU backend was named.
        MissingApiKeyError: If the horde runner was asked for and no key could be found.
    """
    if arguments.command == "horde":
        env_file = arguments.env_file if arguments.env_file is not None else default_env_file()
        api_key = read_api_key(env_file=env_file)
        if arguments.api_url is not None:
            return HordeApiRunner(api_key=api_key, api_base_url=arguments.api_url)
        return HordeApiRunner(api_key=api_key)

    return LocalRunner(
        gpu_backend=resolve_gpu_backend(requested=arguments.gpu_backend),
        device_index=arguments.device_index,
        download_missing_models=not arguments.no_download,
    )


def _run_sampler_cost(arguments: argparse.Namespace, *, spec: SweepSpec, output_root: Path) -> int:
    """Measure sampler cost, write the artifact and print the summary, returning the exit code.

    Everything held constant comes from the specification, so the measurement is anchored to the
    same seed and cfg scale the sampler sign-off was rendered at rather than to numbers stated
    twice. The sampler vocabulary and the schedule pairings come from the shared constraints table.
    """
    try:
        models = _sampler_cost_models(spec=spec, requested_slugs=tuple(arguments.model))
        policy = horde_sdk_schedule_policy()
        samplers = tuple(arguments.sampler) if arguments.sampler else horde_sdk_samplers()
        step_counts = tuple(arguments.steps) if arguments.steps else DEFAULT_STEP_COUNTS
        pairings_by_model = {
            image_model.slug: derive_pairings(samplers=samplers, baseline=image_model.baseline, policy=policy)
            for image_model in models
        }
    except (KeyError, RuntimeError, ValueError) as plan_error:
        print(f"error: {plan_error}", file=sys.stderr)
        return 1

    for image_model in models:
        print(f"{image_model.slug} ({image_model.baseline}, {image_model.native_width}x{image_model.native_height}):")
        for pairing in pairings_by_model[image_model.slug]:
            print(
                f"  {pairing.sampler_name} on {pairing.scheduler}"
                f" [{pairing.schedule_reason}, {pairing.evaluations_per_step} eval/step]",
            )

    render_count = sum(
        len(pairings) * len(step_counts) * arguments.repeats + arguments.warmups
        for pairings in pairings_by_model.values()
    )
    print(f"{render_count} render(s), warmups included.")
    if arguments.plan_only:
        return 0

    try:
        runner = _build_runner(arguments)
    except ValueError as configuration_error:
        print(f"error: {configuration_error}", file=sys.stderr)
        return 1

    report = measure_sampler_cost(
        runner=runner,
        models=models,
        policy=policy,
        samplers=samplers,
        seed=spec.seed,
        cfg_scale=spec.anchor.cfg_scale,
        step_counts=step_counts,
        repeats=arguments.repeats,
        warmup_renders=arguments.warmups,
    )
    report_path = write_report(report=report, output_root=output_root)

    print()
    print(format_summary_table(report))
    print(f"Wrote {report_path}")
    return 1 if any(record.failed_renders for record in report.records) else 0


def _run_step_preview(arguments: argparse.Namespace, *, spec: SweepSpec, output_root: Path) -> int:
    """Capture one render step by step, write both arcs, and return the exit code.

    This renders once however many stops the two strips declare, because the stops are moments of
    a single render rather than renders of their own. Everything it can fail on is a
    configuration problem or a pipeline that did not sample the way a step preview needs, so each
    is reported as one line on stderr rather than a traceback.
    """
    try:
        runner = _build_runner(arguments)
        written = run_step_preview(runner=runner, spec=spec, output_root=output_root)
    except (RuntimeError, ValueError) as capture_error:
        print(f"error: {capture_error}", file=sys.stderr)
        return 1

    for arc, paths in written.items():
        print(f"Wrote {len(paths)} {arc} frame(s) under {paths[0].parent}")
    return 0


def _sampler_cost_models(*, spec: SweepSpec, requested_slugs: tuple[str, ...]) -> list[ModelSpec]:
    """Return the models to measure, in the order they were asked for.

    Raises:
        KeyError: If a named slug is not in the specification.
    """
    slugs = requested_slugs if requested_slugs else DEFAULT_SAMPLER_COST_MODEL_SLUGS
    return [spec.model_for_slug(slug) for slug in slugs]


def _alchemy_api_client(arguments: argparse.Namespace) -> HordeApiClient:
    """Return the client the live alchemy backend submits through.

    Raises:
        MissingApiKeyError: If no key was found where it was looked for.
    """
    env_file = arguments.env_file if arguments.env_file is not None else default_env_file()
    api_key = read_api_key(env_file=env_file)
    if arguments.api_url is not None:
        return HordeApiClient(api_key=api_key, api_base_url=arguments.api_url)
    return HordeApiClient(api_key=api_key)


def _alchemy_result_label(result: AlchemyResult) -> str:
    """Return what to call one applied entry in a report, which for a control map is its detector."""
    return result.control_type if result.control_type is not None else result.form


def _run_alchemy(arguments: argparse.Namespace) -> int:
    """Apply the selected alchemy forms, and return the process exit code.

    A form the backend cannot run is not a failure of the run: it is published as an entry that
    says so, and the exit code stays 0 so a campaign is not made to look broken by a processor
    that this machine was never going to be able to serve.
    """
    try:
        alchemy_spec = load_alchemy_spec(arguments.spec)
        sweep_spec = load_spec(arguments.sweep_spec)
        gpu_backend = resolve_gpu_backend(requested=arguments.gpu_backend)
    except (OSError, ValueError) as configuration_error:
        print(f"error: {configuration_error}", file=sys.stderr)
        return 1

    counts = use_case_counts(alchemy_spec)
    for use_case in AlchemyUseCase:
        placed = [entry.label for entry in alchemy_spec.entries if entry.use_case is use_case]
        print(f"{use_case} ({counts[use_case]}): {', '.join(placed)}")

    if arguments.plan_only:
        return 0

    drawing_control_maps = arguments.backend == AlchemyBackendChoice.IMAGE_UTILITIES
    # An unrestricted run through the control-map service would reach every other entry too and
    # overwrite the reasons the backends that do serve them recorded, so it covers its own form.
    forms = tuple(arguments.form) if arguments.form else (ANNOTATION_FORM,) if drawing_control_maps else ()

    client: HordeApiClient | None = None
    try:
        if arguments.backend == AlchemyBackendChoice.API:
            client = _alchemy_api_client(arguments)
            backend: AlchemyBackend = LiveHordeAlchemyBackend(client=client)
        elif drawing_control_maps:
            backend = HordeImageUtilitiesBackend(service_url=resolve_service_url(requested=arguments.hiu_url))
        else:
            backend = HordelibAlchemyBackend(
                gpu_backend=str(gpu_backend),
                device_index=arguments.device_index,
                beta_categories=BETA_MODEL_CATEGORIES if arguments.beta else (),
            )

        results = run_alchemy(
            spec=alchemy_spec,
            sweep_spec=sweep_spec,
            backend=backend,
            output_root=arguments.output_root,
            forms=forms,
            control_types=tuple(arguments.control_type),
            force=arguments.force,
        )
    except (MissingApiKeyError, RuntimeError, ValueError) as alchemy_error:
        print(f"error: {alchemy_error}", file=sys.stderr)
        return 1
    finally:
        if client is not None:
            client.close()

    unavailable = [_alchemy_result_label(result) for result in results if not result.available]
    reused = [result for result in results if result.reused]
    print(
        f"Applied {len(results) - len(unavailable) - len(reused)}, reused {len(reused)},"
        f" unavailable {len(unavailable)}.",
    )
    if unavailable:
        print(f"Unavailable here (published as entries that say so): {', '.join(unavailable)}", file=sys.stderr)
    return 0


def _run_flows(arguments: argparse.Namespace) -> int:
    """Generate the selected tutorial flows, and return the process exit code.

    A dry run needs no key, because it submits nothing: the plan it prints is about the
    specification rather than about the horde.
    """
    flows_spec = load_flows_spec(arguments.spec)

    if arguments.manifest_only:
        written = write_manifest(spec=flows_spec, output_root=arguments.output_root)
        print(f"Wrote {written}")
        return 0

    if arguments.dry_run:
        for result in run_flows(
            spec=flows_spec,
            output_root=arguments.output_root,
            flow_ids=tuple(arguments.flow),
            plan_only=True,
        ):
            print(f"Would generate {result.flow_id}")
        return 0

    if arguments.local:
        results = _run_flows_locally(arguments, flows_spec=flows_spec)
    else:
        try:
            env_file = arguments.env_file if arguments.env_file is not None else default_env_file()
            api_key = read_api_key(env_file=env_file)
        except MissingApiKeyError as key_error:
            print(f"error: {key_error}", file=sys.stderr)
            return 1

        client = (
            HordeApiClient(api_key=api_key, api_base_url=arguments.api_url)
            if arguments.api_url is not None
            else HordeApiClient(api_key=api_key)
        )
        try:
            results = run_flows(
                spec=flows_spec,
                generator=HordeFlowGenerator(client),
                output_root=arguments.output_root,
                flow_ids=tuple(arguments.flow),
            )
        finally:
            client.close()

    generated = [result for result in results if not result.skipped]
    print(f"Generated {len(generated)}, skipped {len(results) - len(generated)}.")
    return 0


def _run_flows_locally(arguments: argparse.Namespace, *, flows_spec: FlowsSpec) -> list[FlowResult]:
    """Render the selected flows on this machine's GPU, and return what each one produced.

    The local runner is prepared with a model entry per distinct model the selected flows name,
    built from the flow requests themselves: the flows use models the sweep library does not
    (the inpainting checkpoints, Stable Cascade), so the sweep specification cannot supply them.
    Native sizes and the like are unused by the runner's preparation, which only fetches weights.
    """
    selected = [flow for flow in flows_spec.flows if not arguments.flow or flow.flow_id in arguments.flow]
    model_names = sorted(
        {flow.request.model for flow in selected if flow.status is FlowStatus.AVAILABLE}
        | {flow.before_request.model for flow in selected if flow.before_request is not None}
    )
    models = [
        ModelSpec(
            slug=re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-"),
            horde_name=name,
            baseline="",
            native_width=0,
            native_height=0,
            anchor_clip_skip=1,
            approximate_download_size_gb=0.0,
            note="named by a tutorial flow",
        )
        for name in model_names
    ]
    runner = LocalRunner(
        gpu_backend=resolve_gpu_backend(requested=arguments.gpu_backend),
        device_index=arguments.device_index,
        download_missing_models=not arguments.no_download,
    )
    try:
        runner.prepare([], models)
        return run_flows(
            spec=flows_spec,
            generator=LocalFlowGenerator(runner),
            output_root=arguments.output_root,
            flow_ids=tuple(arguments.flow),
        )
    finally:
        runner.close()


def _run_qr_check(arguments: argparse.Namespace) -> int:
    """Scan every published QR image, print what each read, and return the exit code.

    This is the gate over what shipped rather than over what was rendered: a curation batch is
    published one candidate at a time and a human's pick is the last thing that decides which
    image a reader points a phone at.
    """
    report = check_library(output_root=arguments.output_root)

    for check in report.checks:
        verdict = "scans" if check.scans else f"FAILED (read {check.decoded!r})"
        print(f"{verdict}\t{check.path}")

    print(f"Scanned {len(report.checks)} published QR image(s), {len(report.failures)} did not decode.")
    return 0 if report.is_intact else 1


def _run_choose(arguments: argparse.Namespace) -> int:
    """Apply a saved choices file to the specification, refresh the sheet, and return the exit code.

    Every failure this can hit is a malformed choices file or a file problem the reviewer has to
    act on, so each is reported as one line on stderr rather than a traceback.
    """
    try:
        changed_count = apply_choices(spec_path=arguments.spec, choices_path=arguments.choices)
    except (OSError, ValueError) as choices_error:
        print(f"error: {choices_error}", file=sys.stderr)
        return 1

    spec = load_spec(arguments.spec)
    sheet_path = write_contact_sheet(spec=spec, output_root=arguments.output_root, title=arguments.title)
    print(f"Applied {changed_count} changed pick(s) to {arguments.spec.name} and refreshed {sheet_path}")
    return 0


def _run_publishing_command(arguments: argparse.Namespace) -> int:
    """Run ``manifest``, ``upload`` or ``verify``, and return the process exit code.

    Every failure these can hit is a configuration, dependency or bucket problem the operator has
    to act on, so each is reported as one line on stderr rather than a traceback.
    """
    try:
        if arguments.command == "manifest":
            return _run_manifest(arguments)
        if arguments.command == "upload":
            return _run_upload(arguments)
        return _run_verify(arguments)
    except (
        FileNotFoundError,
        IncompleteLibraryError,
        MissingPublishDependencyError,
        MissingS3ConfigurationError,
    ) as library_error:
        print(f"error: {library_error}", file=sys.stderr)
        return 1
    except bucket_transport_errors() as bucket_error:
        print(f"error: the bucket rejected or could not serve the request: {bucket_error}", file=sys.stderr)
        return 1


def _run_manifest(arguments: argparse.Namespace) -> int:
    """List every file the index and the manifests reference, with sizes and a total, and return 0."""
    files = enumerate_library(output_root=arguments.output_root)
    prefix = _prefix_for_manifest(arguments)

    for library_file in files:
        printed_path = (
            object_key(prefix=prefix, relative_path=library_file.relative_path)
            if arguments.keys
            else library_file.relative_path
        )
        print(f"{library_file.size_bytes}\t{printed_path}")

    total_bytes = sum(library_file.size_bytes for library_file in files)
    print(f"{len(files)} files, {format_byte_size(total_bytes)}", file=sys.stderr)
    return 0


def _run_upload(arguments: argparse.Namespace) -> int:
    """Sync the library to the bucket, or print what a sync would do, and return the exit code."""
    files = enumerate_library(output_root=arguments.output_root)
    bucket = SweepLibraryBucket(settings=_bucket_settings(arguments))
    plan = plan_sync(
        files=files,
        remote_objects=bucket.list_objects(),
        prefix=bucket.settings.prefix,
        delete_orphans=arguments.delete,
    )

    _print_plan(plan, dry_run=arguments.dry_run)
    if arguments.dry_run:
        return 0

    apply_sync_plan(bucket=bucket, plan=plan)
    print(f"Uploaded {len(plan.uploads)}, deleted {len(plan.deletions)}, unchanged {len(plan.unchanged_keys)}.")
    return 0


def _run_verify(arguments: argparse.Namespace) -> int:
    """Check the library against its index, and return 0 only when nothing is missing or wrong."""
    if arguments.local:
        report = verify_local(output_root=arguments.output_root)
        where = str(arguments.output_root)
    else:
        files = enumerate_library(output_root=arguments.output_root)
        bucket = SweepLibraryBucket(settings=_bucket_settings(arguments))
        report = verify_remote(
            files=files,
            remote_objects=bucket.list_objects(),
            prefix=bucket.settings.prefix,
        )
        settings = bucket.settings
        where = f"{settings.bucket}/{settings.prefix}" if settings.prefix else settings.bucket

    print(f"Checked {report.checked_count} referenced file(s) against {where}.")
    _print_paths("Missing", report.missing_paths)
    _print_paths("Hash mismatch" if not arguments.local else "Empty", report.mismatched_paths)
    _print_paths("Orphaned (not referenced by the index)", report.orphan_keys, to_stderr=False)

    if report.is_intact:
        print("The library is intact.")
        return 0

    return 1


def _bucket_settings(arguments: argparse.Namespace) -> S3Settings:
    """Return the bucket settings for a subcommand, from its flags, the environment and a ``.env``."""
    env_file = arguments.env_file if arguments.env_file is not None else default_env_file()
    return load_s3_settings(env_file=env_file, prefix_override=arguments.prefix)


def _prefix_for_manifest(arguments: argparse.Namespace) -> str:
    """Return the prefix ``manifest --keys`` should print under, which needs no credentials without it."""
    if not arguments.keys:
        return ""

    return _bucket_settings(arguments).prefix


def _print_plan(plan: SyncPlan, *, dry_run: bool) -> None:
    """Print an upload plan in full, in the order it would be carried out."""
    heading = "Would upload" if dry_run else "Uploading"
    for upload in plan.uploads:
        print(f"{heading} [{upload.reason}] {upload.key} ({format_byte_size(upload.size_bytes)})")

    for key in plan.deletions:
        print(f"{'Would delete' if dry_run else 'Deleting'} {key}")

    undeleted_orphans = [key for key in plan.orphan_keys if key not in plan.deletions]
    for key in undeleted_orphans:
        print(f"Orphan (left in place; pass --delete to remove) {key}")

    print(
        f"{len(plan.uploads)} to upload ({format_byte_size(plan.upload_bytes)}),"
        f" {len(plan.unchanged_keys)} unchanged, {len(plan.deletions)} to delete,"
        f" {len(undeleted_orphans)} orphan(s) left alone.",
    )


def _print_paths(heading: str, paths: list[str], *, to_stderr: bool = True) -> None:
    """Print a heading and its paths, or nothing at all when there are none."""
    if not paths:
        return

    stream = sys.stderr if to_stderr else sys.stdout
    print(f"{heading}: {len(paths)}", file=stream)
    for path in paths:
        print(f"  {path}", file=stream)


def _report(result: CampaignResult) -> int:
    """Print what the campaign did, and return the exit code that reflects it."""
    print(
        f"Selected {result.selected_count}, rendered {result.rendered_count},"
        f" already present {result.skipped_count}, failed {len(result.failed_jobs)}.",
    )
    if not result.failed_jobs:
        return 0

    print("Failed:", file=sys.stderr)
    for failed_image_path in result.failed_jobs:
        print(f"  {failed_image_path}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
