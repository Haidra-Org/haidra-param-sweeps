# SPDX-FileCopyrightText: 2026 Tazlin
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generate the worked examples the tutorials walk a reader through, and the manifest that lists them.

A flow is one tutorial example: a single request, the image it produced, and enough of a record
beside it that a reader can see exactly what was asked for. It is not a sweep. Nothing is held
constant across flows and nothing is being taught by the difference between two of them, so a
flow states its whole request rather than inheriting one.

The output layout is the frontpage's ``src/assets/tutorials``: one directory per flow holding
``request.json``, ``result.webp``, ``meta.json`` and whatever inputs the flow used, with a
``manifest.json`` at the root listing every flow and its files.

**No worker attribution is recorded.** ``meta.json`` carries what the request cost and what
identifies it (kudos, wall time, generation ids) and never who served it. The transport does not
parse a worker name at all, so there is nothing here to leak.

A flow is turned into a [`RenderJob`][param_sweeps.spec.RenderJob] before it is submitted, which
is what lets the flows and the sweeps share one payload builder and one API client: a change to
how a feature reaches the horde cannot apply to one of them and not the other.

Critical public members:

- [`FlowsSpec`][param_sweeps.flows.FlowsSpec]: the whole checked-in flow specification.
- [`FlowSpec`][param_sweeps.flows.FlowSpec]: one flow, its request and its expected artifacts.
- [`run_flows`][param_sweeps.flows.run_flows]: generate the selected flows and write the manifest.
- [`build_manifest`][param_sweeps.flows.build_manifest]: the manifest as the frontpage reads it.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Literal, Protocol

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator
from strenum import StrEnum

from param_sweeps.horde_client import CompletedGeneration, HordeApiClient, HordeGeneration
from param_sweeps.horde_runner import HordeGenerationRequest
from param_sweeps.outputs import WebpEncoding
from param_sweeps.runner import SweepRunner
from param_sweeps.spec import (
    TOOL_ROOT,
    GenerationParameters,
    JobFeatures,
    RenderJob,
    SweepParameter,
)

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "manifest.json"
"""The name of the file listing every flow, at the root of the tutorial asset tree."""

TUTORIALS_DIRECTORY_NAME = "tutorials"
"""The directory the tutorial asset tree lives in, under the library root."""

REQUEST_FILENAME = "request.json"
BEFORE_REQUEST_FILENAME = "request-before.json"
META_FILENAME = "meta.json"
RESULT_FILENAME = "result.webp"
BEFORE_FILENAME = "before.webp"
SOURCE_FILENAME = "source.webp"
MASK_FILENAME = "mask.png"
REFERENCE_FILENAME_PREFIX = "reference-"
"""A remix's extra source images are published beside the source as reference-1.webp, reference-2.webp
and so on, in the order the request carries them, so a reader can see every picture that went in."""

SOURCE_IMAGE_PLACEHOLDER = f"<base64 of {SOURCE_FILENAME}>"
"""What stands in for the encoded source image in the published request.

The request that is published has to be readable and has to be diffable. A megabyte of base64
is neither, and a reader who wants the bytes has the file it names sitting beside it.
"""

FROZEN_MODEL_CONFIG = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())


class FlowStatus(StrEnum):
    """Whether a flow has a worked example behind it."""

    AVAILABLE = "available"
    SKIPPED = "skipped"


class FlowRequest(BaseModel):
    """Represents one request a flow makes: the model, the parameters and any input images.

    Every parameter is stated rather than inherited. A tutorial example is a thing a reader is
    invited to reproduce exactly, so there is no anchor for it to drift away from.
    """

    model_config = FROZEN_MODEL_CONFIG

    model: str
    """The horde model name, as the API takes it."""

    params: GenerationParameters
    features: JobFeatures = Field(default_factory=JobFeatures)
    source_image: str | None = None
    """The image this request starts from, relative to the tool root."""

    source_mask: str | None = None
    """The mask this request uses, relative to the tool root."""

    def to_render_job(self, *, flow_id: str, title: str) -> RenderJob:
        """Return this request as the render job the shared payload builder takes.

        A flow has no sweep, model slug or stop, so the identity fields carry the flow's own
        identity. Nothing downstream of the payload builder reads them for a flow: the files a
        flow writes are named by this module, not by the job.
        """
        features = self.features
        return RenderJob(
            parameter=SweepParameter.BATCH,
            model_slug=flow_id,
            horde_model_name=self.model,
            scene_id=flow_id,
            prompt=self.params.prompt,
            stop_index=1,
            stop_label=flow_id,
            stop_value=flow_id,
            caption=title,
            width=self.params.width,
            height=self.params.height,
            steps=self.params.steps,
            cfg_scale=self.params.cfg_scale,
            sampler_name=self.params.sampler_name,
            clip_skip=self.params.clip_skip,
            hires_fix=self.params.hires_fix,
            tiling=self.params.tiling,
            karras=self.params.karras,
            denoising_strength=self.params.denoising_strength,
            seed=self.params.seed,
            negative_prompt=self.params.negative_prompt,
            source_image=self.source_image,
            source_mask=self.source_mask,
            control_type=features.control_type,
            control_strength=features.control_strength,
            image_is_control=features.image_is_control,
            return_control_map=features.return_control_map,
            source_processing=features.source_processing,
            workflow=features.workflow,
            extra_texts=list(features.extra_texts),
            extra_source_images=list(features.extra_source_images),
            transparent=features.transparent,
            loras=list(features.loras),
            tis=list(features.tis),
            post_processing=list(features.post_processing),
            facefixer_strength=features.facefixer_strength,
            batch_size=features.batch_size,
        )


class FlowSpec(BaseModel):
    """Represents one tutorial flow: what it teaches, what it asks for, and what it leaves behind."""

    model_config = FROZEN_MODEL_CONFIG

    flow_id: str
    title: str
    summary: str
    request: FlowRequest
    before_request: FlowRequest | None = None
    """The other half of a pair, for the flows whose lesson is the difference between two renders."""

    status: FlowStatus = FlowStatus.AVAILABLE
    skipped_reason: str | None = None
    """Why no example exists, which is published in place of one."""

    source_note: str | None = None
    """Where the source image came from, in the words a reader needs."""

    notes: str | None = None

    needs_attention: str | None = None
    """A marker that this flow's recipe is not settled: what was tried, what it produced, and what to
    vary next. Set on a flow whose published example is a compromise (a skipped world, a scan
    that took a seed search, a lesson that reads weaker than the others), so the next campaign
    starts from the record rather than rediscovering it. The dry-run plan lists every flow that
    carries one."""

    @model_validator(mode="after")
    def _check_a_skipped_flow_says_why(self) -> FlowSpec:
        """Validate that a flow without an example explains itself."""
        if self.status is FlowStatus.SKIPPED and not self.skipped_reason:
            raise ValueError(f"flow {self.flow_id} is skipped and must say why")
        return self

    @property
    def directory_name(self) -> str:
        """Return the directory this flow's artifacts live in, under the output root."""
        return self.flow_id


class FlowsSpec(BaseModel):
    """Represents the whole checked-in flow specification."""

    model_config = FROZEN_MODEL_CONFIG

    schema_version: Literal[1]
    generated_against: str
    """The API root the published examples were generated against, as the manifest reports it."""

    flows: list[FlowSpec]

    @model_validator(mode="after")
    def _check_flow_ids_are_unique(self) -> FlowsSpec:
        """Validate that no two flows would write into the same directory."""
        flow_ids = [flow.flow_id for flow in self.flows]
        if len(set(flow_ids)) != len(flow_ids):
            raise ValueError("flow ids must be unique")
        return self

    def flow_for_id(self, flow_id: str) -> FlowSpec:
        """Return the flow with this identifier.

        Raises:
            KeyError: If no flow in the specification carries the identifier.
        """
        for flow in self.flows:
            if flow.flow_id == flow_id:
                return flow
        raise KeyError(f"no flow with id {flow_id!r}")


class FlowMeta(BaseModel):
    """Represents a flow's ``meta.json``: what the request cost and what identifies it.

    There is no worker name or worker id here, and there is no field for one. What an example
    cost and which generation it was are the honest disclosures; who happened to serve it is not
    published.
    """

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    flow: str
    title: str
    model: str
    seed: str
    requested_seed: str
    generation_id: str
    kudos_estimate: float | None = None
    kudos_charged: float | None = None
    """What the horde charged. Absent on a local render, which is charged nothing."""
    wall_time_seconds: float
    censored: bool
    runner: Literal["horde", "local"] = "horde"
    """Which runner made the example: the public horde, or the worker software on a local GPU."""
    gen_metadata: list[dict[str, Any]] = Field(default_factory=list)
    source: str | None = None
    notes: str | None = None
    before_generation_id: str | None = None
    before_seed: str | None = None
    before_kudos_charged: float | None = None
    before_wall_time_seconds: float | None = None


class ManifestFlow(BaseModel):
    """Represents one flow as the frontpage's manifest lists it."""

    model_config = ConfigDict(frozen=True, populate_by_name=True, protected_namespaces=())

    flow_id: str = Field(alias="id")
    """Written ``id`` in the manifest, which is the key the frontpage looks flows up by."""

    directory: str
    title: str
    summary: str
    status: FlowStatus
    files: list[str]
    model: str | None = None
    seed: str | None = None
    result: str | None = None
    source: str | None = None
    mask: str | None = None
    before: str | None = None
    request: str | None = None
    references: list[str] | None = None
    """The extra source images a remix blended in, in request order."""
    skipped_reason: str | None = None


class TutorialManifest(BaseModel):
    """Represents the whole manifest at the root of the tutorial asset tree."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    generated_against: str
    flows: list[ManifestFlow]


class FlowResult(BaseModel):
    """Represents what generating one flow produced."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    flow_id: str
    written_files: list[str]
    skipped: bool = False


def load_flows_spec(spec_path: Path) -> FlowsSpec:
    """Read and validate the flow specification at this path.

    Args:
        spec_path: The specification file to read.

    Returns:
        The validated specification.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the file is not valid JSON, or does not validate as a specification.
    """
    raw_text = spec_path.read_text(encoding="utf-8")
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as decode_error:
        raise ValueError(f"{spec_path} is not valid JSON: {decode_error}") from decode_error

    return FlowsSpec.model_validate(parsed)


def publishable_request(flow_request: FlowRequest, *, flow_id: str, title: str) -> dict[str, Any]:
    """Return the request body a flow publishes, with the encoded images replaced by their names.

    Args:
        flow_request: The request to render publishable.
        flow_id: The flow it belongs to.
        title: The flow's title, which the job carries as its caption.

    Returns:
        The body, ready to be written as ``request.json``.
    """
    body = HordeGenerationRequest.from_job(flow_request.to_render_job(flow_id=flow_id, title=title)).as_request_body()

    if body.get("source_image") is not None:
        body["source_image"] = SOURCE_IMAGE_PLACEHOLDER
    if body.get("source_mask") is not None:
        body["source_mask"] = f"<base64 of {Path(flow_request.source_mask or '').name}>"

    extra_source_images = body.get("extra_source_images")
    if extra_source_images is not None:
        for published, declared in zip(extra_source_images, flow_request.features.extra_source_images, strict=True):
            published["image"] = f"<base64 of {Path(declared.image).name}>"

    return body


def write_flow(
    *,
    flow: FlowSpec,
    completed: CompletedGeneration,
    before_completed: CompletedGeneration | None,
    output_root: Path,
    runner: Literal["horde", "local"] = "horde",
) -> FlowResult:
    """Write one generated flow's whole directory, and return what was written.

    Args:
        flow: The flow that was generated.
        completed: What the horde returned for its request.
        before_completed: What the horde returned for the other half of a pair, when it has one.
        output_root: The root of the tutorial asset tree.
        runner: Which runner made the generations, as the meta records it.

    Returns:
        The flow's result, listing every file written, sorted.

    Side Effects:
        Creates the flow's directory and overwrites every artifact in it.
    """
    flow_directory = output_root / flow.directory_name
    flow_directory.mkdir(parents=True, exist_ok=True)

    _save_image(completed.images[0], flow_directory / RESULT_FILENAME)
    _write_json(
        flow_directory / REQUEST_FILENAME,
        publishable_request(flow.request, flow_id=flow.flow_id, title=flow.title),
    )
    _copy_input(flow.request.source_image, flow_directory / SOURCE_FILENAME)
    mask_name = _copy_mask(flow.request.source_mask, flow_directory)
    for index, extra in enumerate(flow.request.features.extra_source_images, start=1):
        _copy_input(extra.image, flow_directory / f"{REFERENCE_FILENAME_PREFIX}{index}.webp")

    if flow.before_request is not None and before_completed is not None:
        _save_image(before_completed.images[0], flow_directory / BEFORE_FILENAME)
        _write_json(
            flow_directory / BEFORE_REQUEST_FILENAME,
            publishable_request(flow.before_request, flow_id=flow.flow_id, title=flow.title),
        )

    generation = completed.generations[0]
    before_generation = before_completed.generations[0] if before_completed is not None else None
    meta = FlowMeta(
        flow=flow.flow_id,
        title=flow.title,
        model=flow.request.model,
        seed=generation.seed or str(flow.request.params.seed),
        requested_seed=str(flow.request.params.seed),
        generation_id=generation.generation_id,
        kudos_estimate=completed.kudos_estimate if runner == "horde" else None,
        kudos_charged=completed.kudos_charged if runner == "horde" else None,
        wall_time_seconds=completed.wall_time_seconds,
        runner=runner,
        censored=generation.censored,
        gen_metadata=list(generation.gen_metadata),
        source=flow.source_note,
        notes=flow.notes,
        before_generation_id=before_generation.generation_id if before_generation is not None else None,
        before_seed=before_generation.seed if before_generation is not None else None,
        before_kudos_charged=(
            before_completed.kudos_charged if before_completed is not None and runner == "horde" else None
        ),
        before_wall_time_seconds=before_completed.wall_time_seconds if before_completed is not None else None,
    )
    _write_json(flow_directory / META_FILENAME, meta.model_dump(exclude_none=True))

    logger.info("Wrote flow: flow=%s mask=%s", flow.flow_id, mask_name)
    return FlowResult(flow_id=flow.flow_id, written_files=_listed_files(flow_directory))


def build_manifest(*, spec: FlowsSpec, output_root: Path) -> TutorialManifest:
    """Build the manifest from the specification and the files actually on disk.

    A flow with no directory is left out entirely rather than listed as available with nothing
    behind it, so the manifest never advertises a file that is not there.

    Args:
        spec: The flow specification.
        output_root: The root of the tutorial asset tree.

    Returns:
        The manifest.
    """
    manifest_flows: list[ManifestFlow] = []

    for flow in spec.flows:
        flow_directory = output_root / flow.directory_name
        if not flow_directory.is_dir():
            continue

        files = _listed_files(flow_directory)
        if not files:
            continue

        if flow.status is FlowStatus.SKIPPED:
            manifest_flows.append(
                ManifestFlow(
                    flow_id=flow.flow_id,
                    directory=flow.directory_name,
                    title=flow.title,
                    summary=flow.summary,
                    status=FlowStatus.SKIPPED,
                    skipped_reason=flow.skipped_reason,
                    files=files,
                ),
            )
            continue

        manifest_flows.append(
            ManifestFlow(
                flow_id=flow.flow_id,
                directory=flow.directory_name,
                title=flow.title,
                summary=flow.summary,
                status=FlowStatus.AVAILABLE,
                model=flow.request.model,
                seed=str(flow.request.params.seed),
                result=RESULT_FILENAME if RESULT_FILENAME in files else None,
                source=SOURCE_FILENAME if SOURCE_FILENAME in files else None,
                mask=_manifest_mask(files),
                before=BEFORE_FILENAME if BEFORE_FILENAME in files else None,
                request=REQUEST_FILENAME if REQUEST_FILENAME in files else None,
                references=_manifest_references(files),
                files=files,
            ),
        )

    return TutorialManifest(generated_against=spec.generated_against, flows=manifest_flows)


def write_manifest(*, spec: FlowsSpec, output_root: Path) -> Path:
    """Write the manifest to the root of the tutorial asset tree, and return its path."""
    manifest_path = output_root / MANIFEST_FILENAME
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(spec=spec, output_root=output_root)
    _write_json(manifest_path, manifest.model_dump(exclude_none=True, by_alias=True))
    return manifest_path


def run_flows(
    *,
    spec: FlowsSpec,
    generator: FlowGenerator | None = None,
    output_root: Path,
    flow_ids: tuple[str, ...] = (),
    plan_only: bool = False,
) -> list[FlowResult]:
    """Generate the selected flows through the given generator, and rewrite the manifest.

    Args:
        spec: The flow specification.
        generator: What renders each request, through the API or on a local GPU. Only a planning
            run may go without one.
        output_root: The root of the tutorial asset tree.
        flow_ids: Which flows to generate. Empty means every available one.
        plan_only: Log what would be submitted and generate nothing.

    Returns:
        One result per flow that was generated or deliberately skipped.

    Raises:
        RuntimeError: If a generating run was asked for without a client, or the horde could not
            produce an image for a selected flow.

    Side Effects:
        Submits generation requests, writes every selected flow's directory, and rewrites the
        manifest. Nothing is submitted and nothing is written when ``plan_only`` is set.
    """
    if generator is None and not plan_only:
        raise RuntimeError("generating flows needs a generator; only a planning run can go without one")

    selected = [flow for flow in spec.flows if not flow_ids or flow.flow_id in flow_ids]
    results: list[FlowResult] = []

    for flow in selected:
        if flow.status is FlowStatus.SKIPPED:
            logger.info("Skipping flow, as the specification says to: flow=%s", flow.flow_id)
            if flow.needs_attention:
                logger.warning("Needs attention: flow=%s %s", flow.flow_id, flow.needs_attention)
            results.append(FlowResult(flow_id=flow.flow_id, written_files=[], skipped=True))
            continue

        if plan_only:
            logger.info(
                "Would submit: flow=%s model=%s size=%sx%s",
                flow.flow_id,
                flow.request.model,
                flow.request.params.width,
                flow.request.params.height,
            )
            if flow.needs_attention:
                logger.warning("Needs attention: flow=%s %s", flow.flow_id, flow.needs_attention)
            results.append(FlowResult(flow_id=flow.flow_id, written_files=[], skipped=True))
            continue

        if generator is None:
            raise RuntimeError("generating flows needs a generator")

        completed = generator.generate(flow=flow, flow_request=flow.request, label=flow.flow_id)
        before_completed = (
            generator.generate(
                flow=flow,
                flow_request=flow.before_request,
                label=f"{flow.flow_id} (before)",
            )
            if flow.before_request is not None
            else None
        )
        results.append(
            write_flow(
                flow=flow,
                completed=completed,
                before_completed=before_completed,
                output_root=output_root,
                runner=generator.runner_name,
            ),
        )

    if not plan_only:
        write_manifest(spec=spec, output_root=output_root)

    return results


class FlowGenerator(Protocol):
    """Whatever turns one of a flow's requests into a finished generation."""

    @property
    def runner_name(self) -> Literal["horde", "local"]:
        """Which runner this is, as the flow's meta records it."""
        ...

    def generate(self, *, flow: FlowSpec, flow_request: FlowRequest, label: str) -> CompletedGeneration:
        """Render one request and return what came back."""
        ...


class HordeFlowGenerator:
    """Generates flows through the public API, which is what the examples are examples of."""

    def __init__(self, client: HordeApiClient) -> None:
        """Wrap the API client the requests are submitted through."""
        self._client = client

    @property
    def runner_name(self) -> Literal["horde", "local"]:
        """Return the runner name the meta records."""
        return "horde"

    def generate(self, *, flow: FlowSpec, flow_request: FlowRequest, label: str) -> CompletedGeneration:
        """Submit one of a flow's requests and return what came back."""
        job = flow_request.to_render_job(flow_id=flow.flow_id, title=flow.title)
        return self._client.generate(request_body=HordeGenerationRequest.from_job(job).as_request_body(), label=label)


class LocalFlowGenerator:
    """Generates flows through the worker software on this machine's GPU.

    The same payload builder the local sweep runner uses takes the flow's job, so a flow rendered
    here is the request the studio would send, run by the software a worker runs, without a queue
    or another machine's hardware between the specification and the picture. Nothing is charged,
    so the meta carries no kudos and says which runner made it.
    """

    def __init__(self, runner: SweepRunner) -> None:
        """Wrap a prepared local runner."""
        self._runner = runner

    @property
    def runner_name(self) -> Literal["horde", "local"]:
        """Return the runner name the meta records."""
        return "local"

    def generate(self, *, flow: FlowSpec, flow_request: FlowRequest, label: str) -> CompletedGeneration:
        """Render one of a flow's requests on the local GPU and return it as a completed generation."""
        job = flow_request.to_render_job(flow_id=flow.flow_id, title=flow.title)
        started = time.monotonic()
        outcome = self._runner.render(job)
        elapsed = time.monotonic() - started
        if outcome.note:
            logger.warning("Local render reported faults: flow=%s note=%s", label, outcome.note)
        return CompletedGeneration(
            request_id=f"local:{label}",
            images=list(outcome.images),
            generations=[
                HordeGeneration(
                    seed=str(job.seed),
                    model=outcome.model_name_used or job.horde_model_name,
                    id=f"local:{label}",
                )
            ],
            wall_time_seconds=elapsed,
        )


def _save_image(image: Image.Image, destination: Path) -> None:
    """Write one image as the quality-tuned WebP the rest of the pipeline writes."""
    image.save(destination, format="WEBP", quality=WebpEncoding.QUALITY, method=WebpEncoding.METHOD)


def _write_json(destination: Path, content: dict[str, Any]) -> None:
    """Write one JSON file with the trailing newline every other file in this tree has."""
    destination.write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _copy_input(relative_path: str | None, destination: Path) -> None:
    """Copy an input image into the flow's directory, so the published tree is self-contained.

    Raises:
        RuntimeError: If the input is not where the specification says it is.
    """
    if relative_path is None:
        return

    source_path = TOOL_ROOT / relative_path
    if not source_path.is_file():
        raise RuntimeError(f"the input {relative_path} is missing: {source_path}")

    shutil.copyfile(source_path, destination)


def _copy_mask(relative_path: str | None, flow_directory: Path) -> str | None:
    """Copy a mask into the flow's directory as ``mask.png``, and return that name.

    The name is fixed rather than the input's own, so the manifest can find it whatever the mask
    was called under ``inputs/``, and so a re-render with a differently named mask replaces the
    old one instead of leaving two beside each other. The lossless PNG is what is published, so
    the mask's binary edge survives.
    """
    if relative_path is None:
        return None

    source_path = TOOL_ROOT / relative_path
    if not source_path.is_file():
        raise RuntimeError(f"the mask {relative_path} is missing: {source_path}")

    shutil.copyfile(source_path, flow_directory / MASK_FILENAME)
    return MASK_FILENAME


def _manifest_references(files: list[str]) -> list[str] | None:
    """Return the reference images a flow published, in request order, or None where there are none."""
    references = sorted(name for name in files if name.startswith(REFERENCE_FILENAME_PREFIX))
    return references or None


def _manifest_mask(files: list[str]) -> str | None:
    """Return the mask a flow published, preferring the lossless one a reader should open."""
    for candidate in (MASK_FILENAME, "mask.webp"):
        if candidate in files:
            return candidate
    return None


def _listed_files(flow_directory: Path) -> list[str]:
    """Return every file in a flow's directory by name, sorted, as the manifest lists them."""
    return sorted(entry.name for entry in flow_directory.iterdir() if entry.is_file())
