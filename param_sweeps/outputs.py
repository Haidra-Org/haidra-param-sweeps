# SPDX-FileCopyrightText: 2026 Tazlin
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Write the rendered images, their sidecars, and the index a web app renders strips from.

Every image is written as a quality-tuned WebP with a JSON sidecar beside it recording the
exact parameters, model, seed, runner and timestamps that produced it. The index collects the
whole library into one file: a consumer that has ``index.json`` needs nothing else to draw a
scrubber strip, because each strip carries what it holds and what varies, and each stop carries
its value, caption and filename.

Both on-disk shapes are declared as models rather than assembled as mappings, so the contract
this tool offers its consumers is stated in one place and checked when it is written.

Critical public members:

- [`ImageSidecar`][param_sweeps.outputs.ImageSidecar]: the per-image record.
- [`SweepIndex`][param_sweeps.outputs.SweepIndex]: the whole library's index.
- [`save_render`][param_sweeps.outputs.save_render]: write one image and its sidecar.
- [`build_index`][param_sweeps.outputs.build_index]: build the index from the specification and
  the files actually present on disk.
- [`image_size`][param_sweeps.outputs.image_size]: the pixel size of a published file, which is
  what lets the index say how big a picture is without decoding it.
- [`write_index`][param_sweeps.outputs.write_index]: write that index deterministically.
- [`publish_source_images`][param_sweeps.outputs.publish_source_images]: copy the images the
  image-to-image strips start from into the library, so it stays self-contained.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from param_sweeps.spec import (
    PREVIEWABLE_PRESENTATIONS,
    TOOL_ROOT,
    BatchSection,
    CropRegion,
    GenerationParameters,
    JobFeatures,
    PublishedCrop,
    RenderJob,
    RenderStrip,
    StripPresentation,
    SweepParameter,
    SweepSpec,
    resolve_preview_indices,
)

logger = logging.getLogger(__name__)

INDEX_FILENAME = "index.json"
"""The name of the top-level index, at the root of the output tree."""

INDEX_SCHEMA_VERSION = 1
"""The index layout consumers can expect."""

FROZEN_MODEL_CONFIG = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())


class WebpEncoding:
    """The WebP encoder settings used for every image in the library.

    Quality 92 with the slowest encoder method keeps sampler noise and fine texture intact,
    which matters because several strips teach differences that live entirely in that texture.
    Lossless would roughly triple the size of the library for no visible gain at these sizes.
    """

    QUALITY = 92
    METHOD = 6

    CROP_QUALITY = 98
    """What a hundred-percent crop is encoded at.

    A crop exists to be read at its own pixels, so the encoder is given far less room to decide
    what counts as texture than it has on a frame nobody is looking that closely at. The crops
    are a small part of the library by area, so the higher quality costs little.
    """


class ImageSidecar(BaseModel):
    """Represents the JSON record written beside every rendered image."""

    model_config = FROZEN_MODEL_CONFIG

    parameter: SweepParameter
    model_slug: str
    model_requested: str
    model_generated_with: str
    scene: str
    prompt: str
    stop_index: int
    stop_value: bool | int | float | str
    caption: str
    params: GenerationParameters
    features: JobFeatures
    runner: str
    started_at: str
    finished_at: str
    image: str
    candidate_number: int = 1
    """Which candidate of a curation batch this image is, one-based."""

    is_chosen: bool = True
    """Whether this is the candidate the index publishes."""

    source_image: str | None = None
    note: str | None = None


class IndexModel(BaseModel):
    """Represents the model a strip was rendered with, as the index reports it."""

    model_config = FROZEN_MODEL_CONFIG

    slug: str
    name: str
    baseline: str
    note: str


class IndexScene(BaseModel):
    """Represents the scene a strip was rendered from, as the index reports it."""

    model_config = FROZEN_MODEL_CONFIG

    scene_id: str
    label: str
    prompt: str


class IndexStop(BaseModel):
    """Represents one stop of a strip, with everything needed to render and caption it.

    The stop's own resolved parameters are deliberately not repeated here: the strip states
    what it holds and what varies, and the value is what a caption is written against. The full
    set is a sidecar away for anyone who wants it, and carrying it per stop trebled the index.
    """

    model_config = FROZEN_MODEL_CONFIG

    value: bool | int | float | str
    label: str
    caption: str
    file: str
    sidecar: str
    width: int
    """The published frame's width in pixels, read from the file itself.

    A consumer that knows the size can say what it is showing and can reserve the space before
    the picture arrives. It is read rather than taken from the request, because a backend can
    hand back a frame at a size nobody asked for and the file is what a reader is looking at.
    """

    height: int
    """The published frame's height in pixels, read from the file itself."""

    prompt_fragment: str | None = None
    """What this stop added to the scene's prompt, on the strips that vary the prompt itself."""

    batch_size: int = 1
    """How many candidates were rendered for this stop, so a consumer can disclose the curation.

    One means the image is the only one the request produced. Above one, a human picked this
    image out of that many, and saying so is the difference between showing what the model does
    and showing what a careful operator can get it to do.
    """

    chosen_candidate: int | None = None
    """Which candidate was published, one-based, when more than one was rendered."""

    control_type: str | None = None
    """The ControlNet guidance mode this stop conditioned on, when it conditioned on one."""

    control_group: str | None = None
    """Which family of detectors that guidance mode belongs to, for a consumer to group by."""

    control_map: str | None = None
    """The map the detector drew, relative to this index, or None when none was published.

    None on a stop that has a control type is the honest report of a detector that would not
    draw: the render is still here, guided by whatever the backend derived internally, and the
    missing map is the thing a consumer says out loud rather than hides.
    """

    crop: PublishedCrop | None = None
    """The part of this frame published at its own pixels, or None where none is.

    Present only where the file is on disk, so a consumer that has the record can fetch it.
    """

    fit: str | None = None
    """A copy of this frame bounded to a page-sized long edge, or None where the frame is already
    within it.

    A consumer drawing the strip fetches this and leaves the full frame to the reader who asks
    for it, which is the difference between a strip that loads and one that does not.
    """


class IndexGridAxis(BaseModel):
    """Represents one axis of a two-dimensional strip."""

    model_config = FROZEN_MODEL_CONFIG

    parameter: SweepParameter
    values: list[bool | int | float | str]
    caption: str


class IndexGrid(BaseModel):
    """Represents the layout of a two-dimensional strip, in the order its stops are listed."""

    model_config = FROZEN_MODEL_CONFIG

    rows: IndexGridAxis
    columns: IndexGridAxis


class IndexPreview(BaseModel):
    """Represents the two stops a strip's collapsed card puts either side of the wipe handle."""

    model_config = FROZEN_MODEL_CONFIG

    left_index: int
    right_index: int


class IndexStrip(BaseModel):
    """Represents one strip: a parameter swept over one model and one scene.

    ``held`` is the full parameter set of the strip's first stop and ``varying`` names the
    parameter and feature fields that change across it, which is how a consumer can label
    "everything else held at" without comparing stops itself.
    """

    model_config = FROZEN_MODEL_CONFIG

    strip_id: str
    parameter: SweepParameter
    presentation: StripPresentation
    caption: str
    rationale: str
    complete: bool
    image_model: IndexModel
    scene: IndexScene
    held: GenerationParameters
    varying: list[str]
    stops: list[IndexStop]
    source_image: str | None = None
    """The image the strip starts from, relative to this index, for a consumer to show beside it."""

    source_width: int | None = None
    """The source image's width in pixels, or None when the strip starts from noise.

    Read from the published copy, which is what makes an image-to-image strip able to say that
    its stops are four times the size of what they started from.
    """

    source_height: int | None = None
    """The source image's height in pixels, or None when the strip starts from noise."""

    control_map: str | None = None
    """The map the detector drew from the strip's source, on strips held to one control type.

    None when none is published, which covers a strip that changes detector from stop to stop as
    well as one no detector has been run for yet.
    """

    grid: IndexGrid | None = None
    preview: IndexPreview | None = None
    """Which two stops to preview, resolved to positions in ``stops``.

    Present on every strip a consumer draws as a strip, and None on the grids and the tiling
    pair, which are not read by wiping one stop across another.
    """

    features: JobFeatures | None = None
    """The features the strip turns on, or None when it uses none of them."""

    batch_section: BatchSection | None = None
    """Which review-batch section this strip came from, when it came from one."""


class SweepIndex(BaseModel):
    """Represents the whole library: every strip a consumer can render."""

    model_config = FROZEN_MODEL_CONFIG

    schema_version: int = INDEX_SCHEMA_VERSION
    seed: int
    strips: list[IndexStrip] = Field(default_factory=list)


def save_render(
    *,
    images: Sequence[Image.Image],
    job: RenderJob,
    output_root: Path,
    runner_name: str,
    started_at: datetime,
    finished_at: datetime,
    model_name_used: str | None = None,
    worker_note: str | None = None,
) -> list[Path]:
    """Write every candidate a job rendered as WebP with its JSON sidecar, and return their paths.

    A job that renders once writes one file under the plain stem it always had. A curation batch
    writes every candidate it was given, each with its own sidecar saying which candidate it is
    and whether it is the one the index publishes, so the choice a human makes later is a change
    to one field in the specification rather than a re-render.

    Args:
        images: The rendered candidates, in the order the backend produced them.
        job: The [`RenderJob`][param_sweeps.spec.RenderJob] that produced them.
        output_root: The root of the output tree.
        runner_name: Which runner rendered them, recorded in every sidecar.
        started_at: When the render started.
        finished_at: When the render finished.
        model_name_used: The model name the runner actually generated with, when it can report
            one (the API can hand a job to a worker running a different model than requested).
        worker_note: Any runner-specific note worth keeping, such as a fault the backend reported.

    Returns:
        The paths the images were written to, in candidate order.

    Raises:
        ValueError: If no images were given at all.

    Side Effects:
        Creates the strip's directory if it does not exist, and overwrites any existing images
        and sidecars at the job's deterministic paths.
    """
    if not images:
        raise ValueError(f"no images to write for {job.relative_image_path}")

    written_paths: list[Path] = []
    for candidate_number, image in enumerate(images, start=1):
        image_path = output_root / job.relative_candidate_image_path(candidate_number)
        sidecar_path = output_root / job.relative_candidate_sidecar_path(candidate_number)
        image_path.parent.mkdir(parents=True, exist_ok=True)

        image.save(image_path, format="WEBP", quality=WebpEncoding.QUALITY, method=WebpEncoding.METHOD)

        sidecar = ImageSidecar(
            parameter=job.parameter,
            model_slug=job.model_slug,
            model_requested=job.horde_model_name,
            model_generated_with=model_name_used if model_name_used is not None else job.horde_model_name,
            scene=job.scene_id,
            prompt=job.prompt,
            stop_index=job.stop_index,
            stop_value=job.stop_value,
            caption=job.caption,
            params=job.generation_parameters(),
            features=job.features(),
            runner=runner_name,
            started_at=started_at.isoformat(),
            finished_at=finished_at.isoformat(),
            image=image_path.name,
            candidate_number=candidate_number,
            is_chosen=candidate_number == job.chosen_candidate_number,
            source_image=job.source_image,
            note=worker_note,
        )
        sidecar_path.write_text(sidecar.model_dump_json(indent=2) + "\n", encoding="utf-8")
        written_paths.append(image_path)

    return written_paths


def build_index(*, spec: SweepSpec, output_root: Path) -> SweepIndex:
    """Build the index describing every strip whose images are present on disk.

    A stop with no image on disk is left out and its strip is marked incomplete, so a partially
    rendered campaign still produces an index a consumer can use without hitting missing files.

    Args:
        spec: The specification the library was rendered from.
        output_root: The root of the output tree.

    Returns:
        The index.
    """
    index_strips: list[IndexStrip] = []

    for strip in spec.expand_strips():
        stops: list[IndexStop] = []
        missing_count = 0

        for job in strip.jobs:
            frame_path = output_root / job.relative_image_path
            if not frame_path.is_file():
                missing_count += 1
                continue

            frame_width, frame_height = image_size(frame_path)
            stops.append(
                IndexStop(
                    value=job.stop_value,
                    label=job.stop_label,
                    caption=job.caption,
                    file=job.relative_image_path,
                    sidecar=job.relative_sidecar_path,
                    width=frame_width,
                    height=frame_height,
                    crop=_published_stop_crop(job=job, output_root=output_root),
                    fit=_published_fit(job=job, output_root=output_root),
                    prompt_fragment=job.prompt_fragment,
                    batch_size=job.batch_size,
                    chosen_candidate=job.chosen_candidate,
                    control_type=job.control_type,
                    control_group=job.control_group,
                    control_map=_published_control_map(job=job, output_root=output_root),
                ),
            )

        if not stops:
            continue

        source_width, source_height = _published_source_size(strip=strip, output_root=output_root)
        index_strips.append(
            IndexStrip(
                strip_id=strip.strip_id,
                parameter=strip.parameter,
                presentation=strip.presentation,
                caption=strip.caption,
                rationale=strip.rationale,
                complete=missing_count == 0,
                image_model=IndexModel(
                    slug=strip.image_model.slug,
                    name=strip.image_model.horde_name,
                    baseline=strip.image_model.baseline,
                    note=strip.image_model.note,
                ),
                scene=IndexScene(
                    scene_id=strip.scene.scene_id,
                    label=strip.scene.label,
                    prompt=strip.scene.prompt,
                ),
                held=strip.jobs[0].generation_parameters(),
                varying=_varying_field_names(strip),
                stops=stops,
                source_image=strip.published_source_image,
                source_width=source_width,
                source_height=source_height,
                control_map=_published_strip_control_map(strip=strip, output_root=output_root),
                grid=_index_grid(strip),
                preview=_index_preview(strip=strip, stops=stops),
                features=_index_features(strip),
                batch_section=strip.batch_section,
            ),
        )

    return SweepIndex(seed=spec.seed, strips=index_strips)


def write_index(*, spec: SweepSpec, output_root: Path) -> Path:
    """Write the index to the root of the output tree, and return its path.

    Side Effects:
        Overwrites any existing index, and publishes the source image of every strip that
        starts from one. The content depends only on the specification and the files present,
        so rewriting it never introduces a spurious change.
    """
    index_path = output_root / INDEX_FILENAME
    index_path.parent.mkdir(parents=True, exist_ok=True)
    publish_source_images(spec=spec, output_root=output_root)
    index = build_index(spec=spec, output_root=output_root)
    index_path.write_text(index.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return index_path


def publish_source_images(*, spec: SweepSpec, output_root: Path) -> list[Path]:
    """Copy each image-to-image sweep's source into the output tree, and return what was written.

    The library has to be self-contained: a consumer showing an image-to-image strip needs the
    image it started from, and cannot be expected to reach back into this tool's inputs for it.

    Args:
        spec: The specification the library was rendered from.
        output_root: The root of the output tree.

    Returns:
        The published paths, one per sweep that declares a source image.

    Side Effects:
        Overwrites any source image already published at those paths.
    """
    published: list[Path] = []

    for strip in spec.expand_strips():
        published_relative_path = strip.published_source_image
        if published_relative_path is None or strip.source_image is None:
            continue

        destination = output_root / published_relative_path
        if destination in published:
            continue

        source = TOOL_ROOT / strip.source_image
        if not source.exists():
            # A dependent job's source is staged only after the render it names
            # exists, so an unstaged source is an incomplete strip rather than
            # an error: the strip already reports itself incomplete through its
            # missing stops, and the source follows on the next run.
            logger.info("Source not yet staged, skipping publish: %s", strip.source_image)
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        published.append(destination)

    return published


def image_size(path: Path) -> tuple[int, int]:
    """Return an image's pixel size, read from its header rather than by decoding it.

    Raises:
        OSError: If the file is missing, or is not an image this build of Pillow can read.
    """
    with Image.open(path) as opened:
        return opened.width, opened.height


def published_crop(*, file: str, root: Path, region: CropRegion, frame_relative_path: str) -> PublishedCrop | None:
    """Return the record of a crop on disk, or None when the crop or its frame is not there.

    Read from the files rather than computed from the specification alone, so what the index and
    the manifest report is the crop that exists rather than the one a run meant to write.

    Args:
        file: How the index or manifest names the crop, relative to itself.
        root: What both paths are relative to.
        region: The region the crop was cut from.
        frame_relative_path: The frame it was cut from, whose pixels the box is stated in.

    Returns:
        The record, or None when either file is missing.
    """
    crop_path = root / file
    frame_path = root / frame_relative_path
    if not crop_path.is_file() or not frame_path.is_file():
        return None

    frame_width, frame_height = image_size(frame_path)
    left, upper, _right, _lower = region.pixel_box(width=frame_width, height=frame_height)
    crop_width, crop_height = image_size(crop_path)
    return PublishedCrop(file=file, width=crop_width, height=crop_height, x=left, y=upper)


def _published_stop_crop(*, job: RenderJob, output_root: Path) -> PublishedCrop | None:
    """Return the stop's crop record when the crop is on disk, and None when it is not."""
    relative_path = job.relative_crop_path
    if relative_path is None or job.crop is None:
        return None

    return published_crop(
        file=relative_path,
        root=output_root,
        region=job.crop,
        frame_relative_path=job.relative_image_path,
    )


def _published_fit(*, job: RenderJob, output_root: Path) -> str | None:
    """Return the stop's bounded copy when it is on disk, and None when the frame needed none."""
    relative_path = job.relative_fit_path
    return relative_path if (output_root / relative_path).is_file() else None


def _published_source_size(*, strip: RenderStrip, output_root: Path) -> tuple[int | None, int | None]:
    """Return the published source image's size, or a pair of Nones when there is no source."""
    relative_path = strip.published_source_image
    if relative_path is None or not (output_root / relative_path).is_file():
        return None, None

    return image_size(output_root / relative_path)


def _published_control_map(*, job: RenderJob, output_root: Path) -> str | None:
    """Return the job's control map path when the map is on disk, and None when it is not."""
    relative_path = job.relative_control_map_path
    if relative_path is None or not (output_root / relative_path).is_file():
        return None
    return relative_path


def _published_strip_control_map(*, strip: RenderStrip, output_root: Path) -> str | None:
    """Return the strip's control map path when the map is on disk, and None when it is not."""
    relative_path = strip.relative_control_map_path
    if relative_path is None or not (output_root / relative_path).is_file():
        return None
    return relative_path


def _varying_field_names(strip: RenderStrip) -> list[str]:
    """Return the parameter and feature fields that change across the strip, in declaration order.

    Features are named here alongside the parameters because the two are one question for a
    consumer: what is this strip changing from stop to stop. A QR strip that sweeps its
    ControlNet's strength changes a feature and nothing else, and would otherwise report that it
    varies nothing at all.
    """
    if not strip.jobs:
        return []

    first_parameters = strip.jobs[0].generation_parameters()
    varying: set[str] = set()
    for job in strip.jobs[1:]:
        varying |= first_parameters.differing_field_names(job.generation_parameters())

    parameter_names = [field_name for field_name in GenerationParameters.model_fields if field_name in varying]
    return parameter_names + _varying_feature_names(strip)


def _varying_feature_names(strip: RenderStrip) -> list[str]:
    """Return the feature fields that change across the strip, in declaration order."""
    if not strip.jobs:
        return []

    first_features = strip.jobs[0].features().model_dump()
    varying = {
        field_name
        for job in strip.jobs[1:]
        for field_name, value in job.features().model_dump().items()
        if value != first_features[field_name]
    }
    return [field_name for field_name in JobFeatures.model_fields if field_name in varying]


def _index_features(strip: RenderStrip) -> JobFeatures | None:
    """Return the features the strip holds across every stop, or None when it turns none on.

    A feature the strip varies is reported at its default here rather than at the first stop's
    value, and its name is in ``varying`` instead: this block is what a consumer labels "held at",
    so a strip that sweeps a ControlNet's strength must not appear to hold it at its lowest. The
    per-stop truth stays in the sidecars, as it does for the parameters.
    """
    if not strip.jobs:
        return None

    features = strip.jobs[0].features()
    varying = _varying_feature_names(strip)
    if varying:
        features = features.model_copy(
            update={
                field_name: JobFeatures.model_fields[field_name].get_default(call_default_factory=True)
                for field_name in varying
            },
        )

    return None if features.is_plain else features


def _index_grid(strip: RenderStrip) -> IndexGrid | None:
    """Return the strip's grid layout, or None when it is one-dimensional."""
    if strip.grid_rows is None or strip.grid_columns is None:
        return None

    return IndexGrid(
        rows=IndexGridAxis(
            parameter=strip.grid_rows.parameter,
            values=list(strip.grid_rows.values),
            caption=strip.grid_rows.caption,
        ),
        columns=IndexGridAxis(
            parameter=strip.grid_columns.parameter,
            values=list(strip.grid_columns.values),
            caption=strip.grid_columns.caption,
        ),
    )


def _index_preview(*, strip: RenderStrip, stops: Sequence[IndexStop]) -> IndexPreview | None:
    """Return the pair of stops the strip's collapsed card previews, or None where it has no pair.

    Resolved against the stops that were actually published rather than the ladder the
    specification declares, so a strip missing an image still names two pictures that exist.
    """
    if strip.presentation not in PREVIEWABLE_PRESENTATIONS:
        return None

    left_index, right_index = resolve_preview_indices(
        preview=strip.preview,
        stop_values=[stop.value for stop in stops],
        default_value=strip.default_value,
    )
    return IndexPreview(left_index=left_index, right_index=right_index)
