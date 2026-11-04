# SPDX-FileCopyrightText: 2026 Tazlin
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Write the rendered images, their sidecars, and the index that a web app uses to draw the strips.

Every image is written as a quality-tuned WebP with a JSON sidecar beside it recording the
exact parameters, model, seed, runner and timestamps that produced it. The index collects the
whole library into one file: a consumer that has ``index.json`` needs nothing else to draw a
scrubber strip, because each strip states what it holds and what varies, and each stop has
its value, caption and filename.

Both on-disk shapes are declared as models rather than assembled as mappings, so the format that
this tool offers its consumers is defined in one place and checked when it is written.

Critical public members:

- [`ImageSidecar`][param_sweeps.outputs.ImageSidecar]: the per-image record.
- [`SweepIndex`][param_sweeps.outputs.SweepIndex]: the whole library's index.
- [`save_render`][param_sweeps.outputs.save_render]: write one image and its sidecar.
- [`build_index`][param_sweeps.outputs.build_index]: build the index from the specification and
  the files actually present on disk.
- [`image_size`][param_sweeps.outputs.image_size]: the pixel size of a published file, read
  without decoding it.
- [`write_index`][param_sweeps.outputs.write_index]: write that index deterministically.
- [`publish_source_images`][param_sweeps.outputs.publish_source_images]: copy the source images
  of the image-to-image strips into the library, so it stays self-contained.
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
    SweepSubject,
    resolve_preview_indices,
)

logger = logging.getLogger(__name__)

INDEX_FILENAME = "index.json"
"""The name of the top-level index, at the root of the output tree."""

INDEX_SCHEMA_VERSION = 1
"""The index layout that consumers can expect."""

FROZEN_MODEL_CONFIG = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())


class WebpEncoding:
    """The WebP encoder settings used for every image in the library.

    Quality 92 with the slowest encoder method keeps sampler noise and fine texture intact,
    because several strips show differences that are only in that texture. Lossless would
    roughly triple the size of the library without a visible gain at these sizes.
    """

    QUALITY = 92
    METHOD = 6

    CROP_QUALITY = 98
    """The encoder quality for a hundred-percent crop.

    A crop is viewed at its own pixels, so it is encoded with much less loss than a full frame,
    which is not viewed as closely. The crops are a small part of the library by area, so the
    higher quality costs little.
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
    """Whether this is the candidate that the index publishes."""

    source_image: str | None = None
    note: str | None = None


class IndexModel(BaseModel):
    """Represents a strip's model, as reported in the index."""

    model_config = FROZEN_MODEL_CONFIG

    slug: str
    name: str
    baseline: str
    note: str


class IndexScene(BaseModel):
    """Represents a strip's scene, as reported in the index."""

    model_config = FROZEN_MODEL_CONFIG

    scene_id: str
    label: str
    prompt: str


class IndexStop(BaseModel):
    """Represents one stop of a strip, with everything needed to render and caption it.

    The stop's resolved parameters are deliberately not repeated here. The strip states what it
    holds and what varies, and captions are written against the stop's value. The full set is in
    the sidecar, and including it in every stop would triple the size of the index.
    """

    model_config = FROZEN_MODEL_CONFIG

    value: bool | int | float | str
    label: str
    caption: str
    file: str
    sidecar: str
    width: int
    """The published frame's width in pixels, read from the file itself.

    A consumer that has the size can state what it is showing and can reserve the space before
    the picture arrives. It is read from the file rather than taken from the request, because a
    backend can return a frame at a size that was not requested, and the file is what a reader sees.
    """

    height: int
    """The published frame's height in pixels, read from the file itself."""

    prompt_fragment: str | None = None
    """What this stop added to the scene's prompt, on the strips that vary the prompt itself."""

    batch_size: int = 1
    """How many candidates were rendered for this stop, so a consumer can disclose the curation.

    One means that the request produced only this image. Above one, a human picked this image
    out of that many. Disclosing that separates what the model does from what a careful operator
    can get it to do.
    """

    chosen_candidate: int | None = None
    """Which candidate was published, one-based, when more than one was rendered."""

    control_type: str | None = None
    """The ControlNet guidance mode for this stop, if it has one."""

    control_group: str | None = None
    """The detector family of that guidance mode, for a consumer to group by."""

    control_map: str | None = None
    """The detector's map, relative to this index, or None when no map was published.

    None on a stop with a control type means that the detector did not draw a map. The render is
    still present, guided by the map that the backend derived internally, and a consumer should
    report the missing map instead of hiding it.
    """

    crop: PublishedCrop | None = None
    """The part of this frame published at its own pixels, or None when there is no crop.

    Present only when the file is on disk, so a consumer that has the record can fetch it.
    """

    fit: str | None = None
    """A copy of this frame bounded to a page-sized long edge, or None when the frame is already
    within it.

    A consumer drawing the strip fetches this, and fetches the full frame only when a reader
    requests it. Without these copies, a strip of full frames can be too heavy to load.
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
    """Represents the two stops that a strip's collapsed card puts on either side of the wipe handle."""

    model_config = FROZEN_MODEL_CONFIG

    left_index: int
    right_index: int


class IndexStrip(BaseModel):
    """Represents one strip: a parameter swept over one model and one scene.

    ``held`` is the full parameter set of the strip's first stop, and ``varying`` lists the
    parameter and feature fields that change across the strip. A consumer can then label
    "everything else held at" without comparing stops.
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
    """The image that the strip starts from, relative to this index, for a consumer to show beside it."""

    source_width: int | None = None
    """The source image's width in pixels, or None when the strip starts from noise.

    Read from the published copy, so an image-to-image strip can state that its stops are four
    times the size of its source.
    """

    source_height: int | None = None
    """The source image's height in pixels, or None when the strip starts from noise."""

    subject: SweepSubject | None = None
    """The kind of source picture for this strip, when several strips share a lesson.

    The ControlNet gallery runs every detector on a world's anchor picture, on a building and on
    a flat-colour drawing, and a consumer uses this field to offer the three as alternatives.
    None on a strip without such companions.
    """

    control_map: str | None = None
    """The detector's map of the strip's source, on strips with a single control type.

    None when no map is published. That includes a strip that changes detector from stop to
    stop, and a strip whose detector has not run yet.
    """

    grid: IndexGrid | None = None
    preview: IndexPreview | None = None
    """Which two stops to preview, resolved to positions in ``stops``.

    Set on every strip that a consumer draws as a strip. None on the grids and the tiling pair,
    which are not viewed by wiping one stop across another.
    """

    features: JobFeatures | None = None
    """The features that the strip turns on, or None when it does not use any."""

    batch_section: BatchSection | None = None
    """The strip's review-batch section, if it has one."""


class SweepIndex(BaseModel):
    """Represents the whole library: every strip that a consumer can render."""

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
    """Write every candidate that a job rendered as WebP with its JSON sidecar, and return their paths.

    A job that renders one image writes one file under its plain stem. A curation batch writes
    every candidate that it was given, each with a sidecar that records its candidate number and
    whether the index publishes it. A human's later choice is then a change to one field in the
    specification instead of a re-render.

    Args:
        images: The rendered candidates, in the backend's order.
        job: The [`RenderJob`][param_sweeps.spec.RenderJob] that produced them.
        output_root: The root of the output tree.
        runner_name: Which runner rendered them, recorded in every sidecar.
        started_at: When the render started.
        finished_at: When the render finished.
        model_name_used: The model name the runner actually generated with, when it can report
            one (the API can hand a job to a worker running a different model than requested).
        worker_note: Any runner-specific note worth keeping, such as a fault the backend reported.

    Returns:
        The written image paths, in candidate order.

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
    rendered campaign still produces an index that a consumer can use without hitting missing files.

    Args:
        spec: The library's specification.
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
                subject=strip.subject,
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
        so rewriting it does not introduce spurious changes.
    """
    index_path = output_root / INDEX_FILENAME
    index_path.parent.mkdir(parents=True, exist_ok=True)
    publish_source_images(spec=spec, output_root=output_root)
    index = build_index(spec=spec, output_root=output_root)
    index_path.write_text(index.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return index_path


def publish_source_images(*, spec: SweepSpec, output_root: Path) -> list[Path]:
    """Copy each image-to-image sweep's source into the output tree, and return the written paths.

    The library has to be self-contained. A consumer showing an image-to-image strip needs its
    source image, and cannot read this tool's inputs.

    Args:
        spec: The library's specification.
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
            # A dependent job's source is staged only after the render that it depends on
            # exists, so an unstaged source means an incomplete strip, not an error. The
            # missing stops already mark the strip incomplete, and the source is published
            # on a later run.
            logger.info("Source not yet staged, skipping publish: %s", strip.source_image)
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        published.append(destination)

    return published


def image_size(path: Path) -> tuple[int, int]:
    """Return an image's pixel size, read from its header rather than by decoding it.

    Raises:
        OSError: If the file is missing, or is not an image that this build of Pillow can read.
    """
    with Image.open(path) as opened:
        return opened.width, opened.height


def published_crop(*, file: str, root: Path, region: CropRegion, frame_relative_path: str) -> PublishedCrop | None:
    """Return the record of a crop on disk, or None when the crop or its frame is not there.

    Read from the files rather than computed from the specification alone, so the index and the
    manifest report the crop that exists, not the one that a run intended to write.

    Args:
        file: The crop's path as written in the index or manifest, relative to that file.
        root: The directory that both paths are relative to.
        region: The crop's region.
        frame_relative_path: The frame that the crop was cut from. The box is stated in this
            frame's pixels.

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
    """Return the stop's bounded copy when it is on disk, or None when the frame did not need one."""
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

    Features are listed with the parameters because a consumer needs both to know what changes
    from stop to stop. A QR strip that sweeps its ControlNet's strength changes only a feature,
    and would otherwise report that nothing varies.
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
    """Return the features that the strip holds across every stop, or None when it does not turn any on.

    A feature that the strip varies is reported at its default here instead of at the first
    stop's value, and its name is in ``varying``. A consumer labels this block "held at", so a
    strip that sweeps a ControlNet's strength must not appear to hold it at its lowest value. The
    per-stop values stay in the sidecars, as they do for the parameters.
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
    """Return the pair of stops for the strip's collapsed card preview, or None when it has no pair.

    Resolved against the published stops instead of the full list in the specification, so a
    strip with a missing image still refers to two pictures that exist.
    """
    if strip.presentation not in PREVIEWABLE_PRESENTATIONS:
        return None

    left_index, right_index = resolve_preview_indices(
        preview=strip.preview,
        stop_values=[stop.value for stop in stops],
        default_value=strip.default_value,
    )
    return IndexPreview(left_index=left_index, right_index=right_index)
