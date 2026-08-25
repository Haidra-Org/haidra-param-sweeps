# SPDX-FileCopyrightText: 2026 Tazlin
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cut the crops and write the bounded copies a page needs to show what a render actually holds.

A page draws a picture at the width it has, and a 512-pixel render and a 2048-pixel upscale of it
look identical once both are scaled to fit. That makes the evidence for upscalers, the hires pass
and face restoration unreadable: the whole difference lives in pixels the reader never sees. Two
files beside each render answer it.

A crop is the part of the frame the strip is actually about, cut at the frame's own pixels, so a
reader looking at it is looking at what the model produced rather than at a resampling of it. The
region is stated in fractions of the frame, which is what lets one region serve stops published at
different sizes.

A bounded copy is the whole frame with its long edge brought down to
[`FIT_LONG_EDGE_PIXELS`][param_sweeps.derivatives.FIT_LONG_EDGE_PIXELS], which is what a page can
draw without fetching several megabytes to throw most of them away. The full frame stays published
beside it for anyone who wants it.

Both are cut here rather than while the index is built, so that building the index stays a pure
read of the specification and the files on disk.

Critical public members:

- [`DerivativeOutcome`][param_sweeps.derivatives.DerivativeOutcome]: what became of one file.
- [`write_sweep_derivatives`][param_sweeps.derivatives.write_sweep_derivatives]: the crops and
  bounded copies every published render needs.
- [`write_alchemy_crops`][param_sweeps.derivatives.write_alchemy_crops]: the crops the alchemy
  entries that declare a region need, cut from both the source and the result.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import auto
from pathlib import Path

from PIL import Image
from pydantic import BaseModel
from strenum import StrEnum

from param_sweeps.alchemy import (
    RESULT_IMAGE_FILENAME,
    AlchemySpec,
    alchemy_root,
    published_result_crop_name,
    published_source_crop_name,
    published_source_name,
)
from param_sweeps.outputs import WebpEncoding, image_size
from param_sweeps.spec import FROZEN_MODEL_CONFIG, CropRegion, SweepParameter, SweepSpec

MINIMUM_CROPPABLE_PIXELS = 256
"""The smallest side a frame can have and still be worth cutting a crop from.

Below this a page draws the frame at or near its own pixels already, so a crop of it shows the
reader nothing they were not being shown. A region that carries something anyway says so with
[`CropRegion.even_when_small`][param_sweeps.spec.CropRegion.even_when_small].
"""

FIT_LONG_EDGE_PIXELS = 1024
"""What a bounded copy's long edge is brought down to. Frames at or under it get no copy."""


class DerivativeKind(StrEnum):
    """What kind of file a derivative is, which is what a report groups by."""

    CROP = auto()
    FIT = auto()


class DerivativeOutcome(BaseModel):
    """Represents what became of one crop or bounded copy."""

    model_config = FROZEN_MODEL_CONFIG

    kind: DerivativeKind
    relative_path: str
    """Where the file is published, relative to the root it was asked for under."""

    written: bool
    """Whether this run wrote it, as opposed to skipping it or leaving one already there alone."""

    reused: bool = False
    """Whether a file already on disk was left alone."""

    skipped_reason: str | None = None
    """Why nothing was written, when nothing was."""

    width: int = 0
    height: int = 0
    size_bytes: int = 0


def write_sweep_derivatives(
    *,
    spec: SweepSpec,
    output_root: Path,
    parameters: Sequence[SweepParameter] | None = None,
    force: bool = False,
) -> list[DerivativeOutcome]:
    """Cut every crop and write every bounded copy the published renders need.

    A render with no image on disk is passed over rather than reported: an unrendered stop is
    already reported as an incomplete strip by the index, and saying it twice helps nobody.

    Args:
        spec: The specification to read the jobs from.
        output_root: The root of the output tree, which the files land inside.
        parameters: Restrict to these parameters, or None for every one of them.
        force: Rewrite files that are already on disk.

    Returns:
        One outcome per file considered, in specification order.

    Side Effects:
        Writes a WebP beside each published render that declares a region or exceeds the bound.
    """
    wanted = set(parameters) if parameters else None
    outcomes: list[DerivativeOutcome] = []

    for job in spec.expand_jobs():
        if wanted is not None and job.parameter not in wanted:
            continue

        frame_path = output_root / job.relative_image_path
        if not frame_path.is_file():
            continue

        crop_path = job.relative_crop_path
        if crop_path is not None and job.crop is not None:
            outcomes.append(
                _cut_one_crop(
                    frame_path=frame_path,
                    destination=output_root / crop_path,
                    relative_path=crop_path,
                    region=job.crop,
                    force=force,
                ),
            )

        outcomes.append(
            _write_one_bounded_copy(
                frame_path=frame_path,
                destination=output_root / job.relative_fit_path,
                relative_path=job.relative_fit_path,
                force=force,
            ),
        )

    return outcomes


def write_alchemy_crops(
    *,
    spec: AlchemySpec,
    output_root: Path,
    force: bool = False,
) -> list[DerivativeOutcome]:
    """Cut the crops the alchemy entries that declare a region need, and return what became of them.

    Both sides are cut: what a restorer or an upscaler did is a comparison, and a reader holding
    the result at its own pixels with nothing to hold it against learns only that the picture is
    detailed. The same fractional region serves both, which is what makes the pair comparable
    when the result is four times the size of the source.

    Args:
        spec: The alchemy specification.
        output_root: The root of the sweep library, which the alchemy tree sits inside.
        force: Rewrite crops that are already on disk.

    Returns:
        One outcome per crop considered, in specification order, the source's first.

    Side Effects:
        Writes WebPs inside the entry directories under the alchemy tree.
    """
    root = alchemy_root(output_root)
    outcomes: list[DerivativeOutcome] = []

    for entry in spec.entries:
        if entry.crop is None:
            continue

        for frame_name, crop_name in (
            (published_source_name(entry), published_source_crop_name(entry)),
            (f"{entry.relative_directory}/{RESULT_IMAGE_FILENAME}", published_result_crop_name(entry)),
        ):
            frame_path = root / frame_name
            if not frame_path.is_file():
                continue

            outcomes.append(
                _cut_one_crop(
                    frame_path=frame_path,
                    destination=root / crop_name,
                    relative_path=crop_name,
                    region=entry.crop,
                    force=force,
                ),
            )

    return outcomes


def _cut_one_crop(
    *,
    frame_path: Path,
    destination: Path,
    relative_path: str,
    region: CropRegion,
    force: bool,
) -> DerivativeOutcome:
    """Cut one region out of one frame at its own pixels, and return what became of it."""
    if destination.is_file() and not force:
        width, height = image_size(destination)
        return DerivativeOutcome(
            kind=DerivativeKind.CROP,
            relative_path=relative_path,
            written=False,
            reused=True,
            width=width,
            height=height,
            size_bytes=destination.stat().st_size,
        )

    with Image.open(frame_path) as opened:
        frame_width, frame_height = opened.width, opened.height
        if min(frame_width, frame_height) < MINIMUM_CROPPABLE_PIXELS and not region.even_when_small:
            return DerivativeOutcome(
                kind=DerivativeKind.CROP,
                relative_path=relative_path,
                written=False,
                skipped_reason=(
                    f"the frame is {frame_width}x{frame_height}, under the {MINIMUM_CROPPABLE_PIXELS}"
                    f" pixels a crop is worth cutting from"
                ),
            )

        cut = opened.convert("RGB").crop(region.pixel_box(width=frame_width, height=frame_height))

    destination.parent.mkdir(parents=True, exist_ok=True)
    cut.save(destination, format="WEBP", quality=WebpEncoding.CROP_QUALITY, method=WebpEncoding.METHOD)

    return DerivativeOutcome(
        kind=DerivativeKind.CROP,
        relative_path=relative_path,
        written=True,
        width=cut.width,
        height=cut.height,
        size_bytes=destination.stat().st_size,
    )


def _write_one_bounded_copy(
    *,
    frame_path: Path,
    destination: Path,
    relative_path: str,
    force: bool,
) -> DerivativeOutcome:
    """Write one frame down to the bound if it is over it, and return what became of it."""
    if destination.is_file() and not force:
        width, height = image_size(destination)
        return DerivativeOutcome(
            kind=DerivativeKind.FIT,
            relative_path=relative_path,
            written=False,
            reused=True,
            width=width,
            height=height,
            size_bytes=destination.stat().st_size,
        )

    with Image.open(frame_path) as opened:
        frame_width, frame_height = opened.width, opened.height
        if max(frame_width, frame_height) <= FIT_LONG_EDGE_PIXELS:
            return DerivativeOutcome(
                kind=DerivativeKind.FIT,
                relative_path=relative_path,
                written=False,
                skipped_reason=(
                    f"the frame is {frame_width}x{frame_height}, already within"
                    f" {FIT_LONG_EDGE_PIXELS} pixels on its long edge"
                ),
            )

        scale = FIT_LONG_EDGE_PIXELS / max(frame_width, frame_height)
        bounded = opened.convert("RGB").resize(
            (max(1, round(frame_width * scale)), max(1, round(frame_height * scale))),
            resample=Image.Resampling.LANCZOS,
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    bounded.save(destination, format="WEBP", quality=WebpEncoding.QUALITY, method=WebpEncoding.METHOD)

    return DerivativeOutcome(
        kind=DerivativeKind.FIT,
        relative_path=relative_path,
        written=True,
        width=bounded.width,
        height=bounded.height,
        size_bytes=destination.stat().st_size,
    )
