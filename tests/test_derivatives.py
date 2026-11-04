# SPDX-FileCopyrightText: 2026 Tazlin
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Check what a page needs to read a render: its size, the crop, and the bounded copy.

These files exist for detail that is not visible at a page's display size, so the checks here are
about pixels rather than paths: that a region covers the same part of two differently sized
frames, that the cut box matches the region, and that the index reports only files on disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image, ImageStat

from param_sweeps.alchemy import (
    RESULT_IMAGE_FILENAME,
    RESULT_TEXT_FILENAME,
    AlchemySpec,
    alchemy_root,
    build_manifest,
    published_result_crop_name,
    published_result_fit_name,
    published_source_crop_name,
    published_source_name,
)
from param_sweeps.derivatives import (
    FIT_LONG_EDGE_PIXELS,
    MINIMUM_CROPPABLE_PIXELS,
    DerivativeKind,
    write_alchemy_crops,
    write_alchemy_result_fits,
    write_sweep_derivatives,
)
from param_sweeps.outputs import build_index
from param_sweeps.spec import CropRegion, SweepSpec

TOOL_ROOT = Path(__file__).resolve().parents[1]

REGION = {"x": 0.25, "y": 0.5, "w": 0.25, "h": 0.25}
"""A quarter-sized region in the lower left quadrant, which rounding cannot shift."""

MARKED_COLOUR = (255, 0, 0)
"""The region's colour in a test frame, so a misplaced cut has the wrong colour."""

BACKGROUND_COLOUR = (0, 0, 255)


def one_sweep_spec(sweep: dict[str, Any]) -> dict[str, Any]:
    """Return the smallest valid specification, with one sweep."""
    return {
        "schema_version": 1,
        "seed": 1234567890,
        "anchor": {
            "sampler_name": "k_euler_a",
            "cfg_scale": 7.5,
            "steps": 30,
            "karras": True,
            "denoising_strength": 1.0,
            "hires_fix": False,
            "tiling": False,
        },
        "models": [
            {
                "slug": "small",
                "horde_name": "stable_diffusion",
                "baseline": "stable_diffusion_1",
                "native_width": 512,
                "native_height": 512,
                "anchor_clip_skip": 1,
                "approximate_download_size_gb": 4.3,
                "note": "A model to hang a rule on.",
            },
        ],
        "scenes": [
            {
                "scene_id": "thing",
                "label": "A thing",
                "prompt": "a thing",
                "derived_from": "nothing",
            },
        ],
        "sweeps": [sweep],
    }


def steps_sweep(*, crop: dict[str, Any] | None = None, stop_crop: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a two-stop strip, optionally cropping the whole sweep or only its second stop."""
    sweep: dict[str, Any] = {
        "kind": "strip",
        "parameter": "steps",
        "rationale": "Something for the frames to belong to.",
        "caption": "Two step counts.",
        "model_slugs": ["small"],
        "scene_ids": ["thing"],
        "presentation": "strip",
        "stops": [
            {"value": 4, "steps": 4, "caption": "Few."},
            {"value": 30, "steps": 30, "caption": "Enough."},
        ],
    }
    if crop is not None:
        sweep["crop"] = crop
    if stop_crop is not None:
        sweep["stops"][1]["crop"] = stop_crop
    return sweep


def marked_frame(size: tuple[int, int], *, region: CropRegion) -> Image.Image:
    """Return a frame holding one colour inside the region and another everywhere else."""
    width, height = size
    left, upper, right, lower = region.pixel_box(width=width, height=height)
    frame = Image.new("RGB", size, color=BACKGROUND_COLOUR)
    frame.paste(Image.new("RGB", (right - left, lower - upper), color=MARKED_COLOUR), (left, upper))
    return frame


def render_frames(*, spec: SweepSpec, output_root: Path, size: tuple[int, int]) -> None:
    """Write a frame and a sidecar for every job that the specification describes."""
    for job in spec.expand_jobs():
        image_path = output_root / job.relative_image_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        region = job.crop if job.crop is not None else CropRegion.model_validate(REGION)
        marked_frame(size, region=region).save(image_path, format="WEBP", quality=95, method=0)
        (output_root / job.relative_sidecar_path).write_text("{}\n", encoding="utf-8")


def test_a_region_covers_the_same_part_of_two_output_sizes() -> None:
    """Assert one region resolves to proportional boxes, because it is stated in fractions."""
    region = CropRegion.model_validate(REGION)

    assert region.pixel_box(width=1024, height=1024) == (256, 512, 512, 768)
    assert region.pixel_box(width=2048, height=2048) == (512, 1024, 1024, 1536)


def test_a_region_has_to_be_inside_the_frame() -> None:
    with pytest.raises(ValueError, match="extends past it"):
        CropRegion.model_validate({"x": 0.8, "y": 0.1, "w": 0.4, "h": 0.2})


def test_the_index_reports_the_pixel_size_of_every_published_frame(tmp_path: Path) -> None:
    """Assert the sizes come from the files rather than from what the jobs requested."""
    spec = SweepSpec.model_validate(one_sweep_spec(steps_sweep()))
    render_frames(spec=spec, output_root=tmp_path, size=(640, 384))

    (strip,) = build_index(spec=spec, output_root=tmp_path).strips

    assert [(stop.width, stop.height) for stop in strip.stops] == [(640, 384), (640, 384)]
    assert all(stop.crop is None and stop.fit is None for stop in strip.stops)


def test_the_index_reports_the_size_of_a_strip_s_source_image(tmp_path: Path) -> None:
    sweep = steps_sweep()
    sweep["source_image"] = "inputs/tutorial-anchor-512.webp"
    spec = SweepSpec.model_validate(one_sweep_spec(sweep))
    render_frames(spec=spec, output_root=tmp_path, size=(512, 512))

    (strip,) = spec.expand_strips()
    published_source = strip.published_source_image
    assert published_source is not None
    (tmp_path / published_source).parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (512, 256), color=BACKGROUND_COLOUR).save(tmp_path / published_source, format="WEBP")

    (indexed,) = build_index(spec=spec, output_root=tmp_path).strips

    assert (indexed.source_width, indexed.source_height) == (512, 256)


def test_the_cutter_writes_the_pixel_box_of_the_region(tmp_path: Path) -> None:
    """Assert the crop holds exactly the marked region, at the frame's own pixels."""
    spec = SweepSpec.model_validate(one_sweep_spec(steps_sweep(crop=REGION)))
    render_frames(spec=spec, output_root=tmp_path, size=(1024, 512))

    outcomes = write_sweep_derivatives(spec=spec, output_root=tmp_path)

    crops = [outcome for outcome in outcomes if outcome.kind is DerivativeKind.CROP]
    assert [outcome.written for outcome in crops] == [True, True]
    assert [(outcome.width, outcome.height) for outcome in crops] == [(256, 128), (256, 128)]

    with Image.open(tmp_path / crops[0].relative_path) as cut:
        assert cut.size == (256, 128)
        # The encoder is lossy and bleeds a little across the boundary, so the check is that
        # the crop's interior came from the marked region rather than that it survived byte for
        # byte. A box off by a quarter of the frame would pull in the background instead.
        interior = cut.convert("RGB").crop((4, 4, cut.width - 4, cut.height - 4))
        (least_red, _most_red), _green, (_least_blue, most_blue) = ImageStat.Stat(interior).extrema
        assert least_red > 200
        assert most_blue < 60


def test_a_stop_s_own_region_overrides_the_sweep_s() -> None:
    whole_frame = {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}
    spec = SweepSpec.model_validate(one_sweep_spec(steps_sweep(crop=REGION, stop_crop=whole_frame)))

    first, second = spec.expand_jobs()

    assert first.crop == CropRegion.model_validate(REGION)
    assert second.crop == CropRegion.model_validate(whole_frame)


def test_a_crop_already_cut_is_left_alone_unless_it_is_forced(tmp_path: Path) -> None:
    spec = SweepSpec.model_validate(one_sweep_spec(steps_sweep(crop=REGION)))
    render_frames(spec=spec, output_root=tmp_path, size=(1024, 512))
    write_sweep_derivatives(spec=spec, output_root=tmp_path)

    render_frames(spec=spec, output_root=tmp_path, size=(2048, 1024))
    reused = write_sweep_derivatives(spec=spec, output_root=tmp_path)
    forced = write_sweep_derivatives(spec=spec, output_root=tmp_path, force=True)

    assert [outcome.reused for outcome in reused if outcome.kind is DerivativeKind.CROP] == [True, True]
    assert [(outcome.width, outcome.height) for outcome in reused if outcome.kind is DerivativeKind.CROP] == [
        (256, 128),
        (256, 128),
    ]
    assert [(outcome.width, outcome.height) for outcome in forced if outcome.kind is DerivativeKind.CROP] == [
        (512, 256),
        (512, 256),
    ]


def test_a_frame_too_small_to_crop_is_skipped_unless_the_region_is_cut_even_when_small(tmp_path: Path) -> None:
    """Assert the size rule applies to the frame, because a crop shows more of the frame."""
    too_small = (MINIMUM_CROPPABLE_PIXELS - 56, MINIMUM_CROPPABLE_PIXELS - 56)
    skipped_spec = SweepSpec.model_validate(one_sweep_spec(steps_sweep(crop=REGION)))
    render_frames(spec=skipped_spec, output_root=tmp_path, size=too_small)

    outcomes = write_sweep_derivatives(spec=skipped_spec, output_root=tmp_path)
    crops = [outcome for outcome in outcomes if outcome.kind is DerivativeKind.CROP]

    assert [outcome.written for outcome in crops] == [False, False]
    assert all("under" in str(outcome.skipped_reason) for outcome in crops)
    assert not (tmp_path / crops[0].relative_path).exists()

    marked_spec = SweepSpec.model_validate(one_sweep_spec(steps_sweep(crop=dict(REGION, even_when_small=True))))
    marked_outcomes = write_sweep_derivatives(spec=marked_spec, output_root=tmp_path)

    assert all(outcome.written for outcome in marked_outcomes if outcome.kind is DerivativeKind.CROP), (
        "a region marked even_when_small is cut from a small frame"
    )


def test_the_index_lists_a_crop_only_when_the_file_is_on_disk(tmp_path: Path) -> None:
    spec = SweepSpec.model_validate(one_sweep_spec(steps_sweep(crop=REGION)))
    render_frames(spec=spec, output_root=tmp_path, size=(1024, 512))

    (before_cutting,) = build_index(spec=spec, output_root=tmp_path).strips
    write_sweep_derivatives(spec=spec, output_root=tmp_path)
    (after_cutting,) = build_index(spec=spec, output_root=tmp_path).strips

    assert all(stop.crop is None for stop in before_cutting.stops)
    for stop in after_cutting.stops:
        assert stop.crop is not None
        assert (stop.crop.width, stop.crop.height) == (256, 128)
        assert (stop.crop.x, stop.crop.y) == (256, 256)
        assert (tmp_path / stop.crop.file).is_file()


def test_a_frame_over_the_bound_gets_a_bounded_copy_and_one_under_it_does_not(tmp_path: Path) -> None:
    over = SweepSpec.model_validate(one_sweep_spec(steps_sweep()))
    render_frames(spec=over, output_root=tmp_path, size=(FIT_LONG_EDGE_PIXELS * 2, FIT_LONG_EDGE_PIXELS))

    outcomes = write_sweep_derivatives(spec=over, output_root=tmp_path)
    bounded = [outcome for outcome in outcomes if outcome.kind is DerivativeKind.FIT]

    assert [outcome.written for outcome in bounded] == [True, True]
    assert [(outcome.width, outcome.height) for outcome in bounded] == [
        (FIT_LONG_EDGE_PIXELS, FIT_LONG_EDGE_PIXELS // 2),
        (FIT_LONG_EDGE_PIXELS, FIT_LONG_EDGE_PIXELS // 2),
    ]

    (strip,) = build_index(spec=over, output_root=tmp_path).strips
    assert all(stop.fit is not None and (tmp_path / stop.fit).is_file() for stop in strip.stops)

    under_root = tmp_path / "under"
    render_frames(spec=over, output_root=under_root, size=(FIT_LONG_EDGE_PIXELS, FIT_LONG_EDGE_PIXELS))
    under_outcomes = write_sweep_derivatives(spec=over, output_root=under_root)

    assert all(not outcome.written for outcome in under_outcomes)
    (unbounded_strip,) = build_index(spec=over, output_root=under_root).strips
    assert all(stop.fit is None for stop in unbounded_strip.stops)


def test_a_bounded_copy_already_written_is_left_alone_unless_it_is_forced(tmp_path: Path) -> None:
    spec = SweepSpec.model_validate(one_sweep_spec(steps_sweep()))
    render_frames(spec=spec, output_root=tmp_path, size=(2048, 2048))
    write_sweep_derivatives(spec=spec, output_root=tmp_path)

    reused = write_sweep_derivatives(spec=spec, output_root=tmp_path)
    forced = write_sweep_derivatives(spec=spec, output_root=tmp_path, force=True)

    assert all(outcome.reused for outcome in reused if outcome.kind is DerivativeKind.FIT)
    assert all(outcome.written for outcome in forced if outcome.kind is DerivativeKind.FIT)


def alchemy_spec_with_a_cropped_entry() -> AlchemySpec:
    """Return the checked-in alchemy specification, which places a region on the face fixers."""
    return AlchemySpec.model_validate(
        json.loads((TOOL_ROOT / "alchemy_spec.json").read_text(encoding="utf-8")),
    )


def test_an_alchemy_entry_crops_both_its_source_and_its_result(tmp_path: Path) -> None:
    """Assert the pair is cut at one region, so the before and after are comparable."""
    spec = alchemy_spec_with_a_cropped_entry()
    entry = next(entry for entry in spec.entries if entry.crop is not None)
    crop = entry.crop
    assert crop is not None
    root = alchemy_root(tmp_path)

    source_path = root / published_source_name(entry)
    source_path.parent.mkdir(parents=True, exist_ok=True)
    marked_frame((512, 512), region=crop).save(source_path, format="WEBP")
    result_path = root / entry.relative_directory / RESULT_IMAGE_FILENAME
    result_path.parent.mkdir(parents=True, exist_ok=True)
    marked_frame((2048, 2048), region=crop).save(result_path, format="WEBP")

    outcomes = write_alchemy_crops(spec=spec, output_root=tmp_path)
    cut_here = {outcome.relative_path for outcome in outcomes if outcome.written}

    # Entries with the same source share its published copy, so the other entries on that source
    # are cut here too. This test checks only this entry's pair.
    assert {published_source_crop_name(entry), published_result_crop_name(entry)} <= cut_here

    manifest_entry = next(
        listed for listed in build_manifest(spec=spec, output_root=tmp_path).entries if listed.form == entry.form
    )
    assert (manifest_entry.source_width, manifest_entry.source_height) == (512, 512)
    assert (manifest_entry.result_width, manifest_entry.result_height) == (2048, 2048)
    assert manifest_entry.source_crop is not None
    assert manifest_entry.result_crop is not None
    assert manifest_entry.result_crop.width == manifest_entry.source_crop.width * 4


def test_an_alchemy_entry_without_a_result_has_no_result_crop(tmp_path: Path) -> None:
    """Assert an entry whose form could not run reports the gap rather than a phantom crop."""
    spec = alchemy_spec_with_a_cropped_entry()
    entry = next(entry for entry in spec.entries if entry.crop is not None)
    crop = entry.crop
    assert crop is not None
    root = alchemy_root(tmp_path)

    source_path = root / published_source_name(entry)
    source_path.parent.mkdir(parents=True, exist_ok=True)
    marked_frame((512, 512), region=crop).save(source_path, format="WEBP")

    write_alchemy_crops(spec=spec, output_root=tmp_path)
    manifest_entry = next(
        listed for listed in build_manifest(spec=spec, output_root=tmp_path).entries if listed.form == entry.form
    )

    assert manifest_entry.source_crop is not None
    assert manifest_entry.result_crop is None
    assert manifest_entry.result_width is None


def test_an_alchemy_result_over_the_bound_gets_a_bounded_copy_in_the_manifest(tmp_path: Path) -> None:
    """Assert the enlarge use case, whose purpose is more pixels, publishes a page-sized copy."""
    spec = alchemy_spec_with_a_cropped_entry()
    entry = next(entry for entry in spec.entries if entry.crop is not None)
    crop = entry.crop
    assert crop is not None
    root = alchemy_root(tmp_path)

    source_path = root / published_source_name(entry)
    source_path.parent.mkdir(parents=True, exist_ok=True)
    marked_frame((512, 512), region=crop).save(source_path, format="WEBP")
    result_path = root / entry.relative_directory / RESULT_IMAGE_FILENAME
    result_path.parent.mkdir(parents=True, exist_ok=True)
    marked_frame((2048, 1024), region=crop).save(result_path, format="WEBP")

    (written,) = [outcome for outcome in write_alchemy_result_fits(spec=spec, output_root=tmp_path) if outcome.written]

    assert written.kind is DerivativeKind.FIT
    assert written.relative_path == published_result_fit_name(entry)
    assert (written.width, written.height) == (FIT_LONG_EDGE_PIXELS, FIT_LONG_EDGE_PIXELS // 2)

    manifest_entry = next(
        listed for listed in build_manifest(spec=spec, output_root=tmp_path).entries if listed.form == entry.form
    )
    assert manifest_entry.result_fit == published_result_fit_name(entry)
    assert (root / manifest_entry.result_fit).is_file()


def test_an_alchemy_result_within_the_bound_gets_no_copy_and_the_copy_is_written_once(tmp_path: Path) -> None:
    """Assert the alchemy results follow the same skip and force rules as the sweeps."""
    spec = alchemy_spec_with_a_cropped_entry()
    entry = next(entry for entry in spec.entries if entry.crop is not None)
    crop = entry.crop
    assert crop is not None
    root = alchemy_root(tmp_path)

    result_path = root / entry.relative_directory / RESULT_IMAGE_FILENAME
    result_path.parent.mkdir(parents=True, exist_ok=True)
    marked_frame((FIT_LONG_EDGE_PIXELS, FIT_LONG_EDGE_PIXELS), region=crop).save(result_path, format="WEBP")

    (within,) = write_alchemy_result_fits(spec=spec, output_root=tmp_path)

    assert not within.written
    assert "already within" in str(within.skipped_reason)
    manifest_entry = next(
        listed for listed in build_manifest(spec=spec, output_root=tmp_path).entries if listed.form == entry.form
    )
    assert manifest_entry.result_fit is None

    marked_frame((2048, 2048), region=crop).save(result_path, format="WEBP")
    write_alchemy_result_fits(spec=spec, output_root=tmp_path)
    (reused,) = write_alchemy_result_fits(spec=spec, output_root=tmp_path)
    (forced,) = write_alchemy_result_fits(spec=spec, output_root=tmp_path, force=True)

    assert reused.reused
    assert forced.written


def test_an_alchemy_text_result_has_no_bounded_copy(tmp_path: Path) -> None:
    """Assert a form without a raster result is skipped instead of reported as a gap."""
    spec = alchemy_spec_with_a_cropped_entry()
    entry = spec.entries[0]
    root = alchemy_root(tmp_path)

    text_path = root / entry.relative_directory / RESULT_TEXT_FILENAME
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text("a thing on a table\n", encoding="utf-8")

    assert write_alchemy_result_fits(spec=spec, output_root=tmp_path) == []

    manifest_entry = build_manifest(spec=spec, output_root=tmp_path).entries[0]
    assert manifest_entry.files.result_text is not None
    assert manifest_entry.result_fit is None
