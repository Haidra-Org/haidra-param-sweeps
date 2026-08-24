# SPDX-FileCopyrightText: 2026 Tazlin
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Check that the checked-in specification expands the way its consumers assume it does."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from param_sweeps.spec import (
    RenderJob,
    StripPresentation,
    SweepParameter,
    SweepSpec,
    WorkflowName,
    load_spec,
    value_label,
)

SPEC_PATH = Path(__file__).resolve().parents[1] / "sweep_spec.json"


@pytest.fixture(scope="module")
def spec() -> SweepSpec:
    return load_spec(SPEC_PATH)


def test_spec_loads_and_declares_the_expected_models(spec: SweepSpec) -> None:
    assert {image_model.slug for image_model in spec.models} == {
        "icbinp",
        "albedobase-xl",
        "nova-anime-xl",
        "stable-diffusion",
        "flux-schnell",
        "deliberate",
        "anything-diffusion",
    }
    assert spec.seed == 1234567890


def test_every_sweep_states_a_rationale(spec: SweepSpec) -> None:
    for sweep in spec.sweeps:
        assert sweep.rationale.strip(), f"{sweep.parameter} has no rationale"


def test_the_converging_step_strips_cover_what_the_ancestral_ones_do(spec: SweepSpec) -> None:
    """Assert every ancestral steps strip has a converging counterpart on the same model and scene.

    The steps lesson must be readable without a sampler that re-rolls the image underneath it,
    or a reader cannot tell what the extra steps bought from what the sampler moved.
    """
    ancestral = {
        (strip.image_model.slug, strip.scene.scene_id)
        for strip in spec.expand_strips()
        if strip.parameter is SweepParameter.STEPS
    }
    converging = {
        (strip.image_model.slug, strip.scene.scene_id)
        for strip in spec.expand_strips()
        if strip.parameter is SweepParameter.STEPS_CONVERGING
    }
    assert ancestral == converging

    converging_jobs = [job for job in spec.expand_jobs() if job.parameter is SweepParameter.STEPS_CONVERGING]
    assert all(job.sampler_name == "k_dpmpp_2m" for job in converging_jobs)


def test_clip_skip_runs_on_every_model_and_the_anime_one_on_an_anime_scene(spec: SweepSpec) -> None:
    """Assert each model's clip-skip strip runs on a scene where the setting can be judged.

    The anime-lineage model renders the shared photoreal scene as a photograph, where the layer
    its lineage was trained around shows nothing, so it is swept on the anime scene instead.
    """
    clip_skip_strips = [strip for strip in spec.expand_strips() if strip.parameter is SweepParameter.CLIP_SKIP]
    assert {strip.image_model.slug for strip in clip_skip_strips} == {
        "icbinp",
        "albedobase-xl",
        "nova-anime-xl",
    }

    anime_strips = [strip for strip in clip_skip_strips if strip.image_model.slug == "nova-anime-xl"]
    assert [strip.scene.scene_id for strip in anime_strips] == ["anime"]


def test_the_anime_model_anchors_at_clip_skip_two(spec: SweepSpec) -> None:
    anime_model = spec.model_for_slug("nova-anime-xl")
    assert anime_model.anchor_clip_skip == 2

    steps_grid_jobs = [
        job
        for job in spec.expand_jobs()
        if job.model_slug == "nova-anime-xl" and job.parameter is SweepParameter.STEPS_CFG_GRID
    ]
    assert steps_grid_jobs
    assert all(job.clip_skip == 2 for job in steps_grid_jobs)


HIRES_PARAMETERS = frozenset(
    {
        SweepParameter.HIRES_FIX,
        SweepParameter.HIRES_GRID,
        SweepParameter.HIRES_FIX_DENOISING_STRENGTH,
    }
)
"""Everything that teaches the two-pass render: the pair, the sizes, and the second pass's strength."""


def test_the_hires_failure_is_shown_on_a_figure(spec: SweepSpec) -> None:
    """Assert the readings about oversize failure are taken on a subject whose failure has a name.

    Above native size a model repeats what it knows. On a compact object that reads as a general
    softness nobody can point at; on a figure it is a second head or a third arm, which is what
    the captions claim. The pair and the grid both make that claim, so both run on the figure.
    """
    failure_parameters = (SweepParameter.HIRES_FIX, SweepParameter.HIRES_GRID)
    scenes = {job.scene_id for job in spec.expand_jobs() if job.parameter in failure_parameters}
    assert scenes == {"person"}


def test_the_hires_strength_is_read_on_a_close_subject(spec: SweepSpec) -> None:
    """Assert the second pass's strength is read where the detail it buys is visible.

    This strip is not about the oversize failure and does not share the recipe that shows it. On
    the pair strip's distant figure the default SD1.5 model renders almost monochrome and the
    whole range moves nothing a reader could name; what the strength buys is detail, and detail
    needs a subject close enough to carry it.
    """
    jobs = [job for job in spec.expand_jobs() if job.parameter is SweepParameter.HIRES_FIX_DENOISING_STRENGTH]
    assert {job.scene_id for job in jobs} == {"portrait"}
    assert {job.model_slug for job in jobs} == {"icbinp", "albedobase-xl"}


def test_each_hires_pair_renders_where_its_own_baseline_breaks(spec: SweepSpec) -> None:
    """Assert each pair uses the frame shape that makes its baseline repeat the subject.

    The two do not fail the same way. SD1.5 stacks a subject up a long frame, so its pair is
    tall. SDXL does not: it composed a single clean figure at 1152x2048 and again at 1536x3072,
    three times its training height. What breaks SDXL is square area, so its pair is square and
    well past the 2048 where the doubling starts. Holding both to one frame shape would publish a
    pair that teaches nothing on one of them.
    """
    strips = {
        strip.image_model.slug: strip for strip in spec.expand_strips() if strip.parameter is SweepParameter.HIRES_FIX
    }
    assert set(strips) == {"stable-diffusion", "albedobase-xl"}

    for strip in strips.values():
        native = spec.model_for_slug(strip.image_model.slug)
        for job in strip.jobs:
            assert job.width > native.native_width, "both dimensions must clear native or the flag is ignored"
            assert job.height > native.native_height

    tall = strips["stable-diffusion"].jobs[0]
    assert tall.height > tall.width

    square = strips["albedobase-xl"].jobs[0]
    assert square.height == square.width
    assert square.width > 2048, "SDXL still composes one figure at 2048 square; the pair must clear it"


def test_tiling_is_taught_on_a_texture_and_counter_taught_on_an_object(spec: SweepSpec) -> None:
    """Assert tiling is swept on a surface as well as on an object.

    A seam is only visible where one copy of an image meets the next, which a centred object
    hides and a repeating surface shows; the object strip is kept as the counter-example.
    """
    scenes = {job.scene_id for job in spec.expand_jobs() if job.parameter is SweepParameter.TILING}
    assert scenes == {"texture", "thing"}


def test_tiling_is_presented_as_a_repeat(spec: SweepSpec) -> None:
    tiling_strips = [strip for strip in spec.expand_strips() if strip.parameter is SweepParameter.TILING]
    assert tiling_strips
    assert all(strip.presentation is StripPresentation.TILE_REPEAT_3X3 for strip in tiling_strips)


def test_hires_sweeps_render_above_native_size(spec: SweepSpec) -> None:
    hires_jobs = [job for job in spec.expand_jobs() if job.parameter in HIRES_PARAMETERS]
    assert hires_jobs
    for job in hires_jobs:
        native_width = spec.model_for_slug(job.model_slug).native_width
        assert job.width > native_width, "hires-fix does nothing at or below native size"
        assert job.width % 64 == 0


def test_the_hires_strength_strip_asks_for_the_pass_at_every_stop(spec: SweepSpec) -> None:
    """Assert the strip varies the second pass's strength and nothing else about the render.

    The number only means anything while the pass runs, and the strip is a reading of that pass:
    if a stop turned the flag off it would publish a first-pass render beside two-pass ones.
    """
    strips = [
        strip for strip in spec.expand_strips() if strip.parameter is SweepParameter.HIRES_FIX_DENOISING_STRENGTH
    ]
    assert {strip.image_model.slug for strip in strips} == {"icbinp", "albedobase-xl"}

    for strip in strips:
        strengths = [job.hires_fix_denoising_strength for job in strip.jobs]
        assert all(job.hires_fix for job in strip.jobs)
        assert strengths == sorted(strengths), f"{strip.strip_id} is not in ascending order"
        assert strengths == [0.1, 0.3, 0.45, 0.55, 0.65, 0.75, 0.9, 1.0]
        assert 0.65 in strengths, "the strip must show the value a reader gets without touching it"
        assert strip.jobs[0].generation_parameters().differing_field_names(
            strip.jobs[-1].generation_parameters(),
        ) == {"hires_fix_denoising_strength"}


def test_the_hires_grid_pairs_every_size_with_the_flag_off_and_on(spec: SweepSpec) -> None:
    """Assert the grid reads as sizes down and the flag across, which is what makes it a repair."""
    grid_strips = [strip for strip in spec.expand_strips() if strip.parameter is SweepParameter.HIRES_GRID]
    assert len(grid_strips) == 2
    for strip in grid_strips:
        assert strip.grid_rows is not None
        assert strip.grid_columns is not None
        assert strip.grid_rows.parameter is SweepParameter.RESOLUTION
        assert strip.grid_columns.values == [False, True]
        assert [job.hires_fix for job in strip.jobs] == [False, True] * len(strip.grid_rows.values)
        assert [job.width for job in strip.jobs[:2]] == [job.height for job in strip.jobs[:2]]


def test_the_karras_grid_holds_a_converging_sampler_across_every_cell(spec: SweepSpec) -> None:
    """Assert the grid isolates the schedule, as the pair strip does, rather than re-rolling the scene."""
    grid_jobs = [job for job in spec.expand_jobs() if job.parameter is SweepParameter.KARRAS_GRID]
    assert grid_jobs
    assert all(job.sampler_name == "k_dpmpp_2m" for job in grid_jobs)
    assert [job.karras for job in grid_jobs[:2]] == [False, True]
    assert {job.steps for job in grid_jobs} == {6, 10, 16, 30}


def test_prompt_emphasis_varies_the_prompt_and_nothing_else(spec: SweepSpec) -> None:
    """Assert the emphasis strips change only the weighted phrase appended to the scene's prompt."""
    strips = [strip for strip in spec.expand_strips() if strip.parameter is SweepParameter.PROMPT_EMPHASIS]
    assert len(strips) == 2

    for strip in strips:
        fragments = [job.prompt_fragment for job in strip.jobs]
        assert all(fragment is not None and ":" in fragment for fragment in fragments)
        assert len(set(fragments)) == len(fragments)
        assert all(job.prompt.startswith(strip.scene.prompt) for job in strip.jobs)
        assert len({job.steps for job in strip.jobs}) == 1
        assert len({job.cfg_scale for job in strip.jobs}) == 1
        assert strip.jobs[0].generation_parameters().differing_field_names(
            strip.jobs[-1].generation_parameters(),
        ) == {"prompt"}

    # The QR gallery uses the same machinery for a different purpose: its fragment carries the
    # world each finished code is drawn from, so it is the other strip allowed to move the prompt.
    fragment_parameters = (SweepParameter.PROMPT_EMPHASIS, SweepParameter.QR_STYLE)
    jobs_elsewhere = [job for job in spec.expand_jobs() if job.parameter not in fragment_parameters]
    assert all(job.prompt_fragment is None for job in jobs_elsewhere)


def test_the_grid_is_six_by_six_per_baseline(spec: SweepSpec) -> None:
    grid_strips = [strip for strip in spec.expand_strips() if strip.parameter is SweepParameter.STEPS_CFG_GRID]
    assert len(grid_strips) == 3
    for strip in grid_strips:
        assert len(strip.jobs) == 36
        assert strip.grid_rows is not None
        assert strip.grid_rows.values == [4, 8, 12, 20, 30, 50]


def test_the_sampler_grid_reads_samplers_down_and_steps_across(spec: SweepSpec) -> None:
    """Assert the sampler grid is laid out so that a row is one sampler as steps climb.

    Which sampler to use is a question about how it behaves as steps change, so the samplers
    have to be the rows: a reader answers it by reading along one.
    """
    grid_strips = [strip for strip in spec.expand_strips() if strip.parameter is SweepParameter.SAMPLER_STEPS_GRID]
    assert len(grid_strips) == 2
    for strip in grid_strips:
        assert strip.grid_rows is not None
        assert strip.grid_columns is not None
        assert strip.grid_rows.parameter is SweepParameter.SAMPLER
        assert strip.grid_columns.parameter is SweepParameter.STEPS
        assert len(strip.jobs) == len(strip.grid_rows.values) * len(strip.grid_columns.values)
        assert [job.sampler_name for job in strip.jobs[:5]] == ["k_euler"] * 5


def test_the_steps_grid_cell_names_are_unchanged_by_the_generalised_axes(spec: SweepSpec) -> None:
    """Assert the guidance grid still names its cells as it did, so nothing already rendered orphans."""
    grid_jobs = [
        job
        for job in spec.expand_jobs()
        if job.parameter is SweepParameter.STEPS_CFG_GRID and job.model_slug == "icbinp"
    ]
    assert grid_jobs[0].relative_image_path == "steps-cfg-grid/icbinp/person/001_s04_c1_0.webp"
    assert grid_jobs[-1].relative_image_path == "steps-cfg-grid/icbinp/person/036_s50_c15_0.webp"
    assert grid_jobs[0].caption == "4 steps at guidance 1."


def test_resolution_strip_is_sdxl_only_and_ends_at_native(spec: SweepSpec) -> None:
    resolution_jobs = [job for job in spec.expand_jobs() if job.parameter is SweepParameter.RESOLUTION]
    assert {job.model_slug for job in resolution_jobs} == {"albedobase-xl"}
    assert [job.width for job in resolution_jobs] == [128, 256, 512, 768, 1024]


def test_every_strip_outside_the_seed_strip_holds_one_seed(spec: SweepSpec) -> None:
    """Assert no strip but the seed strip varies the seed between its own stops.

    Comparability is a within-strip property: the stops of one strip must share a seed or the
    strip teaches the seed rather than its parameter. A sweep may deliberately hold a non-anchor
    seed (the SD1.5 hires pair pins one whose single pass stacks while the repaired pass
    composes well), so the anchor seed is the expected default rather than a rule.
    """
    for strip in spec.expand_strips():
        if strip.parameter is SweepParameter.SEED:
            continue
        seeds = {job.seed for job in strip.jobs}
        assert len(seeds) == 1, f"{strip.parameter}/{strip.image_model.slug} varies the seed between stops: {seeds}"


def test_the_seed_strips_vary_the_seed_and_label_it_readably(spec: SweepSpec) -> None:
    """Assert every seed strip is labelled by letter, because a raw seed number teaches nobody."""
    strips = [strip for strip in spec.expand_strips() if strip.parameter is SweepParameter.SEED]
    assert {strip.scene.scene_id for strip in strips} == {"person", "place"}

    for strip in strips:
        assert len(strip.jobs) == 6
        assert len({job.seed for job in strip.jobs}) == len(strip.jobs)
        assert [job.stop_value for job in strip.jobs] == [
            "Seed A",
            "Seed B",
            "Seed C",
            "Seed D",
            "Seed E",
            "Seed F",
        ]


SOURCED_PARAMETERS = frozenset(
    {
        SweepParameter.DENOISING_STRENGTH,
        SweepParameter.DENOISING_STRENGTH_CONVERGING,
        SweepParameter.UPSCALER_FORM,
        SweepParameter.MASKED_DENOISING_STRENGTH,
        SweepParameter.MASKING_GRID,
        SweepParameter.OUTPAINT_EXTENT,
    }
)
"""The parameters whose stops start from an image that already exists rather than from noise.

Denoising strength is swept image-to-image, and the three masked readings are the same thing with
part of the picture covered. The upscaler-form strip is not a generation at all: its stops are one
image put through each of the horde's upscalers. The ControlNet strips are sourced too, but by
their control type rather than by their parameter, so they are not listed here.
"""

MASKED_PARAMETERS = frozenset(
    {
        SweepParameter.MASKED_DENOISING_STRENGTH,
        SweepParameter.MASKING_GRID,
        SweepParameter.OUTPAINT_EXTENT,
    }
)
"""The readings that cover part of the picture: what survives, what is replaced, what is added."""

DENOISING_PARAMETERS = frozenset({SweepParameter.DENOISING_STRENGTH, SweepParameter.DENOISING_STRENGTH_CONVERGING})
"""The strips that sweep denoising strength: on the anchor's ancestral sampler, and on one that settles."""


def test_the_image_to_image_strips_are_the_only_ones_with_a_source(spec: SweepSpec) -> None:
    """Assert the sweeps that start from an image are the only ones declaring a source.

    A job that conditions on a control type has a source too, whatever it is sweeping: the picture
    the detector reads is that source, so the strength strips carry one and the parameter alone
    does not say whether a job should.
    """
    for job in spec.expand_jobs():
        expects_source = job.parameter in SOURCED_PARAMETERS or job.control_type is not None
        assert (job.source_image is not None) is expects_source, job.relative_image_path

    source_paths = {job.source_image_path for job in spec.expand_jobs() if job.source_image_path is not None}
    for source_path in source_paths:
        assert source_path.is_file(), f"{source_path} is declared as a source image but is not checked in"

    mask_paths = {job.source_mask_path for job in spec.expand_jobs() if job.source_mask_path is not None}
    for mask_path in mask_paths:
        assert mask_path.is_file(), f"{mask_path} is declared as a mask but is not checked in"


def test_every_masked_job_carries_a_mask(spec: SweepSpec) -> None:
    """Assert the readings about masking actually mask something.

    A masked job whose mask went missing does not fail: image-to-image renders the whole frame and
    the paint modes fall back to repainting everything, either of which publishes a plain render
    under a caption about what the mask kept.
    """
    jobs = [job for job in spec.expand_jobs() if job.parameter in MASKED_PARAMETERS]
    assert jobs
    for job in jobs:
        assert job.source_mask is not None, f"{job.relative_image_path} masks nothing"
        assert job.source_image is not None


def test_outpainting_stops_state_the_size_of_their_own_canvas(spec: SweepSpec) -> None:
    """Assert each extent renders at the size of the canvas it was given.

    Nothing in the backends pads a canvas, so each stop sends a picture padded beforehand. The
    backends do resize a source to the requested dimensions, so a stop inheriting the scene's size
    would have its padding squeezed straight back out and would teach nothing.
    """
    jobs = [job for job in spec.expand_jobs() if job.parameter is SweepParameter.OUTPAINT_EXTENT]
    assert jobs

    for job in jobs:
        assert job.source_image_path is not None
        with Image.open(job.source_image_path) as padded:
            assert padded.size == (job.width, job.height), f"{job.relative_image_path} would be rescaled"

        assert job.source_mask_path is not None
        with Image.open(job.source_mask_path) as mask:
            assert mask.size == (job.width, job.height)

    widths = [job.width for job in jobs]
    assert widths == sorted(widths), "the extents must read outwards"
    assert len(set(widths)) == len(widths), "two stops padded to the same size teach the same thing"


def test_each_image_to_image_strip_publishes_its_own_source(spec: SweepSpec) -> None:
    """Assert two sweeps of one parameter cannot publish over each other's source image."""
    strips = [strip for strip in spec.expand_strips() if strip.parameter in DENOISING_PARAMETERS]
    published = [strip.published_source_image for strip in strips]
    assert len(set(published)) == len(strips)
    for strip in strips:
        assert strip.published_source_image == f"{strip.strip_id}/source.webp"


def test_the_denoising_strips_run_the_full_range_and_end_at_one(spec: SweepSpec) -> None:
    """Assert the strips are granular above 0.5, where the choices are, and end where the source stops counting."""
    for strip in spec.expand_strips():
        if strip.parameter not in DENOISING_PARAMETERS:
            continue

        strengths = [job.denoising_strength for job in strip.jobs]
        assert strengths == sorted(strengths), f"{strip.strip_id} is not in ascending order"
        assert strengths == [0.15, 0.3, 0.45, 0.6, 0.65, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
        assert len([strength for strength in strengths if strength > 0.5]) > len(strengths) / 2


def test_the_negative_prompt_strips_turn_one_prompt_on_and_off(spec: SweepSpec) -> None:
    """Assert both negative-prompt lessons are a clean pair: the same image with the text on and off."""
    strips = [strip for strip in spec.expand_strips() if strip.parameter is SweepParameter.NEGATIVE_PROMPT]
    assert {strip.scene.scene_id for strip in strips} == {"portrait", "harbour"}

    for strip in strips:
        without_negative, with_negative = strip.jobs
        assert without_negative.negative_prompt == ""
        assert with_negative.negative_prompt
        assert without_negative.generation_parameters().differing_field_names(
            with_negative.generation_parameters(),
        ) == {"negative_prompt"}

    # The exceptions are excused for stated reasons. The step-preview strips reproduce the txt2img
    # tutorial's request verbatim, so the frames arrive at the picture the tutorial already showed
    # the reader, and that request carries a negative prompt. The two QR strips carry the one the
    # workflow's recipe was measured with: their stops were scan-tested with it, and dropping it
    # would mean publishing a code nothing had decoded. The ControlNet galleries reproduce the
    # request the ControlNet tutorial makes of each of its worlds, negative prompt included, so a
    # reader comparing detectors is looking at the pictures that tutorial's recipe produces.
    carry_a_negative_prompt = {
        SweepParameter.NEGATIVE_PROMPT,
        SweepParameter.STEP_PREVIEW,
        SweepParameter.STEP_PREVIEW_GUESS,
        SweepParameter.CONTROL_STRENGTH,
        SweepParameter.CONTROLNET_GALLERY,
        SweepParameter.QR_STYLE,
    }
    other_jobs = [job for job in spec.expand_jobs() if job.parameter not in carry_a_negative_prompt]
    assert all(job.negative_prompt == "" for job in other_jobs)


def test_the_karras_strip_forces_the_step_count_it_is_legible_at(spec: SweepSpec) -> None:
    """Assert the schedule pair is rendered at 10 steps, not at the anchor's 30, where it hides."""
    karras_jobs = [job for job in spec.expand_jobs() if job.parameter is SweepParameter.KARRAS]
    assert [job.karras for job in karras_jobs] == [False, True]
    assert all(job.steps == 10 for job in karras_jobs)
    assert spec.anchor.steps == 30


def test_every_scene_belongs_to_one_of_the_three_anchor_worlds(spec: SweepSpec) -> None:
    """Assert the library has one cast rather than a new subject per lesson.

    Two images are only comparable when they are of the same world, so every scene has to sit in
    the skyharbor, the fox shrine or the rooftop garden and say so in its own prompt.
    """
    assert {scene.scene_id for scene in spec.scenes} == {
        "person",
        "place",
        "thing",
        "vivid",
        "portrait",
        "anime",
        "harbour",
        "texture",
        "snowstorm",
        "garden",
        "courier",
        "cloud-sea",
        "qr-artwork",
        "deck-prose",
        "courier-ink",
        "fox-shrine-ink",
        "robot-gardener-ink",
    }
    world_markers = ("skyharbor", "zeppelin", "shrine", "gardening robot", "rooftop")

    # The QR gallery's scene is a treatment rather than a place: it is three worlds in one strip,
    # so each of its stops names its own world in the fragment and the scene prompt names none.
    for scene in spec.scenes:
        if scene.scene_id == "qr-artwork":
            continue
        assert any(marker in scene.prompt for marker in world_markers), scene.scene_id

    gallery_jobs = [job for job in spec.expand_jobs() if job.scene_id == "qr-artwork"]
    assert gallery_jobs
    for job in gallery_jobs:
        assert any(marker in job.prompt for marker in world_markers), job.stop_label


def test_a_worlds_subject_is_described_in_the_same_words_everywhere(spec: SweepSpec) -> None:
    """Assert the descriptor blocks are byte-identical wherever a world's subject appears.

    A subject re-described in different words is a different subject, and a strip of it teaches
    the prompt rather than the parameter it is sweeping.
    """
    descriptor_blocks = {
        "w1": (
            "a woman in her 30s with a sleek silver bob haircut, round copper goggles resting up in her silver hair,"
            " a patched teal canvas coat, a small brass compass on a cord around her neck"
        ),
        "w2": (
            "1girl, shrine keeper, white haori, red hakama, long black hair, single hairpin,"
            " (silver fox:1.2) sitting at her side, fox, amber eyes"
        ),
        "w3": (
            "a small rounded gardening robot with weathered orange paint, a single round blue lens for a head,"
            " a hand-painted daisy decal on its chest plate, young ivy curling around its left arm"
        ),
    }
    subject_scenes = {
        "person": "w1",
        "portrait": "w1",
        "snowstorm": "w1",
        "courier": "w1",
        "courier-ink": "w1",
        "anime": "w2",
        "fox-shrine-ink": "w2",
        "thing": "w3",
        "robot-gardener-ink": "w3",
    }
    for scene_id, world in subject_scenes.items():
        assert descriptor_blocks[world] in spec.scene_for_id(scene_id).prompt, scene_id

    for job in spec.expand_jobs():
        job_world = subject_scenes.get(job.scene_id)
        if job_world is not None:
            assert descriptor_blocks[job_world] in job.prompt, job.relative_image_path


def test_the_control_strength_strips_move_nothing_but_the_strength(spec: SweepSpec) -> None:
    """Assert every strength strip varies its ControlNet and holds everything else, seed included.

    A stop that reached its scan by changing seed as well as strength would be teaching two things
    at once, so each strip is pinned to the one seed that holds across the whole range it publishes.
    The QR strip walks the ladder on the QR workflow; the rest walk the same ladder once per
    detector family, so the grip of each family is read against the others and against the QR one.
    """
    strength_ladder = [1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6]
    strips = [strip for strip in spec.expand_strips() if strip.parameter is SweepParameter.CONTROL_STRENGTH]

    assert len(strips) == 8

    for strip in strips:
        assert [job.control_strength for job in strip.jobs] == strength_ladder
        assert len({job.seed for job in strip.jobs}) == 1
        assert len({job.prompt for job in strip.jobs}) == 1
        assert len({job.control_type for job in strip.jobs}) == 1

    qr_strips = [strip for strip in strips if strip.jobs[0].workflow is WorkflowName.QR_CODE]
    detector_strips = [strip for strip in strips if strip.jobs[0].workflow is not WorkflowName.QR_CODE]

    assert len(qr_strips) == 1
    assert all(job.control_type is None for job in qr_strips[0].jobs)
    assert {strip.jobs[0].control_type for strip in detector_strips} == {
        "canny",
        "mlsd",
        "depth_anything_v2",
        "normal_bae",
        "openpose",
        "oneformer_ade20k",
        "tile",
    }


def test_a_control_map_is_filed_beside_the_render_it_guided(spec: SweepSpec) -> None:
    """Assert a job states a map path when it conditions on a detector and none when it does not.

    The map and the render are only meaningful as a pair, so a consumer holding the render's path
    holds the map's: same directory, same stem.
    """
    unconditioned = next(job for job in spec.expand_jobs() if job.control_type is None)
    assert unconditioned.relative_control_map_path is None

    conditioned = next(job for job in spec.expand_jobs() if job.control_type is not None)
    map_path = conditioned.relative_control_map_path
    assert map_path is not None

    render_directory, _, render_filename = conditioned.relative_image_path.rpartition("/")
    map_directory, _, map_filename = map_path.rpartition("/")
    assert map_directory == render_directory
    assert map_filename == f"{render_filename.removesuffix('.webp')}_map.webp"


def test_the_qr_gallery_shows_the_layout_knobs_it_claims_to(spec: SweepSpec) -> None:
    """Assert each gallery stop states the knobs its caption names, and curates from a batch."""
    jobs = [job for job in spec.expand_jobs() if job.parameter is SweepParameter.QR_STYLE]
    references = {job.stop_label: {extra_text.reference for extra_text in job.extra_texts} for job in jobs}

    assert all(job.batch_size > 1 for job in jobs), "a gallery frame is picked out of several"
    assert len({job.control_strength for job in jobs}) == 1, "the gallery is a treatment, not a strength sweep"
    assert "module_drawer" in references["shrine-leaves"]
    assert {"x_offset", "y_offset"} <= references["ivy-garden"]
    assert references["mist-towers"] == {"qr_code"}, "the first stop is the workflow with nothing else set"


def test_filenames_are_unique_and_deterministic(spec: SweepSpec) -> None:
    first_expansion = spec.expand_jobs()
    second_expansion = load_spec(SPEC_PATH).expand_jobs()

    first_paths = [job.relative_image_path for job in first_expansion]
    assert first_paths == [job.relative_image_path for job in second_expansion]
    assert len(set(first_paths)) == len(first_paths)


def test_stop_indexes_restart_within_each_strip(spec: SweepSpec) -> None:
    for strip in spec.expand_strips():
        assert [job.stop_index for job in strip.jobs] == list(range(1, len(strip.jobs) + 1))


def test_paths_follow_the_documented_layout(spec: SweepSpec) -> None:
    job: RenderJob = next(job for job in spec.expand_jobs() if job.parameter is SweepParameter.CLIP_SKIP)
    assert job.relative_image_path.startswith("clip-skip/")
    assert job.relative_image_path.endswith(".webp")
    assert job.relative_sidecar_path.endswith(".json")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (4, "4"),
        (7.5, "7_5"),
        (True, "on"),
        (False, "off"),
        ("k_euler_a", "k_euler_a"),
        ("128x128", "128x128"),
        ("AlbedoBase XL (SDXL)", "albedobase-xl-sdxl"),
    ],
)
def test_value_label(value: bool | int | float | str, expected: str) -> None:
    assert value_label(value) == expected


def one_sweep_spec(sweep: dict[str, Any]) -> dict[str, Any]:
    """Return a whole specification whose only sweep is the one given.

    The checked-in library is too large to state a rule against directly: a test that perturbs it
    is testing the perturbation as much as the rule. This builds the smallest thing the validator
    will look at, so a raised error can only have come from the sweep under test.
    """
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


def hires_denoise_sweep(**holds: Any) -> dict[str, Any]:
    """Return a two-stop strip sweeping what the hires pass redraws by, holding what is asked."""
    return {
        "kind": "strip",
        "parameter": "hires_fix_denoising_strength",
        "rationale": "A rule needs something to reject.",
        "caption": "The second pass, at two strengths.",
        "model_slugs": ["small"],
        "scene_ids": ["thing"],
        "presentation": "strip",
        "holds": holds,
        "stops": [
            {"value": 0.3, "hires_fix_denoising_strength": 0.3, "caption": "Lightly."},
            {"value": 0.9, "hires_fix_denoising_strength": 0.9, "caption": "Heavily."},
        ],
    }


def test_the_hires_strength_reaches_the_resolved_job(spec: SweepSpec) -> None:
    """Assert a stated hires strength survives resolution, and an unstated one stays unstated."""
    stated = SweepSpec.model_validate(
        one_sweep_spec(hires_denoise_sweep(hires_fix=True, width=768, height=768)),
    )
    assert [job.hires_fix_denoising_strength for job in stated.expand_jobs()] == [0.3, 0.9]
    assert all(job.generation_parameters().hires_fix_denoising_strength is not None for job in stated.expand_jobs())

    for job in spec.expand_jobs():
        if job.parameter is not SweepParameter.HIRES_FIX_DENOISING_STRENGTH:
            assert (
                job.hires_fix_denoising_strength is None
            ), f"{job.relative_image_path} states a hires strength it does not teach"


def test_a_hires_strength_without_the_hires_pass_is_rejected() -> None:
    """Assert the specification cannot publish a number no backend reads.

    The second pass is the only thing that reads this strength. Stating it on a job that never
    runs a second pass would render a strip whose stops are identical and whose captions claim
    they are not.
    """
    with pytest.raises(ValueError, match="without asking for the hires pass"):
        SweepSpec.model_validate(one_sweep_spec(hires_denoise_sweep(width=768, height=768)))


def test_the_hires_pass_is_rejected_where_the_backends_disable_it() -> None:
    """Assert both silent-disable rules are caught: at or below native size, and while painting."""
    with pytest.raises(ValueError, match="silently disable it unless both dimensions exceed"):
        SweepSpec.model_validate(one_sweep_spec(hires_denoise_sweep(hires_fix=True, width=512, height=768)))

    painting = hires_denoise_sweep(hires_fix=True, width=768, height=768)
    painting["source_image"] = "inputs/skyharbor-deck-sunset.webp"
    painting["holds"]["source_processing"] = "inpainting"
    with pytest.raises(ValueError, match="asks for hires fix while inpainting"):
        SweepSpec.model_validate(one_sweep_spec(painting))


def test_a_stop_may_start_from_its_own_picture() -> None:
    """Assert a stop can override the sweep's source, which is what outpainting extents need.

    Nothing in the backends pads a canvas, so each extent is a picture that was padded before it
    was sent. The stops of such a strip do not share an input, and the sweep's own source stays
    the picture a reader is shown as the starting point.
    """
    sweep = {
        "kind": "strip",
        "parameter": "outpaint_extent",
        "rationale": "A rule needs something to accept.",
        "caption": "Pushed out by more each time.",
        "model_slugs": ["small"],
        "scene_ids": ["thing"],
        "presentation": "strip",
        "source_image": "inputs/skyharbor-deck-sunset.webp",
        "holds": {"source_processing": "outpainting"},
        "stops": [
            {"value": 0, "caption": "Where it started."},
            {
                "value": 256,
                "caption": "Pushed out.",
                "source_image": "inputs/rooftop-garden-sketch.webp",
            },
        ],
    }
    jobs = SweepSpec.model_validate(one_sweep_spec(sweep)).expand_jobs()
    assert [job.source_image for job in jobs] == [
        "inputs/skyharbor-deck-sunset.webp",
        "inputs/rooftop-garden-sketch.webp",
    ]
