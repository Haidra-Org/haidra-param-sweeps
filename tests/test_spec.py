# SPDX-FileCopyrightText: 2026 Tazlin
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Check that the checked-in specification expands in the way that its consumers depend on."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from param_sweeps.spec import (
    PREVIEWABLE_PRESENTATIONS,
    RenderJob,
    StripPresentation,
    SweepParameter,
    SweepSpec,
    SweepSubject,
    WorkflowName,
    load_spec,
    resolve_preview_indices,
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
    or a reader cannot separate the effect of the extra steps from the sampler's random changes.
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
    that its lineage was trained around has no visible effect, so it is swept on the anime scene
    instead.
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
    """Assert the oversize-failure strips use a subject whose failure is easy to name.

    Above native size, a model repeats what it knows. On a compact object, that looks like a
    general softness that is hard to point at. On a figure, it is a second head or a third arm,
    as the captions describe. The pair and the grid both make that claim, so both run on the
    figure.
    """
    failure_parameters = (SweepParameter.HIRES_FIX, SweepParameter.HIRES_GRID)
    scenes = {job.scene_id for job in spec.expand_jobs() if job.parameter in failure_parameters}
    assert scenes == {"person"}


def test_the_hires_strength_is_read_on_a_close_subject(spec: SweepSpec) -> None:
    """Assert the second pass's strength is shown on a subject where its added detail is visible.

    This strip is not about the oversize failure and does not share the recipe that shows it. On
    the pair strip's distant figure, the default SD1.5 model renders almost monochrome, and the
    whole range does not produce a visible difference. The strength adds detail, and detail needs
    a close subject.
    """
    jobs = [job for job in spec.expand_jobs() if job.parameter is SweepParameter.HIRES_FIX_DENOISING_STRENGTH]
    assert {job.scene_id for job in jobs} == {"portrait"}
    assert {job.model_slug for job in jobs} == {"icbinp", "albedobase-xl"}


def test_each_hires_pair_renders_where_its_own_baseline_breaks(spec: SweepSpec) -> None:
    """Assert each pair uses the frame shape that makes its baseline repeat the subject.

    The two do not fail the same way. SD1.5 stacks a subject up a long frame, so its pair is
    tall. SDXL does not. It composed a single clean figure at 1152x2048 and again at 1536x3072,
    three times its training height. What breaks SDXL is square area, so its pair is square and
    well past the 2048 where the doubling starts. One frame shape for both would publish a pair
    that does not show anything for one of them.
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
    assert square.width > 2048, "SDXL still composes one figure at 2048 square, so the pair must exceed it"


def test_tiling_is_taught_on_a_texture_and_counter_taught_on_an_object(spec: SweepSpec) -> None:
    """Assert tiling is swept on a surface as well as on an object.

    A seam is only visible where one copy of an image meets the next, which a centred object
    hides and a repeating surface shows. The object strip is kept as the counter-example.
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
        assert job.width > native_width, "hires-fix has no effect at or below native size"
        assert job.width % 64 == 0


def test_the_hires_strength_strip_requests_the_pass_at_every_stop(spec: SweepSpec) -> None:
    """Assert the strip varies the second pass's strength and nothing else about the render.

    The strength only has an effect while the pass runs, and the strip shows that pass. A stop
    that turned the flag off would publish a first-pass render beside two-pass ones.
    """
    strips = [
        strip for strip in spec.expand_strips() if strip.parameter is SweepParameter.HIRES_FIX_DENOISING_STRENGTH
    ]
    assert {strip.image_model.slug for strip in strips} == {"icbinp", "albedobase-xl"}

    for strip in strips:
        strengths = [
            job.hires_fix_denoising_strength for job in strip.jobs if job.hires_fix_denoising_strength is not None
        ]
        assert all(job.hires_fix for job in strip.jobs)
        assert strengths == sorted(strengths), f"{strip.strip_id} is not in ascending order"
        assert strengths == [0.1, 0.3, 0.45, 0.55, 0.65, 0.75, 0.9, 1.0]
        assert 0.65 in strengths, "the strip must show the default value"
        assert strip.jobs[0].generation_parameters().differing_field_names(
            strip.jobs[-1].generation_parameters(),
        ) == {"hires_fix_denoising_strength"}


def test_the_hires_grid_pairs_every_size_with_the_flag_off_and_on(spec: SweepSpec) -> None:
    """Assert the grid has sizes down the rows and the flag across, so it shows the repair at each size."""
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

    # The QR gallery uses the same machinery for a different purpose: its fragment names the
    # world that each finished code is drawn from, so it is the other strip allowed to move the
    # prompt.
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

    Which sampler to use is a question about how it behaves as steps change, so the samplers are
    the rows, and a reader can answer the question by reading along one row.
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


def test_the_steps_grid_cell_names_are_stable(spec: SweepSpec) -> None:
    """Assert the guidance grid's cell names, so rendered images keep their paths."""
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

    Comparability is within a strip: the stops of one strip must share a seed, or the strip shows
    the effect of the seed instead of its parameter. A sweep can deliberately hold a non-anchor
    seed (the SD1.5 hires pair uses one where the single pass stacks the subject and the repaired
    pass composes well), so the anchor seed is the usual default, not a rule.
    """
    for strip in spec.expand_strips():
        if strip.parameter is SweepParameter.SEED:
            continue
        seeds = {job.seed for job in strip.jobs}
        assert len(seeds) == 1, f"{strip.parameter}/{strip.image_model.slug} varies the seed between stops: {seeds}"


def test_the_seed_strips_vary_the_seed_and_label_it_readably(spec: SweepSpec) -> None:
    """Assert every seed strip is labelled by letter, because a raw seed number is not meaningful to a reader."""
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
"""The parameters whose stops start from an existing image instead of from noise.

Denoising strength is swept image-to-image, and the three masked strips are the same sweep with
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
"""The strips that cover part of the picture: what is kept, what is replaced, what is added."""

DENOISING_PARAMETERS = frozenset({SweepParameter.DENOISING_STRENGTH, SweepParameter.DENOISING_STRENGTH_CONVERGING})
"""The strips that sweep denoising strength: on the anchor's ancestral sampler, and on one that converges."""


def test_the_image_to_image_strips_are_the_only_ones_with_a_source(spec: SweepSpec) -> None:
    """Assert the sweeps that start from an image are the only ones declaring a source.

    A job that conditions on a control type has a source too, whatever it is sweeping: the picture
    that the detector reads is that source, so the strength strips have one, and the parameter
    alone does not determine whether a job has a source.
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


def test_every_masked_job_has_a_mask(spec: SweepSpec) -> None:
    """Assert the masking strips use a mask.

    A masked job whose mask went missing does not fail: image-to-image renders the whole frame and
    the paint modes fall back to repainting everything, either of which publishes a plain render
    under a caption about what the mask kept.
    """
    jobs = [job for job in spec.expand_jobs() if job.parameter in MASKED_PARAMETERS]
    assert jobs
    for job in jobs:
        assert job.source_mask is not None, f"{job.relative_image_path} does not have a mask"
        assert job.source_image is not None


def test_outpainting_stops_state_the_size_of_their_own_canvas(spec: SweepSpec) -> None:
    """Assert each extent renders at the size of its canvas.

    The backends do not pad a canvas, so each stop sends a picture padded beforehand. The
    backends do resize a source to the requested dimensions, so a stop inheriting the scene's size
    would have its padding scaled away, and would not show anything.
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
    """Assert the strips are granular above 0.5, where the choices are, and end at 1, where the source is ignored."""
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

    # These exceptions have stated reasons. The step-preview strips reproduce the txt2img
    # tutorial's request verbatim, so the frames end at the picture that the tutorial already
    # showed the reader, and that request includes a negative prompt. The two QR strips use the
    # negative prompt that the workflow's recipe was measured with: their stops were scan-tested
    # with it, and dropping it would mean publishing a code that was never decoded. The ControlNet
    # galleries reproduce the request that the ControlNet tutorial makes for each of its worlds,
    # negative prompt included, so a reader comparing detectors is looking at the pictures that
    # tutorial's recipe produces.
    with_a_negative_prompt = {
        SweepParameter.NEGATIVE_PROMPT,
        SweepParameter.STEP_PREVIEW,
        SweepParameter.STEP_PREVIEW_GUESS,
        SweepParameter.CONTROL_STRENGTH,
        SweepParameter.CONTROLNET_GALLERY,
        SweepParameter.QR_STYLE,
    }
    other_jobs = [job for job in spec.expand_jobs() if job.parameter not in with_a_negative_prompt]
    assert all(job.negative_prompt == "" for job in other_jobs)


def test_the_karras_strip_uses_the_step_count_where_its_effect_is_visible(spec: SweepSpec) -> None:
    """Assert the schedule pair is rendered at 10 steps, not at the anchor's 30, where the difference is hidden."""
    karras_jobs = [job for job in spec.expand_jobs() if job.parameter is SweepParameter.KARRAS]
    assert [job.karras for job in karras_jobs] == [False, True]
    assert all(job.steps == 10 for job in karras_jobs)
    assert spec.anchor.steps == 30


def test_every_scene_belongs_to_one_of_the_three_anchor_worlds(spec: SweepSpec) -> None:
    """Assert the library has one cast rather than a new subject per lesson.

    Two images are only comparable when they are of the same world, so every scene has to be set in
    the skyharbor, the fox shrine or the rooftop garden, and name it in its own prompt.
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
        "skyharbor-tower-ink",
        "shrine-gate-ink",
        "glasshouse-ink",
        "skyharbor-poster-ink",
        "shrine-gate-flat-ink",
        "robot-gardener-flat-ink",
    }
    world_markers = ("skyharbor", "zeppelin", "shrine", "gardening robot", "rooftop")

    # The QR gallery's scene is a treatment rather than a place: it is three worlds in one strip,
    # so each of its stops names its own world in the fragment and the scene prompt does not name any.
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


def test_the_control_strength_strips_vary_only_the_strength(spec: SweepSpec) -> None:
    """Assert every strength strip varies its ControlNet strength and holds everything else, seed included.

    A stop that reached a scannable code by changing the seed as well as the strength would show
    two changes at once, so each strip uses one seed across its whole published range. Every
    control type that the horde serves has its own strip on each of world 1's three source
    subjects, all with the same strengths, so each detector's hold can be compared with every
    other detector, and with the same detector on a different kind of picture. The QR strip uses a
    narrower range on the QR workflow, which does not condition on a detector.
    """
    detector_ladder = [
        0.01,
        0.05,
        0.15,
        0.25,
        0.375,
        0.5,
        0.625,
        0.75,
        0.875,
        1.0,
        1.125,
        1.25,
        1.375,
        1.5,
        1.75,
        2.0,
        2.25,
        2.5,
        2.75,
        3.0,
    ]
    qr_ladder = [1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6]
    detector_order = [
        "canny",
        "hed",
        "pidinet",
        "teed",
        "pyracanny",
        "binary",
        "scribble",
        "fakescribbles",
        "scribble_xdog",
        "scribble_pidinet",
        "standard_lineart",
        "lineart",
        "lineart_anime",
        "lineart_anime_denoise",
        "mlsd",
        "depth",
        "midas_depth",
        "zoe_depth",
        "depth_anything",
        "depth_anything_v2",
        "normal",
        "normal_bae",
        "openpose",
        "seg",
        "oneformer_ade20k",
        "oneformer_coco",
        "color",
        "shuffle",
        "recolor_luminance",
        "recolor_intensity",
        "tile",
        "tile_ttplanet_guided",
        "tile_ttplanet_simple",
    ]

    subject_scenes = {
        SweepSubject.ANCHOR: "courier-ink",
        SweepSubject.BUILDING: "skyharbor-tower-ink",
        SweepSubject.FLAT: "skyharbor-poster-ink",
    }

    strips = [strip for strip in spec.expand_strips() if strip.parameter is SweepParameter.CONTROL_STRENGTH]

    assert len(strips) == len(detector_order) * len(subject_scenes) + 1

    for strip in strips:
        assert len({job.seed for job in strip.jobs}) == 1
        assert len({job.prompt for job in strip.jobs}) == 1
        assert len({job.control_type for job in strip.jobs}) == 1

    qr_strips = [strip for strip in strips if strip.jobs[0].workflow is WorkflowName.QR_CODE]
    detector_strips = [strip for strip in strips if strip.jobs[0].workflow is not WorkflowName.QR_CODE]

    assert len(qr_strips) == 1
    assert all(job.control_type is None for job in qr_strips[0].jobs)
    assert [job.control_strength for job in qr_strips[0].jobs] == qr_ladder

    assert len(detector_strips) == len(detector_order) * len(subject_scenes)
    for detector_strip in detector_strips:
        assert [job.control_strength for job in detector_strip.jobs] == detector_ladder
        assert len({job.model_slug for job in detector_strip.jobs}) == 1
        assert len({job.scene_id for job in detector_strip.jobs}) == 1

    assert {strip.jobs[0].model_slug for strip in detector_strips} == {"deliberate"}
    for subject, scene_id in subject_scenes.items():
        of_subject = [strip for strip in detector_strips if strip.subject is subject]
        assert [strip.jobs[0].control_type for strip in of_subject] == detector_order
        assert {strip.jobs[0].scene_id for strip in of_subject} == {scene_id}


def test_a_control_map_is_filed_beside_the_render_it_guided(spec: SweepSpec) -> None:
    """Assert a job states a map path when it conditions on a detector, and does not state one otherwise.

    The map and the render are only meaningful as a pair, so the map's path follows from the
    render's path: same directory, same stem.
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


def test_a_strip_with_one_detector_files_its_map_beside_its_source(spec: SweepSpec) -> None:
    """Assert a strip states one map path when every stop shares a detector, and does not state one otherwise.

    A reader of a strength strip judges how strongly the ControlNet holds the render to a map,
    which is not possible without the map. It is the same map at every stop, so the strip
    publishes it once, beside its source.
    """
    strips = spec.expand_strips()

    unconditioned = next(strip for strip in strips if strip.jobs[0].control_type is None)
    assert unconditioned.relative_control_map_path is None

    held = next(strip for strip in strips if strip.parameter is SweepParameter.CONTROL_STRENGTH and strip.source_image)
    assert len({job.control_type for job in held.jobs}) == 1

    published_source = held.published_source_image
    map_path = held.relative_control_map_path
    assert published_source is not None
    assert map_path is not None
    assert map_path == f"{published_source.removesuffix('.webp')}_map.webp"

    changing_detector = next(strip for strip in strips if len({job.control_type for job in strip.jobs}) > 1)
    assert changing_detector.relative_control_map_path is None


def test_the_qr_gallery_states_the_layout_knobs_named_in_its_captions(spec: SweepSpec) -> None:
    """Assert each gallery stop states the knobs that its caption names, and curates from a batch."""
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

    The checked-in library is too large to test a rule against directly, because a test that
    modifies it tests the modification as much as the rule. This builds the smallest valid
    specification, so a raised error can only come from the sweep under test.
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
    """Return a two-stop strip that sweeps the hires pass strength, holding the requested values."""
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
            assert job.hires_fix_denoising_strength is None, (
                f"{job.relative_image_path} states a hires strength it does not teach"
            )


def test_a_hires_strength_without_the_hires_pass_is_rejected() -> None:
    """Assert the specification cannot publish a value that is not read by any backend.

    Only the second pass reads this strength. Stating it on a job without a second pass would
    render a strip whose stops are identical, while its captions describe differences.
    """
    with pytest.raises(ValueError, match="without requesting the hires pass"):
        SweepSpec.model_validate(one_sweep_spec(hires_denoise_sweep(width=768, height=768)))


def test_the_hires_pass_is_rejected_where_the_backends_disable_it() -> None:
    """Assert both silent-disable rules are caught: at or below native size, and while painting."""
    with pytest.raises(ValueError, match="silently disable it unless both dimensions exceed"):
        SweepSpec.model_validate(one_sweep_spec(hires_denoise_sweep(hires_fix=True, width=512, height=768)))

    painting = hires_denoise_sweep(hires_fix=True, width=768, height=768)
    painting["source_image"] = "inputs/skyharbor-deck-sunset.webp"
    painting["holds"]["source_processing"] = "inpainting"
    with pytest.raises(ValueError, match="requests hires fix while inpainting"):
        SweepSpec.model_validate(one_sweep_spec(painting))


def test_a_stop_may_start_from_its_own_picture() -> None:
    """Assert a stop can override the sweep's source, as outpainting extents need.

    The backends do not pad a canvas, so each extent is a picture that was padded before it
    was sent. The stops of such a strip do not share an input, and the sweep's source is still
    the picture that a reader sees as the starting point.
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


def steps_sweep(*, preview: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a step sweep whose stops are on both sides of the anchor's step count."""
    sweep: dict[str, Any] = {
        "kind": "strip",
        "parameter": "steps",
        "rationale": "A ladder for a preview to be resolved against.",
        "caption": "More steps each time.",
        "model_slugs": ["small"],
        "scene_ids": ["thing"],
        "stops": [
            {"value": 4, "caption": "Barely started."},
            {"value": 28, "caption": "Almost the held count."},
            {"value": 50, "caption": "Past where it helps."},
        ],
    }
    if preview is not None:
        sweep["preview"] = preview
    return sweep


def preview_of(sweep: dict[str, Any]) -> tuple[int, int]:
    """Return the resolved preview positions for the only strip of a specification built from this sweep."""
    (strip,) = SweepSpec.model_validate(one_sweep_spec(sweep)).expand_strips()
    return resolve_preview_indices(
        preview=strip.preview,
        stop_values=[job.stop_value for job in strip.jobs],
        default_value=strip.default_value,
    )


def test_an_unstated_preview_shows_the_held_value_against_the_top() -> None:
    """Assert a strip's default preview pair is its reference value, then the top stop.

    The anchor holds 30 steps, which is not one of the stops, so the preview starts from the
    nearest stop.
    """
    assert preview_of(steps_sweep()) == (1, 2)


def test_a_preview_may_name_its_stops_by_value() -> None:
    assert preview_of(steps_sweep(preview={"left": 4, "right": 28})) == (0, 1)
    assert preview_of(steps_sweep(preview={"left": "first", "right": "last"})) == (0, 2)


def test_a_default_that_is_not_a_stop_previews_the_first_stop() -> None:
    """Assert a strip whose held value is not one of its stops previews from the first stop.

    Sampler names do not have a distance between them, so a held value that does not match a
    stop makes the preview start from the first stop instead of an arbitrary neighbour.
    """
    sampler_sweep = {
        "kind": "strip",
        "parameter": "sampler",
        "rationale": "A ladder the held sampler is not on.",
        "caption": "One sampler each.",
        "model_slugs": ["small"],
        "scene_ids": ["thing"],
        "stops": [
            {"value": "k_euler", "caption": "The plain one."},
            {"value": "k_dpmpp_2m", "caption": "A converging one."},
        ],
    }
    assert preview_of(sampler_sweep) == (0, 1)


def test_a_parameter_with_no_held_value_previews_from_the_first_stop() -> None:
    """Assert a strip for a parameter without a held value previews its first stop against its last."""
    seed_sweep = {
        "kind": "strip",
        "parameter": "seed",
        "rationale": "Nothing in the specification says what a seed defaults to.",
        "caption": "One seed each.",
        "model_slugs": ["small"],
        "scene_ids": ["thing"],
        "stops": [
            {"value": "Seed A", "caption": "One.", "seed": 1},
            {"value": "Seed B", "caption": "Another.", "seed": 2},
            {"value": "Seed C", "caption": "A third.", "seed": 3},
        ],
    }
    (strip,) = SweepSpec.model_validate(one_sweep_spec(seed_sweep)).expand_strips()
    assert strip.default_value is None
    assert preview_of(seed_sweep) == (0, 2)


def test_a_preview_naming_a_stop_the_strip_does_not_render_is_rejected() -> None:
    """Assert a preview cannot point at a value that is not a stop, on either side."""
    with pytest.raises(ValueError, match="right side names 33"):
        SweepSpec.model_validate(one_sweep_spec(steps_sweep(preview={"left": 4, "right": 33})))

    with pytest.raises(ValueError, match="left side names 33"):
        SweepSpec.model_validate(one_sweep_spec(steps_sweep(preview={"left": 33, "right": "last"})))


def test_a_preview_side_cannot_take_the_other_side_s_anchor() -> None:
    """Assert the two sides keep their own vocabulary, so a preview cannot wipe a stop onto itself."""
    with pytest.raises(ValueError, match="left side cannot be last"):
        SweepSpec.model_validate(one_sweep_spec(steps_sweep(preview={"left": "last", "right": "last"})))

    with pytest.raises(ValueError, match="right side cannot be first"):
        SweepSpec.model_validate(one_sweep_spec(steps_sweep(preview={"left": "first", "right": "first"})))


def test_a_grid_cannot_state_a_preview(spec: SweepSpec) -> None:
    """Assert a two-dimensional sweep cannot have a preview: it is not read by wiping."""
    grid = {
        "kind": "grid",
        "parameter": "karras_grid",
        "rationale": "A grid to try a preview on.",
        "caption": "Two axes.",
        "model_slugs": ["small"],
        "scene_ids": ["thing"],
        "rows": {"parameter": "steps", "values": [8, 30], "caption": "Steps"},
        "columns": {"parameter": "karras", "values": [False, True], "caption": "Karras"},
        "cell_caption": "{row} steps, karras {column}.",
        "preview": {"left": "first", "right": "last"},
    }
    with pytest.raises(ValueError, match="preview"):
        SweepSpec.model_validate(one_sweep_spec(grid))

    assert all(strip.preview is None for strip in spec.expand_strips() if strip.grid_rows is not None)


def test_every_strip_in_the_library_previews_two_different_stops(spec: SweepSpec) -> None:
    """Assert no strip's collapsed card would wipe a picture onto itself.

    A pair that resolves to one stop twice is a useless card, and it happens without a warning: a
    strip whose top stop is also its reference value resolves both sides to the same stop unless
    the sweep states a preview.
    """
    for strip in spec.expand_strips():
        if strip.presentation not in PREVIEWABLE_PRESENTATIONS or len(strip.jobs) < 2:
            continue

        left_index, right_index = resolve_preview_indices(
            preview=strip.preview,
            stop_values=[job.stop_value for job in strip.jobs],
            default_value=strip.default_value,
        )
        assert left_index != right_index, f"{strip.strip_id} previews one stop against itself"


def test_the_control_strength_strips_preview_the_default_strength_against_the_top(
    spec: SweepSpec,
) -> None:
    """Assert a strength strip previews the default strength, used when a job does not send one, against the top.

    The specification does not hold a ControlNet strength, so the strip's reference value is the
    backend's default.
    """
    strength_strips = [
        strip
        for strip in spec.expand_strips()
        if strip.parameter is SweepParameter.CONTROL_STRENGTH and strip.directory_override is not None
    ]
    assert len(strength_strips) == 99

    for strip in strength_strips:
        stop_values = [job.stop_value for job in strip.jobs]
        left_index, right_index = resolve_preview_indices(
            preview=strip.preview,
            stop_values=stop_values,
            default_value=strip.default_value,
        )
        assert (stop_values[left_index], stop_values[right_index]) == (1, 3)
