# SPDX-FileCopyrightText: 2026 Tazlin
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Check the parts a campaign depends on: resumption, output layout, and the index."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import override

import pytest
from PIL import Image

from param_sweeps import horde_client
from param_sweeps.campaign import JobFilter, run_campaign
from param_sweeps.horde_client import ENV_FILE_VARIABLE
from param_sweeps.horde_runner import (
    API_KEY_VARIABLE,
    HordeGenerationRequest,
    MissingApiKeyError,
    default_env_file,
    read_api_key,
)
from param_sweeps.local_runner import GpuBackend, HordelibPayload, resolve_gpu_backend
from param_sweeps.outputs import INDEX_FILENAME, SweepIndex, build_index
from param_sweeps.runner import FetchPlanEntry, RenderOutcome, SweepRunner
from param_sweeps.spec import ModelSpec, RenderJob, StripPresentation, SweepParameter, SweepSpec, load_spec

SPEC_PATH = Path(__file__).resolve().parents[1] / "sweep_spec.json"

ONE_STRIP = JobFilter(parameters=("clip_skip",), model_slugs=("icbinp",), scene_ids=("person",))
EVERY_IMAGE_TO_IMAGE_STRIP = JobFilter(parameters=("denoising_strength",))
ONE_GALLERY = JobFilter(parameters=("controlnet_gallery",), model_slugs=("deliberate",), scene_ids=("courier-ink",))


class CountingRunner(SweepRunner):
    """A runner that produces a solid colour, so the driver can be tested without a GPU."""

    def __init__(self, *, fail_on: str | None = None) -> None:
        """Create a runner that renders every job, except one that is asked to fail."""
        self.rendered_paths: list[str] = []
        self.prepare_count = 0
        self._fail_on = fail_on

    @property
    @override
    def name(self) -> str:
        return "test-runner"

    @override
    def fetch_plan(self, jobs: Sequence[RenderJob], models: Sequence[ModelSpec]) -> list[FetchPlanEntry]:
        return [
            FetchPlanEntry(
                model_name=image_model.horde_name,
                approximate_size_gb=image_model.approximate_download_size_gb,
                already_present=False,
                image_count=len(jobs),
            )
            for image_model in models
        ]

    @override
    def prepare(self, jobs: Sequence[RenderJob], models: Sequence[ModelSpec]) -> None:
        self.prepare_count += 1

    @override
    def render(self, job: RenderJob) -> RenderOutcome:
        if self._fail_on is not None and job.relative_image_path == self._fail_on:
            raise RuntimeError("deliberate failure")

        self.rendered_paths.append(job.relative_image_path)
        return RenderOutcome(
            images=[Image.new("RGB", (16, 16), color=(20, 40, 60)) for _ in job.candidate_numbers],
        )


@pytest.fixture(scope="module")
def spec() -> SweepSpec:
    return load_spec(SPEC_PATH)


def test_campaign_writes_images_sidecars_and_index(spec: SweepSpec, tmp_path: Path) -> None:
    runner = CountingRunner()
    result = run_campaign(spec=spec, runner=runner, output_root=tmp_path, job_filter=ONE_STRIP)

    assert result.rendered_count == 3
    assert result.failed_jobs == ()

    image_path = tmp_path / "clip-skip/icbinp/person/002_2.webp"
    sidecar_path = tmp_path / "clip-skip/icbinp/person/002_2.json"
    assert image_path.is_file()

    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["params"]["clip_skip"] == 2
    assert sidecar["params"]["seed"] == spec.seed
    assert sidecar["runner"] == "test-runner"
    assert sidecar["model_requested"] == "ICBINP - I Can't Believe It's Not Photography"

    index = SweepIndex.model_validate_json((tmp_path / INDEX_FILENAME).read_text(encoding="utf-8"))
    assert len(index.strips) == 1
    strip = index.strips[0]
    assert strip.complete is True
    assert [stop.value for stop in strip.stops] == [1, 2, 3]
    assert strip.varying == ["clip_skip"]
    assert all(stop.caption for stop in strip.stops)


def test_a_second_run_renders_nothing_new(spec: SweepSpec, tmp_path: Path) -> None:
    first_result = run_campaign(spec=spec, runner=CountingRunner(), output_root=tmp_path, job_filter=ONE_STRIP)

    resumed_runner = CountingRunner()
    second_result = run_campaign(spec=spec, runner=resumed_runner, output_root=tmp_path, job_filter=ONE_STRIP)

    assert first_result.rendered_count == 3
    assert second_result.rendered_count == 0
    assert second_result.skipped_count == 3
    assert resumed_runner.rendered_paths == []


def test_an_interrupted_run_resumes_at_the_failed_image(spec: SweepSpec, tmp_path: Path) -> None:
    failing_path = "clip-skip/icbinp/person/002_2.webp"
    first_result = run_campaign(
        spec=spec,
        runner=CountingRunner(fail_on=failing_path),
        output_root=tmp_path,
        job_filter=ONE_STRIP,
    )
    assert first_result.rendered_count == 2
    assert first_result.failed_jobs == (failing_path,)

    partial_index = SweepIndex.model_validate_json((tmp_path / INDEX_FILENAME).read_text(encoding="utf-8"))
    assert partial_index.strips[0].complete is False

    resumed_runner = CountingRunner()
    second_result = run_campaign(spec=spec, runner=resumed_runner, output_root=tmp_path, job_filter=ONE_STRIP)

    assert resumed_runner.rendered_paths == [failing_path]
    assert second_result.rendered_count == 1
    assert second_result.skipped_count == 2

    completed_index = SweepIndex.model_validate_json((tmp_path / INDEX_FILENAME).read_text(encoding="utf-8"))
    assert completed_index.strips[0].complete is True


def test_deleting_an_image_makes_the_next_run_regenerate_it(spec: SweepSpec, tmp_path: Path) -> None:
    run_campaign(spec=spec, runner=CountingRunner(), output_root=tmp_path, job_filter=ONE_STRIP)
    (tmp_path / "clip-skip/icbinp/person/001_1.webp").unlink()

    resumed_runner = CountingRunner()
    run_campaign(spec=spec, runner=resumed_runner, output_root=tmp_path, job_filter=ONE_STRIP)

    assert resumed_runner.rendered_paths == ["clip-skip/icbinp/person/001_1.webp"]


def test_plan_only_generates_nothing(spec: SweepSpec, tmp_path: Path) -> None:
    runner = CountingRunner()
    result = run_campaign(spec=spec, runner=runner, output_root=tmp_path, job_filter=ONE_STRIP, plan_only=True)

    assert result.rendered_count == 0
    assert runner.prepare_count == 0
    assert runner.rendered_paths == []


def test_an_empty_filter_selection_is_rejected(spec: SweepSpec, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="selected no jobs"):
        run_campaign(
            spec=spec,
            runner=CountingRunner(),
            output_root=tmp_path,
            job_filter=JobFilter(model_slugs=("no-such-model",)),
        )


def test_index_omits_strips_with_no_images(spec: SweepSpec, tmp_path: Path) -> None:
    empty_index = build_index(spec=spec, output_root=tmp_path)
    assert empty_index.strips == []


def test_each_image_to_image_strip_publishes_the_source_it_starts_from(spec: SweepSpec, tmp_path: Path) -> None:
    """Assert every image-to-image strip carries its own source, so a consumer can show it beside the strip."""
    run_campaign(spec=spec, runner=CountingRunner(), output_root=tmp_path, job_filter=EVERY_IMAGE_TO_IMAGE_STRIP)

    index = SweepIndex.model_validate_json((tmp_path / INDEX_FILENAME).read_text(encoding="utf-8"))
    published: list[str] = []

    for strip in index.strips:
        assert strip.source_image is not None
        assert strip.source_image == f"{strip.strip_id}/source.webp"
        assert (tmp_path / strip.source_image).is_file()
        assert strip.varying == ["denoising_strength"]
        published.append(strip.source_image)

    assert len(set(published)) == len(index.strips)

    sources_used = {
        json.loads((tmp_path / strip.stops[0].sidecar).read_text(encoding="utf-8"))["source_image"]
        for strip in index.strips
    }
    assert sources_used == {
        "inputs/skyharbor-deck-sunset.webp",
        "inputs/rooftop-garden-sketch.webp",
    }


def test_the_index_leaves_per_stop_parameters_to_the_sidecars(spec: SweepSpec, tmp_path: Path) -> None:
    """Assert a stop carries only what a strip is drawn and captioned from.

    The full parameter set is in the sidecar beside every image; repeating it per stop trebled
    an index that every visitor to the Learn pages downloads.
    """
    run_campaign(spec=spec, runner=CountingRunner(), output_root=tmp_path, job_filter=ONE_STRIP)

    index_content = json.loads((tmp_path / INDEX_FILENAME).read_text(encoding="utf-8"))
    stop = index_content["strips"][0]["stops"][0]
    assert set(stop) == {
        "value",
        "label",
        "caption",
        "file",
        "sidecar",
        "width",
        "height",
        "crop",
        "fit",
        "prompt_fragment",
        "batch_size",
        "chosen_candidate",
        "control_type",
        "control_group",
        "control_map",
    }
    assert stop["prompt_fragment"] is None
    assert stop["batch_size"] == 1
    assert (stop["width"], stop["height"]) == (16, 16)

    sidecar = json.loads((tmp_path / stop["sidecar"]).read_text(encoding="utf-8"))
    assert sidecar["params"]["clip_skip"] == 1


def test_a_triptych_stop_names_its_detector_and_the_map_it_drew(spec: SweepSpec, tmp_path: Path) -> None:
    """Assert a ControlNet gallery's stops carry their control type, family and map into the index.

    The map is half of what the stop teaches, so a consumer reads it from the index rather than
    guessing at a path. A map that was never drawn is reported as absent rather than as a path to
    a file that is not there: the render exists either way, and the gap is what the page says.
    """
    run_campaign(spec=spec, runner=CountingRunner(), output_root=tmp_path, job_filter=ONE_GALLERY)

    index = SweepIndex.model_validate_json((tmp_path / INDEX_FILENAME).read_text(encoding="utf-8"))
    (gallery,) = index.strips

    assert gallery.presentation is StripPresentation.TRIPTYCH
    assert gallery.source_image is not None
    assert all(stop.control_type for stop in gallery.stops)
    assert all(stop.control_group for stop in gallery.stops)
    assert all(stop.control_map is None for stop in gallery.stops), "no detector has drawn anything yet"

    drawn = next(job for job in spec.expand_jobs() if job.relative_image_path == gallery.stops[0].file)
    map_path = drawn.relative_control_map_path
    assert map_path is not None
    Image.new("RGB", (16, 16), color=(0, 0, 0)).save(tmp_path / map_path, format="WEBP")

    rebuilt = build_index(spec=spec, output_root=tmp_path)
    (rebuilt_gallery,) = [strip for strip in rebuilt.strips if strip.strip_id == gallery.strip_id]
    published_maps = [stop.control_map for stop in rebuilt_gallery.stops if stop.control_map is not None]

    assert published_maps == [map_path]


def test_a_strength_strip_names_the_map_its_dial_holds_the_render_to(spec: SweepSpec, tmp_path: Path) -> None:
    """Assert a strip held to one detector carries its map into the index once the map is on disk.

    A map that has not been drawn is reported as absent rather than as a path to a file that is
    not there, for the same reason a stop's map is: the renders are on disk either way.
    """
    strip = next(candidate for candidate in spec.expand_strips() if candidate.relative_control_map_path is not None)
    for job in strip.jobs:
        image_path = tmp_path / job.relative_image_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16), color=(0, 0, 0)).save(image_path, format="WEBP")

    (built,) = [
        entry for entry in build_index(spec=spec, output_root=tmp_path).strips if entry.strip_id == strip.strip_id
    ]
    assert built.control_map is None, "no detector has drawn anything yet"

    map_path = strip.relative_control_map_path
    assert map_path is not None
    (tmp_path / map_path).parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=(0, 0, 0)).save(tmp_path / map_path, format="WEBP")

    (rebuilt,) = [
        entry for entry in build_index(spec=spec, output_root=tmp_path).strips if entry.strip_id == strip.strip_id
    ]
    assert rebuilt.control_map == map_path
    assert all(stop.control_map is None for stop in rebuilt.stops), "the strip's map stands for every stop"


def test_hordelib_payload_asks_for_img2img_only_when_the_job_starts_from_an_image(spec: SweepSpec) -> None:
    text_to_image_job = next(job for job in spec.expand_jobs() if job.parameter is SweepParameter.STEPS)
    image_to_image_job = next(job for job in spec.expand_jobs() if job.parameter is SweepParameter.DENOISING_STRENGTH)

    text_to_image_payload = HordelibPayload.from_job(text_to_image_job).as_hordelib_mapping()
    assert "source_processing" not in text_to_image_payload
    assert "source_image" not in text_to_image_payload

    source = Image.new("RGB", (16, 16), color=(10, 20, 30))
    image_to_image_payload = HordelibPayload.from_job(image_to_image_job, source_image=source).as_hordelib_mapping()
    assert image_to_image_payload["source_processing"] == "img2img"
    assert image_to_image_payload["source_image"] is source
    assert image_to_image_payload["denoising_strength"] == image_to_image_job.denoising_strength


def test_the_api_request_carries_the_negative_prompt_the_way_the_api_takes_it(spec: SweepSpec) -> None:
    """Assert the negative prompt is appended with ``###``, which is the API's only way to send one."""
    with_negative = next(
        job for job in spec.expand_jobs() if job.parameter is SweepParameter.NEGATIVE_PROMPT and job.stop_value is True
    )
    without_negative = next(
        job
        for job in spec.expand_jobs()
        if job.parameter is SweepParameter.NEGATIVE_PROMPT and job.stop_value is False
    )

    assert HordeGenerationRequest.from_job(with_negative).prompt.endswith(f" ### {with_negative.negative_prompt}")
    assert HordeGenerationRequest.from_job(without_negative).prompt == without_negative.prompt
    assert "###" not in HordeGenerationRequest.from_job(without_negative).prompt


def test_hordelib_payload_renames_the_fields_hordelib_renames(spec: SweepSpec) -> None:
    job = next(job for job in spec.expand_jobs() if job.parameter is SweepParameter.STEPS)
    payload = HordelibPayload.from_job(job).as_hordelib_mapping()

    assert payload["ddim_steps"] == job.steps
    assert payload["n_iter"] == 1
    assert payload["model"] == job.horde_model_name
    assert payload["seed"] == job.seed
    assert "steps" not in payload


def test_gpu_backend_defaults_to_cuda_and_rejects_nonsense(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PARAM_SWEEPS_GPU_BACKEND", raising=False)
    assert resolve_gpu_backend(requested=None) is GpuBackend.CUDA
    assert resolve_gpu_backend(requested="rocm") is GpuBackend.ROCM

    monkeypatch.setenv("PARAM_SWEEPS_GPU_BACKEND", "xpu")
    assert resolve_gpu_backend(requested=None) is GpuBackend.XPU
    assert resolve_gpu_backend(requested="cpu") is GpuBackend.CPU

    with pytest.raises(ValueError, match="unsupported GPU backend"):
        resolve_gpu_backend(requested="quantum")


def test_api_key_is_read_from_an_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(API_KEY_VARIABLE, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(f'# a comment\n{API_KEY_VARIABLE}="secret-value"\n', encoding="utf-8")

    assert read_api_key(env_file=env_file) == "secret-value"


def test_a_missing_api_key_names_only_the_location(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(API_KEY_VARIABLE, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("SOMETHING_ELSE=1\n", encoding="utf-8")

    with pytest.raises(MissingApiKeyError) as raised:
        read_api_key(env_file=env_file)

    assert str(env_file) in str(raised.value)


def test_the_env_file_variable_wins_over_the_repository_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    (repository_root / ".env").write_text("SOMETHING=1\n", encoding="utf-8")
    named_file = tmp_path / "elsewhere.env"
    monkeypatch.setattr(horde_client, "TOOL_ROOT", repository_root)
    monkeypatch.setenv(ENV_FILE_VARIABLE, str(named_file))

    assert default_env_file() == named_file


def test_the_repository_env_is_used_when_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".env").write_text("SOMETHING=1\n", encoding="utf-8")
    monkeypatch.setattr(horde_client, "TOOL_ROOT", tmp_path)
    monkeypatch.setenv(ENV_FILE_VARIABLE, "  ")

    assert default_env_file() == tmp_path / ".env"


def test_there_is_no_default_env_file_without_the_variable_or_a_repository_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(horde_client, "TOOL_ROOT", tmp_path)
    monkeypatch.delenv(ENV_FILE_VARIABLE, raising=False)

    assert default_env_file() is None


def test_the_index_carries_the_pair_a_collapsed_card_previews(spec: SweepSpec, tmp_path: Path) -> None:
    """Assert every strip the index publishes names two stops to wipe between, and the grids none.

    The pair is resolved here rather than by the consumer, because it depends on the value the
    strip is read against, which the index does not otherwise carry.
    """
    run_campaign(
        spec=spec,
        runner=CountingRunner(),
        output_root=tmp_path,
        job_filter=JobFilter(parameters=("clip_skip", "karras_grid", "tiling"), model_slugs=("icbinp",)),
    )

    index = SweepIndex.model_validate_json((tmp_path / INDEX_FILENAME).read_text(encoding="utf-8"))
    previews = {strip.presentation: strip.preview for strip in index.strips}
    assert set(previews) == {StripPresentation.STRIP, StripPresentation.GRID, StripPresentation.TILE_REPEAT_3X3}

    assert previews[StripPresentation.GRID] is None
    assert previews[StripPresentation.TILE_REPEAT_3X3] is None

    (clip_skip,) = [strip for strip in index.strips if strip.parameter is SweepParameter.CLIP_SKIP]
    assert clip_skip.preview is not None
    assert clip_skip.stops[clip_skip.preview.left_index].value == 1, "the layer count this model was trained for"
    assert clip_skip.stops[clip_skip.preview.right_index].value == 3


def test_a_preview_falls_back_to_the_stops_that_were_actually_rendered(spec: SweepSpec, tmp_path: Path) -> None:
    """Assert a strip missing its top stop still previews two pictures the index carries.

    The index leaves out a stop whose image is not on disk, so a preview resolved against the
    declared ladder could point past the end of a partly rendered strip.
    """
    strip = next(
        candidate
        for candidate in spec.expand_strips()
        if candidate.parameter is SweepParameter.CLIP_SKIP and candidate.image_model.slug == "icbinp"
    )
    for job in strip.jobs[:-1]:
        image_path = tmp_path / job.relative_image_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16), color=(0, 0, 0)).save(image_path, format="WEBP")

    (built,) = [
        entry for entry in build_index(spec=spec, output_root=tmp_path).strips if entry.strip_id == strip.strip_id
    ]
    assert not built.complete
    assert built.preview is not None
    assert built.preview.right_index == len(built.stops) - 1
    assert built.stops[built.preview.right_index].value == 2
