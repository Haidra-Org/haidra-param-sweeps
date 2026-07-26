# SPDX-FileCopyrightText: 2026 Tazlin
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The step-preview capture: what it stashes, what it puts back, and what it refuses.

Nothing here renders or touches an accelerator. The seams the capture wraps are stated as fakes
that call the wrapper the way ComfyUI calls the real one, which is what lets the arming, the
per-pass separation and the restoration be proven rather than assumed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from param_sweeps.campaign import JobFilter, select_jobs
from param_sweeps.spec import SweepParameter, load_spec
from param_sweeps.step_preview import (
    ARC_PARAMETERS,
    CaptureTargets,
    LatentArc,
    armed_step_capture,
    check_arcs_agree,
    decode_latent_frames,
    step_preview_strips,
)

SPEC_PATH = Path(__file__).resolve().parents[1] / "sweep_spec.json"

TOTAL_STEPS = 3


class FakeLatent:
    """A tensor-shaped stand-in that records the copy the capture makes of it."""

    def __init__(self, name: str) -> None:
        """Name a latent so the arc it was stashed into can be read back."""
        self.name = name
        self.is_nested = False
        self.was_detached = False
        self.was_copied_to_host = False

    def detach(self) -> FakeLatent:
        """Return a copy, recording that the capture detached this latent."""
        copy = FakeLatent(self.name)
        copy.was_detached = True
        return copy

    def float(self) -> FakeLatent:
        """Return the same latent: the fake carries no dtype to widen."""
        return self

    def cpu(self) -> FakeLatent:
        """Return the same latent, recording that it was asked onto the host."""
        self.was_copied_to_host = True
        return self

    def clone(self) -> FakeLatent:
        """Return the same latent: the fake shares no buffer to clone away from."""
        return self


class FakeModel:
    """Stands in for the inner model, whose ``process_latent_out`` leaves the sampler's space."""

    def __init__(self) -> None:
        """Open a model that has converted nothing yet."""
        self.converted: list[str] = []

    def process_latent_out(self, latent: FakeLatent) -> FakeLatent:
        """Return the latent renamed, so the conversion is visible in what was stashed."""
        self.converted.append(latent.name)
        latent.name = f"{latent.name}_out"
        return latent


class FakeModelPatcher:
    """Stands in for the patcher ``prepare_callback`` is handed, which carries the inner model."""

    def __init__(self) -> None:
        """Open a patcher around a fresh inner model."""
        self.model = FakeModel()


class FakeLatentPreviewModule:
    """Stands in for ComfyUI's ``latent_preview``, recording the callbacks it was asked for."""

    def __init__(self) -> None:
        """Open a module that has been asked for no callbacks yet."""
        self.prepared_step_counts: list[int] = []
        self.delegated_steps: list[int] = []

    def prepare_callback(self, model: Any, steps: int, x0_output_dict: Any = None) -> Any:
        """Return a callback that records the steps it was delegated, as the real one previews them."""
        self.prepared_step_counts.append(steps)

        def callback(step: int, x0: Any, x: Any, total_steps: int) -> None:
            self.delegated_steps.append(step)

        return callback


class FakeDecodeNode:
    """Stands in for ComfyUI's ``VAEDecode``, whose ``decode`` the graph hands its VAE to."""

    def decode(self, vae: Any, samples: Any) -> tuple[str]:
        """Return a stand-in decode result, in the one-tuple a ComfyUI node returns."""
        return ("decoded",)


class FakeVae:
    """A VAE whose decode returns a stated tensor, so a decoded frame can be checked exactly."""

    def __init__(self, pixels: Any) -> None:
        """State the pixels every decode returns."""
        self.pixels = pixels
        self.decoded_latents: list[Any] = []

    def decode(self, latent: Any) -> Any:
        """Return the stated pixels, recording which latent was decoded."""
        self.decoded_latents.append(latent)
        return self.pixels


def _targets() -> tuple[CaptureTargets, FakeLatentPreviewModule, Any]:
    latent_preview_module = FakeLatentPreviewModule()
    original_decode = FakeDecodeNode.decode
    return (
        CaptureTargets(latent_preview_module=latent_preview_module, decode_node_class=FakeDecodeNode),
        latent_preview_module,
        original_decode,
    )


def _sample(latent_preview_module: FakeLatentPreviewModule, *, steps: int, pass_name: str) -> FakeModelPatcher:
    """Drive the wrapped seam the way ComfyUI's sampling path drives the real one."""
    model = FakeModelPatcher()
    callback = latent_preview_module.prepare_callback(model, steps)
    for step in range(steps):
        callback(step, FakeLatent(f"{pass_name}_x0_{step}"), FakeLatent(f"{pass_name}_x_{step}"), steps)

    return model


def test_a_capture_stashes_both_arcs_in_step_order() -> None:
    targets, latent_preview_module, _ = _targets()

    with armed_step_capture(targets=targets) as capture:
        _sample(latent_preview_module, steps=TOTAL_STEPS, pass_name="first")
        FakeDecodeNode().decode(vae="the vae", samples={"samples": "latent"})

    sampler_pass = capture.single_pass()
    assert sampler_pass.requested_steps == TOTAL_STEPS
    assert sampler_pass.total_steps == TOTAL_STEPS
    assert [latent.name for latent in sampler_pass.latents_for(LatentArc.CANVAS)] == [
        "first_x_0_out",
        "first_x_1_out",
        "first_x_2_out",
    ]
    assert [latent.name for latent in sampler_pass.latents_for(LatentArc.GUESS)] == [
        "first_x0_0_out",
        "first_x0_1_out",
        "first_x0_2_out",
    ]
    assert capture.require_vae() == "the vae"


def test_every_stashed_latent_leaves_the_space_the_sampler_works_in() -> None:
    targets, latent_preview_module, _ = _targets()

    with armed_step_capture(targets=targets):
        model = _sample(latent_preview_module, steps=TOTAL_STEPS, pass_name="first")

    assert model.model.converted == [
        "first_x_0",
        "first_x0_0",
        "first_x_1",
        "first_x0_1",
        "first_x_2",
        "first_x0_2",
    ]


def test_a_model_that_cannot_leave_its_latent_space_is_refused() -> None:
    targets, latent_preview_module, _ = _targets()

    with (
        pytest.raises(RuntimeError, match="no process_latent_out"),
        armed_step_capture(targets=targets),
    ):
        latent_preview_module.prepare_callback(object(), TOTAL_STEPS)


def test_a_stashed_latent_is_detached_onto_the_host() -> None:
    targets, latent_preview_module, _ = _targets()

    with armed_step_capture(targets=targets) as capture:
        _sample(latent_preview_module, steps=1, pass_name="only")

    stashed = capture.single_pass().latents_for(LatentArc.CANVAS)[0]
    assert stashed.was_detached
    assert stashed.was_copied_to_host


def test_the_original_callback_still_runs() -> None:
    targets, latent_preview_module, _ = _targets()

    with armed_step_capture(targets=targets):
        _sample(latent_preview_module, steps=TOTAL_STEPS, pass_name="first")

    assert latent_preview_module.delegated_steps == [0, 1, 2]


def test_both_seams_are_put_back_even_when_the_render_raises() -> None:
    targets, latent_preview_module, original_decode = _targets()
    original_prepare_callback = latent_preview_module.prepare_callback

    with pytest.raises(RuntimeError, match="the render fell over"), armed_step_capture(targets=targets):
        raise RuntimeError("the render fell over")

    assert latent_preview_module.prepare_callback == original_prepare_callback
    assert FakeDecodeNode.decode == original_decode


def test_two_sampler_passes_are_kept_apart_and_refused() -> None:
    targets, latent_preview_module, _ = _targets()

    with armed_step_capture(targets=targets) as capture:
        _sample(latent_preview_module, steps=2, pass_name="first")
        _sample(latent_preview_module, steps=TOTAL_STEPS, pass_name="second")

    assert [sampler_pass.step_count for sampler_pass in capture.passes] == [2, TOTAL_STEPS]
    with pytest.raises(RuntimeError, match="sampled 2 time"):
        capture.single_pass()


def test_a_render_that_never_decoded_is_refused() -> None:
    targets, latent_preview_module, _ = _targets()

    with armed_step_capture(targets=targets) as capture:
        _sample(latent_preview_module, steps=1, pass_name="only")

    with pytest.raises(RuntimeError, match="never passed a VAE"):
        capture.require_vae()


def test_decode_turns_latents_into_frames_with_the_renders_own_vae() -> None:
    torch = pytest.importorskip("torch", reason="the decode path converts a tensor the VAE returned")
    pixels = torch.zeros((1, 2, 3, 3), dtype=torch.float32)
    pixels[0, 1, 2, 0] = 1.0
    vae = FakeVae(pixels)

    frames = decode_latent_frames(vae=vae, latents=["first latent", "second latent"])

    assert vae.decoded_latents == ["first latent", "second latent"]
    assert [frame.size for frame in frames] == [(3, 2), (3, 2)]
    assert frames[0].getpixel((2, 1)) == (255, 0, 0)
    assert frames[0].getpixel((0, 0)) == (0, 0, 0)


def test_the_two_arcs_are_two_views_of_one_render() -> None:
    spec = load_spec(SPEC_PATH)
    strips = step_preview_strips(spec=spec)

    assert {arc: strip.parameter for arc, strip in strips.items()} == ARC_PARAMETERS
    assert strips[LatentArc.CANVAS].parameter is SweepParameter.STEP_PREVIEW

    render_job = check_arcs_agree(strips)
    assert render_job.steps == len(strips[LatentArc.CANVAS].jobs)
    assert [job.stop_value for job in strips[LatentArc.GUESS].jobs] == list(range(1, render_job.steps + 1))


def test_a_campaign_never_renders_the_captured_stops_itself() -> None:
    spec = load_spec(SPEC_PATH)

    selected = select_jobs(spec=spec, job_filter=JobFilter(parameters=(str(SweepParameter.STEP_PREVIEW),)))

    assert selected == []
