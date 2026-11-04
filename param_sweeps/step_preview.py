# SPDX-FileCopyrightText: 2026 Tazlin
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Capture the in-progress latents of one render and publish them as two strips.

Every other strip in this library is N separate renders. This one is not. A strip that shows a
render part-way through has to come from a single render, because thirty renders stopped at
thirty step counts are thirty different pictures, not thirty moments of one.

The capture wraps two seams that the sampling graph already calls. ComfyUI's sampler calls
``callback(step, x0, x, total_steps)`` once per step, and builds that callback by looking up
``prepare_callback`` on the ``latent_preview`` module at call time. Wrapping that attribute puts a
recorder in front of the callback that the graph would have used anyway, and the original is
still called, so nothing that the pipeline depends on is removed. The decode seam is wrapped only
to capture the VAE. The graph passes its VAE to ``VAEDecode.decode``, and that VAE decodes the
stored latents into pictures whose last frame is the delivered image, not an approximation of it.
``comfy.utils.PROGRESS_BAR_HOOK`` is deliberately untouched, because horde-engine owns that hook
and installs it once.

Two arcs come out of one render, because the sampler holds two different values at each step:

- ``x``, the canvas: what is on the canvas, including the noise that the remaining steps remove.
- ``x0``, the guess: the model's current estimate of the finished picture.

Critical public members:

- [`armed_step_capture`][param_sweeps.step_preview.armed_step_capture]: the context manager that
  arms both seams and restores them afterwards.
- [`StepPreviewCapture`][param_sweeps.step_preview.StepPreviewCapture]: the latents and VAE that a
  capture stored.
- [`LatentArc`][param_sweeps.step_preview.LatentArc]: the two arcs, and the parameter each
  publishes under.
- [`decode_latent_frames`][param_sweeps.step_preview.decode_latent_frames]: turn stashed latents
  into images with the render's own VAE.
- [`run_step_preview`][param_sweeps.step_preview.run_step_preview]: render once, decode both arcs
  and write them into the library.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image
from strenum import StrEnum

from param_sweeps.checkpoint import CampaignCheckpoint
from param_sweeps.outputs import save_render, write_index
from param_sweeps.runner import SweepRunner
from param_sweeps.spec import RenderJob, RenderStrip, SweepParameter, SweepSpec

logger = logging.getLogger(__name__)

PREPARE_CALLBACK_ATTRIBUTE = "prepare_callback"
"""The module attribute that the sampling node uses to look up its per-step callback."""

DECODE_ATTRIBUTE = "decode"
"""The method on the decode node that receives the graph's VAE."""


class LatentArc(StrEnum):
    """The two values that a sampler holds at every step, each published as its own strip."""

    CANVAS = "canvas"
    """``x``: what is on the canvas at this step, noise included."""

    GUESS = "guess"
    """``x0``: the model's guess at the finished picture, at this step."""


ARC_PARAMETERS: dict[LatentArc, SweepParameter] = {
    LatentArc.CANVAS: SweepParameter.STEP_PREVIEW,
    LatentArc.GUESS: SweepParameter.STEP_PREVIEW_GUESS,
}
"""Which sweep parameter each arc is published under, and so which directory it is written to."""


class SamplerPass:
    """The latents stored from one run of one sampler, in step order.

    There is one pass per ``prepare_callback`` call, not one buffer for the whole render.
    Otherwise a pipeline that samples twice (as the hires-fix path does) would interleave two arcs
    into one list, and a strip built from it would show the wrong content in valid files.
    """

    def __init__(self, *, requested_steps: int, process_latent_out: Callable[[Any], Any]) -> None:
        """Open a pass for this many steps.

        Args:
            requested_steps: The step count that the callback is built for.
            process_latent_out: The model's conversion from its sampling space to the space that
                a VAE decodes from.
        """
        self.requested_steps = requested_steps
        self.process_latent_out = process_latent_out
        self.canvas_latents: list[Any] = []
        self.guess_latents: list[Any] = []
        self.total_steps: int | None = None

    def record(self, *, canvas_latent: Any, guess_latent: Any, total_steps: int) -> None:
        """Store one step's pair of latents, detached onto the host and in the decoder's space.

        The tensors are copied off the accelerator immediately. Keeping the sampler's tensors
        would keep the whole arc in video memory for the whole render. The sampler also reuses
        its buffers, so a reference would not preserve the values.

        A sampler works in a scaled latent space, and the model's ``process_latent_out`` converts
        a result back out of it. The sampling path applies it to the finished result on the way
        to the decode node. A latent taken mid-render has not been converted, and a VAE given an
        unconverted latent returns flat grey instead of a picture, so each stored latent goes
        through the same conversion as the delivered one.
        """
        self.total_steps = total_steps
        self.canvas_latents.append(self.process_latent_out(_stashed(canvas_latent)))
        self.guess_latents.append(self.process_latent_out(_stashed(guess_latent)))

    def latents_for(self, arc: LatentArc) -> list[Any]:
        """Return the arc's latents, in step order."""
        return self.canvas_latents if arc is LatentArc.CANVAS else self.guess_latents

    @property
    def step_count(self) -> int:
        """Return how many steps this pass recorded."""
        return len(self.canvas_latents)


class StepPreviewCapture:
    """The data stored by one armed window: every sampler pass, and the VAE that the graph decoded with.

    Thread Safety:
        Not thread-safe, by design. The window is armed around a single render.
    """

    def __init__(self) -> None:
        """Open an empty capture."""
        self.passes: list[SamplerPass] = []
        self.vae: Any | None = None

    def begin_pass(self, *, requested_steps: int, process_latent_out: Callable[[Any], Any]) -> SamplerPass:
        """Open and return a pass for one sampler run."""
        sampler_pass = SamplerPass(requested_steps=requested_steps, process_latent_out=process_latent_out)
        self.passes.append(sampler_pass)
        return sampler_pass

    def record_vae(self, vae: Any) -> None:
        """Store the VAE that the graph decoded its result with."""
        self.vae = vae

    def single_pass(self) -> SamplerPass:
        """Return the only sampler pass that this capture recorded.

        Raises:
            RuntimeError: If the render did not sample exactly once, which means the pipeline was
                not the single-pass pipeline that a step-preview strip can be built from.
        """
        if len(self.passes) != 1:
            raise RuntimeError(
                f"a step-preview strip needs a render that samples exactly once; this one sampled"
                f" {len(self.passes)} time(s)",
            )
        return self.passes[0]

    def require_vae(self) -> Any:
        """Return the captured VAE.

        Raises:
            RuntimeError: If the graph did not reach the decode node that the capture wraps. The
                stored latents then cannot be decoded in the render's space.
        """
        if self.vae is None:
            raise RuntimeError(
                "the render never passed a VAE through the decode node, so the stashed latents"
                " cannot be decoded the way the delivered image was",
            )
        return self.vae


class CaptureTargets:
    """The two objects that an armed window patches, held together so a test can supply its own."""

    def __init__(self, *, latent_preview_module: Any, decode_node_class: Any) -> None:
        """Store the objects to patch.

        Args:
            latent_preview_module: The module that has ``prepare_callback``.
            decode_node_class: The node class that has ``decode``.
        """
        self.latent_preview_module = latent_preview_module
        self.decode_node_class = decode_node_class


def comfyui_capture_targets() -> CaptureTargets:
    """Return the seams inside the vendored ComfyUI, imported at call time.

    Both imports only resolve once horde-engine has initialised, because initialisation puts the
    vendored ComfyUI on the import path. Calling this before
    [`prepare`][param_sweeps.local_runner.LocalRunner.prepare] is therefore an error.

    Raises:
        RuntimeError: If the modules are not importable, which means the GPU backend was never
            initialised in this process.
    """
    try:
        import latent_preview
        import nodes
    except ImportError as missing_comfyui:
        raise RuntimeError(
            "the vendored ComfyUI is not importable; the local runner has to be prepared before a"
            " step preview can be captured",
        ) from missing_comfyui

    return CaptureTargets(latent_preview_module=latent_preview, decode_node_class=nodes.VAEDecode)


@contextmanager
def armed_step_capture(*, targets: CaptureTargets) -> Iterator[StepPreviewCapture]:
    """Context manager that records every sampler step of the renders run inside it.

    Both patches are removed when the block exits, however it ends, because they are
    process-wide. A wrapper left armed would keep storing latents for every later render in the
    process until the process exited.

    Args:
        targets: The module and node class to patch.

    Yields:
        The capture, which fills as the render inside the block runs.

    Side Effects:
        Replaces two attributes on objects that this tool does not own, for the length of the block.
    """
    capture = StepPreviewCapture()
    original_prepare_callback = getattr(targets.latent_preview_module, PREPARE_CALLBACK_ATTRIBUTE)
    original_decode = getattr(targets.decode_node_class, DECODE_ATTRIBUTE)

    setattr(
        targets.latent_preview_module,
        PREPARE_CALLBACK_ATTRIBUTE,
        _recording_prepare_callback(capture=capture, original=original_prepare_callback),
    )
    setattr(
        targets.decode_node_class,
        DECODE_ATTRIBUTE,
        _vae_remembering_decode(capture=capture, original=original_decode),
    )
    try:
        yield capture
    finally:
        setattr(targets.latent_preview_module, PREPARE_CALLBACK_ATTRIBUTE, original_prepare_callback)
        setattr(targets.decode_node_class, DECODE_ATTRIBUTE, original_decode)


def _recording_prepare_callback(*, capture: StepPreviewCapture, original: Any) -> Callable[..., Any]:
    """Return a ``prepare_callback`` that stores each step before calling the original callback."""

    def prepare_callback(model: Any, steps: int, *args: Any, **kwargs: Any) -> Callable[..., Any]:
        delegate = original(model, steps, *args, **kwargs)
        sampler_pass = capture.begin_pass(
            requested_steps=steps,
            process_latent_out=_process_latent_out_of(model),
        )
        logger.info("Step capture armed for a sampler pass: requested_steps=%s", steps)

        def callback(step: int, x0: Any, x: Any, total_steps: int) -> Any:
            sampler_pass.record(canvas_latent=x, guess_latent=x0, total_steps=total_steps)
            return delegate(step, x0, x, total_steps)

        return callback

    return prepare_callback


def _vae_remembering_decode(*, capture: StepPreviewCapture, original: Any) -> Callable[..., Any]:
    """Return a ``decode`` that stores the VAE that it receives, then decodes as usual."""

    def decode(node: Any, *args: Any, **kwargs: Any) -> Any:
        vae = kwargs.get("vae", args[0] if args else None)
        if vae is not None:
            capture.record_vae(vae)
        return original(node, *args, **kwargs)

    return decode


def _process_latent_out_of(model: Any) -> Callable[[Any], Any]:
    """Return the model's conversion out of the space that its sampler works in.

    Taken from the patcher that the callback is built for, which is the same object and method
    that the sampling path applies to the finished result.

    Raises:
        RuntimeError: If the object passed to ``prepare_callback`` does not have this conversion.
            Every stored latent would then be in a space that a VAE cannot decode.
    """
    process_latent_out = getattr(getattr(model, "model", None), "process_latent_out", None)
    if process_latent_out is None:
        raise RuntimeError(
            "the sampling node was given a model with no process_latent_out, so a latent taken"
            " mid-render cannot be put into the space the decoder reads",
        )

    resolved: Callable[[Any], Any] = process_latent_out
    return resolved


def _stashed(latent: Any) -> Any:
    """Return a latent copied onto the host as float32, detached from the sampler's own buffer.

    A nested tensor is unwrapped to its first tensor, as the wrapped preview path does with the
    same value.
    """
    if getattr(latent, "is_nested", False):
        latent = latent.tensors[0]
    return latent.detach().float().cpu().clone()


def decode_latent_frames(*, vae: Any, latents: Sequence[Any]) -> list[Image.Image]:
    """Return one image per latent, decoded with the VAE that the render used.

    Latents are decoded one at a time instead of as a batch. A thirty-frame batch of SDXL latents
    is a large allocation for no gain, and a failure then loses one frame instead of the arc.

    The decode runs under inference mode, as the render's own decode did. The stored latents were
    taken inside that mode and do not have a version counter, so a VAE that writes its output in
    place (as the decode does) rejects them outside it.

    Args:
        vae: The VAE captured from the render's own graph.
        latents: The stashed latents, in step order.

    Returns:
        The decoded frames, in the same order.

    Side Effects:
        Runs the VAE, which loads it onto the accelerator if it is not already there.
    """
    import torch

    frames: list[Image.Image] = []
    with torch.inference_mode():
        for step_number, latent in enumerate(latents, start=1):
            logger.info("Decoding step frame: step=%s of %s", step_number, len(latents))
            frames.append(_as_image(vae.decode(latent)))

    return frames


def _as_image(decoded: Any) -> Image.Image:
    """Return one decoded VAE output as an image.

    The decode returns pixels as height, width, channel floats in 0..1 with any leading batch and
    frame axes still on the front, so the leading axes are dropped before the conversion. The
    channel axis is moved into place rather than copied, which leaves a strided view that an image
    cannot be built from, so the bytes are made contiguous before the conversion.
    """
    frame = decoded
    while frame.ndim > 3:
        frame = frame[0]

    pixels = frame.clamp(0.0, 1.0).mul(255.0).round().byte().cpu().contiguous()
    return Image.fromarray(pixels.numpy(), mode="RGB")


def step_preview_strips(*, spec: SweepSpec) -> dict[LatentArc, RenderStrip]:
    """Return the strip for each arc, read from the specification.

    Raises:
        ValueError: If the specification does not describe exactly one strip per arc. One strip
            per arc lets one render cover both.
    """
    strips: dict[LatentArc, RenderStrip] = {}
    for arc, parameter in ARC_PARAMETERS.items():
        matching = [strip for strip in spec.expand_strips() if strip.parameter is parameter]
        if len(matching) != 1:
            raise ValueError(f"the specification describes {len(matching)} {parameter} strips; it has to describe one")
        strips[arc] = matching[0]

    return strips


def check_arcs_agree(strips: dict[LatentArc, RenderStrip]) -> RenderJob:
    """Return the job whose render covers both arcs, after checking that it does.

    The two strips are two views of a single render, so every stop of both has to state the same
    parameters. A specification in which they differ would publish arcs of two different pictures
    and label them as one.

    Args:
        strips: The strip for each arc.

    Returns:
        The first stop's job, whose render provides every stop of both strips.

    Raises:
        ValueError: If a stop states parameters that differ from the first stop's, or if the two
            strips do not have the same number of stops.
    """
    canvas_jobs = strips[LatentArc.CANVAS].jobs
    guess_jobs = strips[LatentArc.GUESS].jobs
    if len(canvas_jobs) != len(guess_jobs):
        raise ValueError(
            f"the two step-preview strips have {len(canvas_jobs)} and {len(guess_jobs)} stops;"
            " they are two views of one render and must have the same number",
        )

    render_job = canvas_jobs[0]
    reference = render_job.generation_parameters()
    for job in [*canvas_jobs, *guess_jobs]:
        differing = reference.differing_field_names(job.generation_parameters())
        if differing:
            raise ValueError(
                f"step-preview stop {job.relative_image_path} states its own {sorted(differing)};"
                " every stop is a moment of one render and cannot state parameters of its own",
            )

    return render_job


def run_step_preview(
    *,
    runner: SweepRunner,
    spec: SweepSpec,
    output_root: Path,
    targets: CaptureTargets | None = None,
) -> dict[LatentArc, list[Path]]:
    """Render once with the capture armed, then write both arcs into the library.

    The one render is the canvas strip's first stop.
    [`check_arcs_agree`][param_sweeps.step_preview.check_arcs_agree] has already checked that
    every other stop of both strips matches it. Each frame is then written through the
    ordinary save path against its own stop's job, so a step-preview image has the same sidecar,
    filename and checkpoint entry as an ordinary stop.

    Args:
        runner: The backend to render through, unprepared.
        spec: The specification containing both step-preview strips.
        output_root: The root of the output tree.
        targets: The seams to patch, or None to resolve them from the prepared backend.

    Returns:
        The files written, per arc, in step order.

    Raises:
        RuntimeError: If the render did not produce an image, did not sample exactly once, or
            never reached a decode node.
        ValueError: If the specification's two strips do not match.

    Side Effects:
        Renders on the accelerator, writes images, sidecars, the checkpoint and the index.
    """
    strips = step_preview_strips(spec=spec)
    render_job = check_arcs_agree(strips)

    started_at = datetime.now(UTC)
    try:
        runner.prepare([render_job], [strips[LatentArc.CANVAS].image_model])
        resolved_targets = targets if targets is not None else comfyui_capture_targets()

        logger.info(
            "Rendering the step-preview job: steps=%s job=%s",
            render_job.steps,
            render_job.relative_image_path,
        )
        with armed_step_capture(targets=resolved_targets) as capture:
            outcome = runner.render(render_job)

        finished_at = datetime.now(UTC)
        sampler_pass = capture.single_pass()
        vae = capture.require_vae()
        logger.info(
            "Captured a sampler pass: steps_recorded=%s total_steps=%s delivered_images=%s",
            sampler_pass.step_count,
            sampler_pass.total_steps,
            len(outcome.images),
        )

        written: dict[LatentArc, list[Path]] = {}
        for arc, strip in strips.items():
            written[arc] = _write_arc(
                arc=arc,
                strip=strip,
                frames=decode_latent_frames(vae=vae, latents=sampler_pass.latents_for(arc)),
                output_root=output_root,
                runner_name=runner.name,
                started_at=started_at,
                finished_at=finished_at,
                model_name_used=outcome.model_name_used,
                note=outcome.note,
            )
    finally:
        runner.close()
        write_index(spec=spec, output_root=output_root)

    return written


def _write_arc(
    *,
    arc: LatentArc,
    strip: RenderStrip,
    frames: Sequence[Image.Image],
    output_root: Path,
    runner_name: str,
    started_at: datetime,
    finished_at: datetime,
    model_name_used: str | None,
    note: str | None,
) -> list[Path]:
    """Write one arc's frames against its strip's stops, and return their paths.

    Raises:
        RuntimeError: If the number of captured frames differs from the strip's stop count.
            Publishing them would label frames with the wrong step numbers.
    """
    if len(frames) != len(strip.jobs):
        raise RuntimeError(
            f"the {arc} arc captured {len(frames)} frame(s) for a strip declaring {len(strip.jobs)} stop(s)",
        )

    checkpoint = CampaignCheckpoint(output_root=output_root)
    written: list[Path] = []
    for job, frame in zip(strip.jobs, frames, strict=True):
        written.extend(
            save_render(
                images=[frame],
                job=job,
                output_root=output_root,
                runner_name=runner_name,
                started_at=started_at,
                finished_at=finished_at,
                model_name_used=model_name_used,
                worker_note=note,
            ),
        )
        checkpoint.mark_complete(
            relative_image_path=job.checkpoint_key,
            runner_name=runner_name,
            seed=job.seed,
        )

    logger.info("Wrote the %s arc: frames=%s directory=%s", arc, len(written), strip.jobs[0].relative_directory)
    return written
