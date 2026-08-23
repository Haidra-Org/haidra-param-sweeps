# SPDX-FileCopyrightText: 2026 Tazlin
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Render sweep jobs on this machine's GPU through horde-engine (``hordelib``).

This is the primary runner: it is the same inference stack an AI Horde worker runs, so what it
produces is what the horde produces. The GPU backend is always stated rather than guessed, and
model weights come from the standard ``AIWORKER_CACHE_HOME`` location, so an existing worker
cache is reused instead of re-downloaded.

``hordelib`` is imported inside the methods that need it. Importing it starts a ComfyUI
environment check and pulls in torch, which must not happen merely because something wanted to
read the specification or rebuild the index.

Critical public members:

- [`LocalRunner`][param_sweeps.local_runner.LocalRunner]: the runner itself.
- [`GpuBackend`][param_sweeps.local_runner.GpuBackend]: the compute backends that can be selected.
- [`resolve_gpu_backend`][param_sweeps.local_runner.resolve_gpu_backend]: the flag/environment
  resolution, with CUDA as the documented default for this machine.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, override

from PIL import Image
from pydantic import BaseModel, ConfigDict
from strenum import StrEnum

from param_sweeps.runner import FetchPlanEntry, RenderOutcome, SweepRunner
from param_sweeps.spec import ModelSpec, RenderJob

logger = logging.getLogger(__name__)

GPU_BACKEND_ENVIRONMENT_VARIABLE = "PARAM_SWEEPS_GPU_BACKEND"
"""Environment variable naming the compute backend, when the flag is not given."""

COMPVIS_MODEL_MANAGER = "compvis"
MISCELLANEOUS_MODEL_MANAGER = "miscellaneous"
"""The only model manager these sweeps need: plain checkpoint image models."""

POST_PROCESSOR_MODEL_MANAGERS = ("gfpgan", "codeformer", "esrgan")
"""The managers a post-processor is resolved against: face fixers and upscalers.

Post-processing looks its model up across these three and fails outright when none of them is
loaded, so they are loaded up front. They carry only their references until a job actually names
one of their models, at which point the weights (tens of megabytes) are fetched on demand.
"""

IMG2IMG_SOURCE_PROCESSING = "img2img"
"""What ``hordelib`` calls starting from a source image rather than from noise."""


class GpuBackend(StrEnum):
    """The compute backends a local run can be pointed at.

    These mirror ``hordelib.utils.torch_memory.AcceleratorKind``. They are restated here so that
    the command line can be parsed, and an unsupported choice rejected, without importing torch.
    """

    CUDA = "cuda"
    ROCM = "rocm"
    XPU = "xpu"
    NPU = "npu"
    MLU = "mlu"
    MPS = "mps"
    DIRECTML = "directml"
    CPU = "cpu"


DEFAULT_GPU_BACKEND = GpuBackend.CUDA
"""CUDA is the documented default; every other backend has to be asked for."""


class HordelibPayload(BaseModel):
    """Represents one job in the payload shape ``hordelib`` accepts.

    ``hordelib`` takes a plain Horde-style payload mapping, so this model exists to build that
    mapping in one checked place rather than assembling a dictionary at the call site. Note the
    field names that differ from the horde API's: steps is ``ddim_steps``, the batch size is
    ``n_iter``, and the negative prompt is its own field rather than the API's ``###`` suffix.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, protected_namespaces=())

    prompt: str
    model: str
    sampler_name: str
    cfg_scale: float
    ddim_steps: int
    denoising_strength: float
    seed: int
    width: int
    height: int
    karras: bool
    tiling: bool
    hires_fix: bool
    clip_skip: int
    hires_fix_denoising_strength: float | None = None
    scheduler: str | None = None
    sampler_eta: float | None = None
    sampler_s_noise: float | None = None
    sampler_s_churn: float | None = None
    sampler_solver_type: str | None = None
    sampler_order: int | None = None
    flow_shift: float | None = None
    negative_prompt: str = ""
    source_image: Image.Image | None = None
    source_processing: str | None = None
    source_mask: Image.Image | None = None
    extra_source_images: list[dict[str, Any]] | None = None
    n_iter: int = 1
    control_type: str | None = None
    control_strength: float | None = None
    image_is_control: bool = False
    return_control_map: bool = False
    workflow: str | None = None
    extra_texts: list[dict[str, str]] | None = None
    transparent: bool = False
    loras: list[dict[str, Any]] | None = None
    tis: list[dict[str, Any]] | None = None
    post_processing: list[str] | None = None
    facefixer_strength: float | None = None

    @classmethod
    def from_job(
        cls,
        job: RenderJob,
        *,
        source_image: Image.Image | None = None,
        source_mask: Image.Image | None = None,
        extra_source_images: Sequence[tuple[Image.Image, float]] = (),
    ) -> HordelibPayload:
        """Create a payload from a render job, starting from a source image when it has one.

        Args:
            job: The job to render.
            source_image: The already-loaded image the job starts from, when it declares one.
            source_mask: The already-loaded mask, when the job declares one.
            extra_source_images: The already-loaded additional inputs and their strengths, in the
                order the job declares them.
        """
        return cls(
            prompt=job.prompt,
            model=job.horde_model_name,
            sampler_name=job.sampler_name,
            cfg_scale=job.cfg_scale,
            ddim_steps=job.steps,
            denoising_strength=job.denoising_strength,
            seed=job.seed,
            width=job.width,
            height=job.height,
            karras=job.karras,
            tiling=job.tiling,
            hires_fix=job.hires_fix,
            clip_skip=job.clip_skip,
            hires_fix_denoising_strength=job.hires_fix_denoising_strength,
            # A named schedule is more specific than the karras flag and wins over it in hordelib;
            # an unset one leaves the flag to decide, which is what every earlier sweep relied on.
            scheduler=str(job.scheduler) if job.scheduler is not None else None,
            sampler_eta=job.sampler_eta,
            sampler_s_noise=job.sampler_s_noise,
            sampler_s_churn=job.sampler_s_churn,
            sampler_solver_type=job.sampler_solver_type,
            sampler_order=job.sampler_order,
            flow_shift=job.flow_shift,
            negative_prompt=job.negative_prompt,
            source_image=source_image,
            source_processing=_source_processing_for(job, source_image=source_image),
            source_mask=source_mask,
            extra_source_images=[
                {"image": extra_image, "strength": strength} for extra_image, strength in extra_source_images
            ]
            or None,
            n_iter=job.batch_size,
            control_type=job.control_type,
            control_strength=job.control_strength,
            image_is_control=job.image_is_control,
            return_control_map=job.return_control_map,
            workflow=str(job.workflow) if job.workflow is not None else None,
            extra_texts=[extra_text.model_dump() for extra_text in job.extra_texts] or None,
            transparent=job.transparent,
            loras=[lora.model_dump() for lora in job.loras] or None,
            tis=[textual_inversion.model_dump() for textual_inversion in job.tis] or None,
            post_processing=list(job.post_processing) or None,
            facefixer_strength=job.facefixer_strength,
        )

    def as_hordelib_mapping(self) -> dict[str, Any]:
        """Return the payload in the mapping form ``hordelib`` expects.

        Keys whose value is None are dropped rather than passed through: ``hordelib`` reads the
        presence of a key as the feature being asked for in several places, so a payload that
        states every feature as absent is not the same thing as one that never mentions them.
        """
        return {key: value for key, value in self.model_dump().items() if value is not None}


def _source_processing_for(job: RenderJob, *, source_image: Image.Image | None) -> str | None:
    """Return what the backend should do with the source image, or None when there is none."""
    if source_image is None:
        return None

    resolved = job.resolved_source_processing
    return str(resolved) if resolved is not None else IMG2IMG_SOURCE_PROCESSING


def resolve_gpu_backend(*, requested: str | None) -> GpuBackend:
    """Return the compute backend to run on, from the flag, then the environment, then the default.

    Args:
        requested: The value given on the command line, or None.

    Returns:
        The selected backend.

    Raises:
        ValueError: If a backend was named that this tool does not support.
    """
    candidate = requested if requested is not None else os.environ.get(GPU_BACKEND_ENVIRONMENT_VARIABLE)
    if candidate is None or not candidate.strip():
        return DEFAULT_GPU_BACKEND

    try:
        return GpuBackend(candidate.strip().lower())
    except ValueError as lookup_error:
        supported = ", ".join(str(backend) for backend in GpuBackend)
        raise ValueError(f"unsupported GPU backend {candidate!r}; supported backends: {supported}") from lookup_error


class LocalRunner(SweepRunner):
    """Renders sweep jobs through horde-engine on a local accelerator.

    The backend and device are applied by ``hordelib.initialise`` before torch is imported, which
    is the only point at which a device mask can still take effect, so both are settled in
    [`prepare`][param_sweeps.local_runner.LocalRunner.prepare] and never changed afterwards.

    Examples:
        ```python
        runner = LocalRunner(gpu_backend=GpuBackend.CUDA, device_index=0)
        run_campaign(spec=spec, runner=runner, output_root=output_root)
        ```
    """

    def __init__(self, *, gpu_backend: GpuBackend, device_index: int, download_missing_models: bool = True) -> None:
        """Configure a local runner.

        Args:
            gpu_backend: The compute backend to pin the process to.
            device_index: The accelerator index to pin the process to.
            download_missing_models: Whether preparation may download missing model weights into
                the worker cache. False turns a missing model into an error instead.
        """
        self._gpu_backend = gpu_backend
        self._device_index = device_index
        self._download_missing_models = download_missing_models
        self._horde: Any | None = None
        self._model_managers_loaded = False
        self._source_images: dict[Path, Image.Image] = {}

    @property
    @override
    def name(self) -> str:
        return f"local:{self._gpu_backend}"

    @override
    def fetch_plan(self, jobs: Sequence[RenderJob], models: Sequence[ModelSpec]) -> list[FetchPlanEntry]:
        image_counts = _image_counts_by_model(jobs)
        return [
            FetchPlanEntry(
                model_name=image_model.horde_name,
                approximate_size_gb=image_model.approximate_download_size_gb,
                already_present=self._is_model_on_disk(image_model.horde_name),
                image_count=image_counts.get(image_model.slug, 0),
            )
            for image_model in models
        ]

    @override
    def prepare(self, jobs: Sequence[RenderJob], models: Sequence[ModelSpec]) -> None:
        import hordelib
        from hordelib.horde import HordeLib
        from hordelib.shared_model_manager import SharedModelManager
        from hordelib.utils.torch_memory import AcceleratorKind

        logger.info("Initialising hordelib: backend=%s device_index=%s", self._gpu_backend, self._device_index)
        hordelib.initialise(
            setup_logging=True,
            logging_verbosity=3,
            accelerator_kind=AcceleratorKind(str(self._gpu_backend)),
            device_index=self._device_index,
        )

        # The miscellaneous manager owns the layer-diffusion weights the
        # `transparent` pipeline loads; without it the LayeredDiffusionApply
        # node resolves no model folder and every transparent job dies.
        SharedModelManager.load_model_managers(
            [COMPVIS_MODEL_MANAGER, MISCELLANEOUS_MODEL_MANAGER, *POST_PROCESSOR_MODEL_MANAGERS],
        )
        self._model_managers_loaded = True
        for image_model in models:
            self._ensure_model_present(image_model)

        self._horde = HordeLib()

    @override
    def render(self, job: RenderJob) -> RenderOutcome:
        if self._horde is None:
            raise RuntimeError("prepare() must be called before render()")

        payload = HordelibPayload.from_job(
            job,
            source_image=self._source_image_for(job),
            source_mask=self._source_mask_for(job),
            extra_source_images=self._extra_source_images_for(job),
        )
        mapping = payload.as_hordelib_mapping()

        if job.batch_size > 1:
            return self._render_candidates(job, mapping)

        if mapping.get("post_processing"):
            return self._render_post_processed(job, mapping)

        result = self._horde.basic_inference_single_image(mapping)
        rendered_image: Image.Image | None = result.image
        if rendered_image is None:
            raise RuntimeError(f"hordelib returned no image for {job.relative_image_path}")

        fault_note = "; ".join(str(fault) for fault in result.faults) if result.faults else None
        return RenderOutcome(images=[rendered_image], model_name_used=job.horde_model_name, note=fault_note)

    def _render_post_processed(self, job: RenderJob, mapping: dict[str, Any]) -> RenderOutcome:
        """Render one image through the entry point that also applies the requested post-processors.

        The single-image entry point runs the sampler and nothing else: only the multi-image call
        reads ``post_processing``. A face fixer or upscaler asked for through the single-image call
        is dropped without a word, and the render comes back byte-identical to one that never asked
        for it, which is exactly the silence a post-processing comparison cannot afford.

        Raises:
            RuntimeError: If the backend did not return exactly one image.
        """
        if self._horde is None:
            raise RuntimeError("prepare() must be called before render()")

        results = self._horde.basic_inference(mapping)
        rendered_images = [result.image for result in results if result.image is not None]
        if len(rendered_images) != 1:
            raise RuntimeError(
                f"hordelib returned {len(rendered_images)} images for {job.relative_image_path}, expected one",
            )

        faults = [str(fault) for result in results for fault in result.faults]
        fault_note = "; ".join(faults) if faults else None
        return RenderOutcome(images=rendered_images, model_name_used=job.horde_model_name, note=fault_note)

    def _render_candidates(self, job: RenderJob, mapping: dict[str, Any]) -> RenderOutcome:
        """Render a curation batch through the multi-image entry point.

        The single-image entry point is kept for the jobs that ask for one image, because it is
        the call the whole library was rendered with and there is no reason to move it.

        Raises:
            RuntimeError: If the backend returned fewer images than the batch asked for.
        """
        if self._horde is None:
            raise RuntimeError("prepare() must be called before render()")

        if job.workflow is not None:
            return self._render_candidates_one_by_one(job, mapping)

        results = self._horde.basic_inference(mapping)
        rendered_images = [result.image for result in results if result.image is not None]
        if len(rendered_images) != job.batch_size:
            raise RuntimeError(
                f"hordelib returned {len(rendered_images)} of the {job.batch_size} candidates asked for"
                f" for {job.relative_image_path}",
            )

        faults = [str(fault) for result in results for fault in result.faults]
        fault_note = "; ".join(faults) if faults else None
        return RenderOutcome(images=rendered_images, model_name_used=job.horde_model_name, note=fault_note)

    def _render_candidates_one_by_one(self, job: RenderJob, mapping: dict[str, Any]) -> RenderOutcome:
        """Render a curation batch as separate single calls, stepping the seed per candidate.

        Workflow pipelines (the QR overlay among them) generate one image per request whatever
        ``n_iter`` says, the same way the horde itself refuses to batch them. Separate calls with
        the same seed would be the same picture eight times, so each candidate takes the job's
        seed plus its zero-based index, and the sidecars record the seed each candidate really used.
        """
        if self._horde is None:
            raise RuntimeError("prepare() must be called before render()")

        rendered_images: list[Image.Image] = []
        faults: list[str] = []
        for offset in range(job.batch_size):
            candidate_mapping = dict(mapping)
            candidate_mapping["seed"] = int(mapping["seed"]) + offset
            candidate_mapping["n_iter"] = 1
            result = self._horde.basic_inference_single_image(candidate_mapping)
            if result.image is None:
                raise RuntimeError(
                    f"hordelib returned no image for candidate {offset + 1} of {job.relative_image_path}",
                )
            rendered_images.append(result.image)
            faults.extend(str(fault) for fault in result.faults)

        fault_note = "; ".join(faults) if faults else None
        return RenderOutcome(images=rendered_images, model_name_used=job.horde_model_name, note=fault_note)

    @override
    def close(self) -> None:
        self._horde = None
        self._source_images.clear()

    def _source_image_for(self, job: RenderJob) -> Image.Image | None:
        """Return the image this job starts from, decoded once and kept for the strip's other stops.

        Raises:
            RuntimeError: If the job names a source image that is not where it says it is.
        """
        source_path = job.source_image_path
        if source_path is None:
            return None

        return self._decoded(source_path, description=f"the source image for {job.relative_image_path}")

    def _source_mask_for(self, job: RenderJob) -> Image.Image | None:
        """Return this job's mask, decoded once, or None when it has none.

        Raises:
            RuntimeError: If the job names a mask that is not where it says it is.
        """
        mask_path = job.source_mask_path
        if mask_path is None:
            return None

        return self._decoded(mask_path, description=f"the mask for {job.relative_image_path}")

    def _extra_source_images_for(self, job: RenderJob) -> list[tuple[Image.Image, float]]:
        """Return this job's additional inputs and their strengths, in the order it declares them.

        Raises:
            RuntimeError: If the job names an input that is not where it says it is.
        """
        return [
            (
                self._decoded(
                    extra_source_image.absolute_path,
                    description=f"an extra source image for {job.relative_image_path}",
                ),
                extra_source_image.strength,
            )
            for extra_source_image in job.extra_source_images
        ]

    def _decoded(self, image_path: Path, *, description: str) -> Image.Image:
        """Return an input image, decoded once and kept for every job that names the same file.

        One decode per path is safe because a staged input is written once per run: the job ids
        that name one are unique across the specification.

        Raises:
            RuntimeError: If the file is not where the specification says it is.
        """
        cached = self._source_images.get(image_path)
        if cached is not None:
            return cached

        if not image_path.is_file():
            raise RuntimeError(f"{description} is missing: {image_path}")

        with Image.open(image_path) as opened:
            loaded = opened.convert("RGB")

        self._source_images[image_path] = loaded
        return loaded

    def _ensure_model_present(self, image_model: ModelSpec) -> None:
        """Download this model's weights if they are not already in the worker cache.

        Raises:
            RuntimeError: If the model is unknown to the reference, or the download failed.
        """
        from hordelib.shared_model_manager import SharedModelManager

        if self._is_model_on_disk(image_model.horde_name):
            logger.info("Model already in the worker cache: model=%s", image_model.horde_name)
            return

        if not self._download_missing_models:
            raise RuntimeError(f"model {image_model.horde_name} is not in the cache and downloading is disabled")

        logger.info(
            "Downloading model: model=%s approximate_size_gb=%.1f",
            image_model.horde_name,
            image_model.approximate_download_size_gb,
        )
        download_succeeded = SharedModelManager.manager.download_model(image_model.horde_name)
        if not download_succeeded:
            raise RuntimeError(f"could not download model {image_model.horde_name}")

    def _is_model_on_disk(self, horde_model_name: str) -> bool | None:
        """Return whether this model's weights are already present, or None before prepare().

        ``SharedModelManager.manager`` only exists once ``load_model_managers`` has run, which
        ``prepare()`` records in ``_model_managers_loaded``; before that (the state a first-run
        fetch plan is printed in) presence is reported as unknown rather than guessed.
        """
        if not self._model_managers_loaded:
            return None

        from hordelib.shared_model_manager import SharedModelManager

        compvis_manager = SharedModelManager.manager.compvis
        if compvis_manager is None:
            return None

        presence: bool = compvis_manager.is_model_available(horde_model_name)
        return presence


def _image_counts_by_model(jobs: Sequence[RenderJob]) -> dict[str, int]:
    """Return how many images each model slug is responsible for, counting batch candidates."""
    counts: dict[str, int] = {}
    for job in jobs:
        counts[job.model_slug] = counts.get(job.model_slug, 0) + job.batch_size
    return counts
