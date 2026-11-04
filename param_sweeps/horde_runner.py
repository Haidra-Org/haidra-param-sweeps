# SPDX-FileCopyrightText: 2026 Tazlin
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Render sweep jobs through the public AI Horde API.

The API runner exists so the library can be produced without a local GPU, and so a strip can be
re-rendered from anywhere. The queueing, polling, retrying and downloading all live in
[`HordeApiClient`][param_sweeps.horde_client.HordeApiClient], which the tutorial flows also use.
This module only turns a [`RenderJob`][param_sweeps.spec.RenderJob] into a request body, which is
the part that is specific to sweeps.

The API key is read from the environment or a ``.env`` file, and is never
printed, logged, or written to any output.

Critical public members:

- [`HordeApiRunner`][param_sweeps.horde_runner.HordeApiRunner]: the runner itself.
- [`HordeGenerationRequest`][param_sweeps.horde_runner.HordeGenerationRequest]: the request body.
- [`HordeGenerationParams`][param_sweeps.horde_runner.HordeGenerationParams]: its ``params`` object.
"""

from __future__ import annotations

import base64
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, override

import requests
from pydantic import BaseModel, ConfigDict

from param_sweeps.horde_client import (
    API_KEY_VARIABLE,
    CLIENT_AGENT,
    DEFAULT_API_BASE_URL,
    ENV_FILE_VARIABLE,
    HordeApiClient,
    HordePacing,
    MissingApiKeyError,
    default_env_file,
    read_api_key,
)
from param_sweeps.runner import FetchPlanEntry, RenderOutcome, SweepRunner
from param_sweeps.spec import ExtraText, ModelSpec, RenderJob, SourceProcessing

logger = logging.getLogger(__name__)

__all__ = [
    "API_KEY_VARIABLE",
    "CLIENT_AGENT",
    "DEFAULT_API_BASE_URL",
    "ENV_FILE_VARIABLE",
    "HordeApiRunner",
    "HordeGenerationParams",
    "HordeGenerationRequest",
    "HordePacing",
    "MissingApiKeyError",
    "default_env_file",
    "read_api_key",
    "silently_ignored_features_in",
    "unsupported_parameters_in",
]

IMG2IMG_SOURCE_PROCESSING = "img2img"
"""The API's name for starting from a source image instead of from noise."""

RECENTLY_ADDED_PARAMETERS: tuple[str, ...] = (
    "scheduler",
    "sampler_eta",
    "sampler_s_noise",
    "sampler_s_churn",
    "sampler_solver_type",
    "sampler_order",
    "flow_shift",
)
"""Parameters that a deployed API may not accept yet. A job can still request them.

The public API validates its request body strictly and refuses the whole request over a field
that it does not recognize, so a job that sets one of these fails as a job instead of rendering
something different without notice. That is the intended failure, but it should be reported
before a campaign spends its queue time on it.
[`unsupported_parameters_in`][param_sweeps.horde_runner.unsupported_parameters_in] and the
preflight warning report it. Rendering these strips locally does not need server support.
"""


SILENTLY_IGNORED_FEATURES: tuple[str, ...] = ()
"""Features that do not have a field in the public API's request payload, so a request cannot include them.

This is the worse failure of the two. A parameter that the API refuses fails the job and reports
the failure. A feature without a field is dropped by the API, and the render uses the backend's
own defaults. A strip that swept such a feature would publish the same image at every stop, so
the runner reports this before it queues anything, and the strip is rendered locally instead.
The tuple is empty when every feature has a field. ``control_strength`` has a field and is sent
in the request body like any other per-job feature.
"""


def unsupported_parameters_in(job: RenderJob) -> list[str]:
    """Return the recently added parameters that this job sets, which a deployed API may refuse."""
    stated = job.solver_parameters()
    return [name for name in RECENTLY_ADDED_PARAMETERS if stated.get(name) is not None]


def silently_ignored_features_in(job: RenderJob) -> list[str]:
    """Return the features set by this job that the public API's payload cannot include."""
    stated = job.features().model_dump()
    return [name for name in SILENTLY_IGNORED_FEATURES if stated.get(name) is not None]


class HordeExtraSourceImage(BaseModel):
    """Represents one additional input image in a request body, already encoded."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    image: str
    strength: float = 1.0


class HordeGenerationParams(BaseModel):
    """Represents the ``params`` object of an image generation request.

    Every feature field is left unset unless the job turns it on, and the request body is dumped
    without its unset fields, so a job without features sends only the base fields.
    """

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    sampler_name: str
    cfg_scale: float
    width: int
    height: int
    steps: int
    seed: str
    karras: bool
    hires_fix: bool
    tiling: bool
    clip_skip: int
    denoising_strength: float
    n: int = 1
    hires_fix_denoising_strength: float | None = None
    scheduler: str | None = None
    sampler_eta: float | None = None
    sampler_s_noise: float | None = None
    sampler_s_churn: float | None = None
    sampler_solver_type: str | None = None
    sampler_order: int | None = None
    flow_shift: float | None = None
    control_type: str | None = None
    control_strength: float | None = None
    image_is_control: bool | None = None
    return_control_map: bool | None = None
    workflow: str | None = None
    extra_texts: list[ExtraText] | None = None
    transparent: bool | None = None
    loras: list[dict[str, Any]] | None = None
    tis: list[dict[str, Any]] | None = None
    post_processing: list[str] | None = None
    facefixer_strength: float | None = None


class HordeGenerationRequest(BaseModel):
    """Represents the body of a ``POST /generate/async`` request.

    The API takes the negative prompt as a ``###``-separated suffix of the prompt instead of as
    its own field. This is the only place where a job's parameters are not passed through
    unchanged.
    """

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    prompt: str
    params: HordeGenerationParams
    models: list[str]
    nsfw: bool = False
    censor_nsfw: bool = True
    r2: bool = True
    trusted_workers: bool = False
    slow_workers: bool = True
    source_image: str | None = None
    source_processing: str | None = None
    source_mask: str | None = None
    extra_source_images: list[HordeExtraSourceImage] | None = None

    @classmethod
    def from_job(cls, job: RenderJob) -> HordeGenerationRequest:
        """Create a request body from a render job.

        Raises:
            RuntimeError: If an input image named by the job does not exist.
        """
        source_image = _encoded_image(
            job.source_image_path, description=f"the source image for {job.relative_image_path}"
        )

        return cls(
            prompt=f"{job.prompt} ### {job.negative_prompt}" if job.negative_prompt else job.prompt,
            models=[job.horde_model_name],
            params=HordeGenerationParams(
                sampler_name=job.sampler_name,
                cfg_scale=job.cfg_scale,
                width=job.width,
                height=job.height,
                steps=job.steps,
                seed=str(job.seed),
                karras=job.karras,
                hires_fix=job.hires_fix,
                tiling=job.tiling,
                clip_skip=job.clip_skip,
                denoising_strength=job.denoising_strength,
                n=job.batch_size,
                hires_fix_denoising_strength=job.hires_fix_denoising_strength,
                scheduler=str(job.scheduler) if job.scheduler is not None else None,
                sampler_eta=job.sampler_eta,
                sampler_s_noise=job.sampler_s_noise,
                sampler_s_churn=job.sampler_s_churn,
                sampler_solver_type=job.sampler_solver_type,
                sampler_order=job.sampler_order,
                flow_shift=job.flow_shift,
                control_type=job.control_type,
                control_strength=job.control_strength,
                image_is_control=job.image_is_control or None,
                return_control_map=job.return_control_map or None,
                workflow=str(job.workflow) if job.workflow is not None else None,
                extra_texts=list(job.extra_texts) or None,
                transparent=job.transparent or None,
                loras=[lora.model_dump(exclude_none=True) for lora in job.loras] or None,
                tis=[textual_inversion.model_dump(exclude_none=True) for textual_inversion in job.tis] or None,
                post_processing=list(job.post_processing) or None,
                facefixer_strength=job.facefixer_strength,
            ),
            source_image=source_image,
            source_processing=_source_processing_for(job, source_image=source_image),
            source_mask=_encoded_image(job.source_mask_path, description=f"the mask for {job.relative_image_path}"),
            extra_source_images=_encoded_extra_source_images(job),
        )

    def as_request_body(self) -> dict[str, Any]:
        """Return the body to post, without the fields that the job did not set."""
        return self.model_dump(exclude_none=True)


def _source_processing_for(job: RenderJob, *, source_image: str | None) -> str | None:
    """Return what the API should do with the source image, or None when there is none."""
    if source_image is None:
        return None

    resolved: SourceProcessing | None = job.resolved_source_processing
    return str(resolved) if resolved is not None else IMG2IMG_SOURCE_PROCESSING


def _encoded_image(image_path: Path | None, *, description: str) -> str | None:
    """Return an input image as bare base64, or None when there is no image.

    Raises:
        RuntimeError: If the file does not exist.
    """
    if image_path is None:
        return None

    if not image_path.is_file():
        raise RuntimeError(f"{description} is missing: {image_path}")

    return base64.b64encode(image_path.read_bytes()).decode("ascii")


def _encoded_extra_source_images(job: RenderJob) -> list[HordeExtraSourceImage] | None:
    """Return this job's additional inputs, encoded, or None when it does not declare any.

    Raises:
        RuntimeError: If an input file does not exist.
    """
    if not job.extra_source_images:
        return None

    encoded: list[HordeExtraSourceImage] = []
    for extra_source_image in job.extra_source_images:
        image_data = _encoded_image(
            extra_source_image.absolute_path,
            description=f"an extra source image for {job.relative_image_path}",
        )
        if image_data is None:
            raise RuntimeError(f"an extra source image for {job.relative_image_path} could not be read")
        encoded.append(HordeExtraSourceImage(image=image_data, strength=extra_source_image.strength))

    return encoded


class HordeApiRunner(SweepRunner):
    """Renders sweep jobs on the public AI Horde, one job at a time.

    Network Behavior:
        Every request goes through [`HordeApiClient`][param_sweeps.horde_client.HordeApiClient],
        which submits one job at a time, polls until it is done, downloads the images and retries
        a small number of times before the job is reported as failed and left for the next run.
    """

    def __init__(
        self,
        *,
        api_key: str,
        api_base_url: str = DEFAULT_API_BASE_URL,
        session: requests.Session | None = None,
    ) -> None:
        """Configure an API runner.

        Args:
            api_key: The AI Horde API key. Held only in the session header.
            api_base_url: The API root to use.
            session: An existing session to use. The tests use it to substitute a transport.
        """
        self._client = HordeApiClient(api_key=api_key, api_base_url=api_base_url, session=session)

    @property
    @override
    def name(self) -> str:
        return "ai-horde-api"

    @override
    def fetch_plan(self, jobs: Sequence[RenderJob], models: Sequence[ModelSpec]) -> list[FetchPlanEntry]:
        image_counts: dict[str, int] = {}
        for job in jobs:
            image_counts[job.model_slug] = image_counts.get(job.model_slug, 0) + job.batch_size

        # Nothing is downloaded locally. Whether a worker currently serves a model depends on the
        # horde's live state, not on this machine, so presence is reported as unknown.
        return [
            FetchPlanEntry(
                model_name=image_model.horde_name,
                approximate_size_gb=0.0,
                already_present=None,
                image_count=image_counts.get(image_model.slug, 0),
            )
            for image_model in models
        ]

    @override
    def prepare(self, jobs: Sequence[RenderJob], models: Sequence[ModelSpec]) -> None:
        jobs_with_new_parameters = {parameter for job in jobs for parameter in unsupported_parameters_in(job)}
        if jobs_with_new_parameters:
            logger.warning(
                "These strips set parameters that a deployed API may not accept yet. Those jobs are"
                " reported as failed if the API refuses them. The local runner does not need server"
                " support: parameters=%s",
                ", ".join(sorted(jobs_with_new_parameters)),
            )

        ignored_features = {feature for job in jobs for feature in silently_ignored_features_in(job)}
        if ignored_features:
            logger.warning(
                "These strips set features that do not have a field in this API, so the renders use the"
                " backend's defaults instead of the requested values. Render them locally instead:"
                " features=%s",
                ", ".join(sorted(ignored_features)),
            )

        available_models = self._client.available_model_names()
        if not available_models:
            logger.warning("Could not read the horde's model list; proceeding without the availability check")
            return

        for image_model in models:
            if image_model.horde_name not in available_models:
                logger.warning(
                    "No worker currently serves this model; its jobs will queue until one does: model=%s",
                    image_model.horde_name,
                )

    @override
    def render(self, job: RenderJob) -> RenderOutcome:
        request_body = HordeGenerationRequest.from_job(job).as_request_body()
        try:
            completed = self._client.generate(
                request_body=request_body,
                label=job.relative_image_path,
                expected_image_count=job.batch_size,
            )
        except RuntimeError as generate_error:
            # A refusal looks like an opaque request failure, so the error names the parameters
            # that most likely caused it instead of leaving only a 400 to go on.
            recently_added = unsupported_parameters_in(job)
            if not recently_added:
                raise
            raise RuntimeError(
                f"{generate_error}; this job sets {', '.join(recently_added)}, which this API may not accept yet",
            ) from generate_error

        model_used = completed.model_generated_with or job.horde_model_name
        return RenderOutcome(images=completed.images, model_name_used=model_used, note=completed.note)

    @override
    def close(self) -> None:
        self._client.close()
