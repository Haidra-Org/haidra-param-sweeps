# SPDX-FileCopyrightText: 2026 Tazlin
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Submit requests to the AI Horde API one at a time, poll politely, and download the images.

This is the transport shared by every part of the tool that reaches the horde, so the sweeps and
the tutorial flows queue, poll, retry and download identically. It does not depend on
specifications, strips or flows. It takes a request body that is already built and returns the
result.

It is deliberately unhurried. One request is in flight at a time and the next is not submitted
until the previous images are in hand, because the requests use other people's hardware.

**No worker attribution is collected anywhere in this module.** The status models do not include a
worker name or id, so later code cannot record one, even by accident. Kudos, wall time and
generation ids are kept, because they are needed to disclose an example's cost.

The API key is held only in a session header. Nothing here prints, logs or returns it.

The horde's other half, alchemy, is served by the same transport under its own endpoints: a form
is submitted to ``/interrogate/async`` and polled at ``/interrogate/status`` exactly as an image
request is. ``horde_sdk`` models both sides of that exchange, and its vocabulary is used here for
the states and the form names. The results themselves are kept as the mapping that the API
returned, because the SDK has result classes only for the forms that it already supported, and
the describe family now has more forms than that list.

Critical public members:

- [`HordeApiClient`][param_sweeps.horde_client.HordeApiClient]: the client itself.
- [`CompletedGeneration`][param_sweeps.horde_client.CompletedGeneration]: what one request produced.
- [`CompletedInterrogation`][param_sweeps.horde_client.CompletedInterrogation]: what one form returned.
- [`read_api_key`][param_sweeps.horde_client.read_api_key]: the ``.env`` and environment lookup.
- [`HordePacing`][param_sweeps.horde_client.HordePacing]: the client's pacing.
"""

from __future__ import annotations

import io
import logging
import os
import re
import time
from http import HTTPStatus
from pathlib import Path
from typing import Any

import requests
from horde_sdk.ai_horde_api.consts import GENERATION_STATE
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from param_sweeps.spec import TOOL_ROOT

logger = logging.getLogger(__name__)

API_KEY_VARIABLE = "AIHORDE_DEV_KEY"
"""The name of the key, both in a ``.env`` file and in the environment."""

ENV_FILE_VARIABLE = "PARAM_SWEEPS_ENV_FILE"
"""Environment variable naming the ``.env`` to read the key from."""

DEFAULT_API_BASE_URL = "https://aihorde.net/api/v2"
"""The public AI Horde API."""

CLIENT_AGENT = "haidra-param-sweeps:1.0.0:https://github.com/Haidra-Org/haidra-param-sweeps"
"""Identifies this tool to the horde, as the API requires of every client."""

ENV_ASSIGNMENT_PATTERN = re.compile(r"^\s*(?:export\s+)?(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<value>.*?)\s*$")

FINISHED_STATE: str = str(GENERATION_STATE.done)
"""The state the horde reports for a piece of work once its result is ready."""

ABANDONED_STATES: frozenset[str] = frozenset({str(GENERATION_STATE.faulted), str(GENERATION_STATE.cancelled)})
"""The states that mean the work will not produce a result, however long it is polled for."""


class HordePacing:
    """The pacing that keeps the client's load on the horde low.

    One request in flight at a time, a poll interval that will not add meaningful load, and a
    generous ceiling because a large or unpopular model can legitimately queue for a long while.
    """

    POLL_INTERVAL_SECONDS = 4.0
    MAX_WAIT_SECONDS = 1800.0
    PAUSE_BETWEEN_JOBS_SECONDS = 1.5
    REQUEST_TIMEOUT_SECONDS = 60.0
    MAX_ATTEMPTS_PER_JOB = 3

    INTERROGATION_MAX_WAIT_SECONDS = 600.0
    """A shorter ceiling than a render's, because no alchemy form is slow to run.

    A form still queues for an alchemist worker to pick it up, so the ceiling allows for queue
    time rather than processing time. After ten minutes, a missing result most likely means that
    no worker is serving the form, not that the result is nearly ready.
    """

    INTERROGATION_MAX_ATTEMPTS = 2
    """How many times one form is submitted before its failure is raised.

    A text form costs almost nothing in kudos, so a retry is cheap. A form that failed twice is
    being refused, so the run reports that instead of retrying again.
    """


class HordeGeneration(BaseModel):
    """Represents one finished generation in a status response.

    The worker that served the generation is deliberately not modelled. Examples published from
    this pipeline do not include worker attribution, and the surest way to keep it out is to not
    parse the name at all.
    """

    model_config = ConfigDict(frozen=True, extra="ignore", populate_by_name=True, protected_namespaces=())

    img: str = ""
    seed: str = ""
    model_used: str = Field(default="", alias="model")
    generation_id: str = Field(default="", alias="id")
    censored: bool = False
    gen_metadata: list[dict[str, Any]] = Field(default_factory=list)


class HordeAsyncRequestAccepted(BaseModel):
    """Represents the response to a submitted generation request."""

    model_config = ConfigDict(frozen=True, extra="ignore", populate_by_name=True, protected_namespaces=())

    request_id: str = Field(alias="id")
    kudos: float = 0.0


class HordeServedModel(BaseModel):
    """Represents one model that the horde's workers currently serve."""

    model_config = ConfigDict(frozen=True, extra="ignore", protected_namespaces=())

    name: str = ""
    count: int = 0


class HordeStatusResponse(BaseModel):
    """Represents a ``/generate/status`` or ``/generate/check`` response."""

    model_config = ConfigDict(frozen=True, extra="ignore", protected_namespaces=())

    done: bool = False
    faulted: bool = False
    is_possible: bool = True
    queue_position: int = 0
    wait_time: int = 0
    kudos: float = 0.0
    generations: list[HordeGeneration] = Field(default_factory=list)


class CompletedGeneration(BaseModel):
    """Represents everything one finished request produced.

    It records the request's cost and identity: kudos, wall time and the generation ids. By
    design, it does not record the worker that served the request.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, protected_namespaces=())

    request_id: str
    images: list[Image.Image] = Field(min_length=1)
    generations: list[HordeGeneration] = Field(min_length=1)
    kudos_estimate: float = 0.0
    kudos_charged: float = 0.0
    wall_time_seconds: float = 0.0

    @property
    def model_generated_with(self) -> str:
        """Return the model reported by the first generation."""
        return self.generations[0].model_used

    @property
    def note(self) -> str | None:
        """Return a note about the request, or None when there is nothing to note."""
        return "censored by the horde's safety checker" if any(gen.censored for gen in self.generations) else None


class HordeInterrogationAccepted(BaseModel):
    """Represents the response to a submitted alchemy request."""

    model_config = ConfigDict(frozen=True, extra="ignore", populate_by_name=True, protected_namespaces=())

    request_id: str = Field(alias="id")


class HordeFormStatus(BaseModel):
    """Represents one form's state, and its result once it has one.

    The result is kept as the API's mapping instead of being parsed into a result class.
    ``horde_sdk`` has a class for each form that it supported, and the forms added to the live
    alchemist since then return shapes that those classes do not describe. With the mapping, a
    form that the SDK does not support yet is published instead of discarded.
    """

    model_config = ConfigDict(frozen=True, extra="ignore", protected_namespaces=())

    form: str = ""
    state: str = ""
    result: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_done(self) -> bool:
        """Return whether this form has finished and has its result."""
        return self.state == FINISHED_STATE


class HordeInterrogationStatusResponse(BaseModel):
    """Represents an ``/interrogate/status`` response."""

    model_config = ConfigDict(frozen=True, extra="ignore", protected_namespaces=())

    state: str = ""
    forms: list[HordeFormStatus] = Field(default_factory=list)


class CompletedInterrogation(BaseModel):
    """Represents what one alchemy form returned for one image.

    As with a generation, the worker that served it is deliberately not modelled.
    """

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    request_id: str
    form: str
    result: dict[str, Any]
    wall_time_seconds: float = 0.0


class FormRefusedError(RuntimeError):
    """The API refused the form itself, instead of failing to run it.

    The client raises this when the deployed horde rejects a form that its request schema does not
    accept. The rejection shows what the live alchemist serves and is not a transport failure, so
    it has its own type and is never retried.
    """


class MissingApiKeyError(RuntimeError):
    """The API key was not found in the environment or the ``.env`` file."""


def read_api_key(*, env_file: Path | None) -> str:
    """Return the AI Horde API key, from the environment or the given ``.env``.

    The environment takes precedence, so a run can use a different key without editing a file. The
    key itself is never logged: a failure reports only the locations that were checked.

    Args:
        env_file: The ``.env`` to read, or None to use only the environment.

    Returns:
        The key.

    Raises:
        MissingApiKeyError: If no key was found, or the one found is empty.
    """
    key_from_environment = os.environ.get(API_KEY_VARIABLE, "").strip()
    if key_from_environment:
        logger.info("Using the API key from the environment: variable=%s", API_KEY_VARIABLE)
        return key_from_environment

    if env_file is None:
        raise MissingApiKeyError(f"{API_KEY_VARIABLE} is not set and no .env was given")

    if not env_file.is_file():
        raise MissingApiKeyError(f"{API_KEY_VARIABLE} is not set and {env_file} does not exist")

    key_from_file = read_env_value(env_file=env_file, variable_name=API_KEY_VARIABLE)
    if not key_from_file:
        raise MissingApiKeyError(f"{env_file} has no non-empty {API_KEY_VARIABLE}")

    logger.info("Using the API key from %s", env_file)
    return key_from_file


def default_env_file() -> Path | None:
    """Return the ``.env`` to read by default, or None when there is none.

    The file named by ``PARAM_SWEEPS_ENV_FILE`` comes first. Otherwise the ``.env`` at the root of
    this repository is used if it exists. A file anywhere else is given with ``--env-file`` or the
    environment variable.
    """
    from_environment = os.environ.get(ENV_FILE_VARIABLE, "").strip()
    if from_environment:
        return Path(from_environment)

    repository_env = TOOL_ROOT / ".env"
    return repository_env if repository_env.is_file() else None


def read_env_value(*, env_file: Path, variable_name: str) -> str:
    """Return one variable's value from a ``.env``, or the empty string when it is not there."""
    for line in env_file.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("#"):
            continue

        assignment = ENV_ASSIGNMENT_PATTERN.match(line)
        if assignment is None or assignment.group("name") != variable_name:
            continue

        return assignment.group("value").strip().strip("\"'")

    return ""


def refusal_reason(*, response: requests.Response, form: str) -> str:
    """Return, in one sentence, the API's message about a form that it did not accept.

    The deployed API's response to a form that its request schema does not include is a validation
    error that names the offending field and lists the forms that it does accept. That list shows
    what the live alchemist serves, which is the most useful information about the published gap,
    so it is kept verbatim instead of summarised.
    """
    try:
        body = response.json()
    except ValueError:
        return f"the API refused the {form} form: {response.text.strip()}"

    errors = body.get("errors") if isinstance(body, dict) else None
    stated = "; ".join(str(detail) for detail in errors.values()) if isinstance(errors, dict) else str(body)
    return f"the API refused the {form} form: {stated}"


class HordeApiClient:
    """Submits one generation request at a time to the AI Horde and returns the results.

    Network Behavior:
        Each request is one ``POST /generate/async``, then ``/generate/check`` polled until it is
        done, then ``/generate/status`` once to collect the image URLs, then the downloads. A
        request that faults or times out is retried a small number of times before the failure is
        raised to the caller.

    Thread Safety:
        Not thread-safe, by design. The pacing between requests is per-client state.
    """

    def __init__(
        self,
        *,
        api_key: str,
        api_base_url: str = DEFAULT_API_BASE_URL,
        session: requests.Session | None = None,
    ) -> None:
        """Configure a client.

        Args:
            api_key: The AI Horde API key. Held only in the session header.
            api_base_url: The API root to use.
            session: An existing session to use. The tests use it to substitute a transport.
        """
        self._api_base_url = api_base_url.rstrip("/")
        self._session = session if session is not None else requests.Session()
        self._session.headers.update(
            {
                "apikey": api_key,
                "Client-Agent": CLIENT_AGENT,
                "accept": "application/json",
            },
        )
        self._is_first_request = True

    @property
    def api_base_url(self) -> str:
        """Return this client's API root."""
        return self._api_base_url

    def close(self) -> None:
        """Close the underlying session."""
        self._session.close()

    def generate(
        self, *, request_body: dict[str, Any], label: str, expected_image_count: int = 1
    ) -> CompletedGeneration:
        """Submit one request, wait for it, download its images, and return the whole result.

        Args:
            request_body: The already-built ``POST /generate/async`` body.
            label: The request's name in logs and errors.
            expected_image_count: How many images the request requested.

        Returns:
            The finished generation.

        Raises:
            RuntimeError: If the horde could not produce the images after every attempt.
        """
        last_error = ""
        for attempt in range(1, HordePacing.MAX_ATTEMPTS_PER_JOB + 1):
            if not self._is_first_request:
                time.sleep(HordePacing.PAUSE_BETWEEN_JOBS_SECONDS)
            self._is_first_request = False

            try:
                return self._generate_once(
                    request_body=request_body,
                    label=label,
                    expected_image_count=expected_image_count,
                )
            except (requests.RequestException, TimeoutError) as request_error:
                last_error = str(request_error)
                logger.warning(
                    "Horde request failed, retrying: attempt=%s/%s label=%s error=%s",
                    attempt,
                    HordePacing.MAX_ATTEMPTS_PER_JOB,
                    label,
                    last_error,
                )

        raise RuntimeError(f"the horde could not render {label}: {last_error}")

    def available_model_names(self) -> set[str]:
        """Return the names of the image models that workers currently serve, or an empty set.

        An empty set means the model list could not be read. Callers treat that as "continue
        without the availability check" rather than as "no models are served".
        """
        try:
            models_response = self._session.get(
                f"{self._api_base_url}/status/models",
                params={"type": "image"},
                timeout=HordePacing.REQUEST_TIMEOUT_SECONDS,
            )
            models_response.raise_for_status()
        except requests.RequestException as request_error:
            logger.warning("Could not read the horde's model list: error=%s", request_error)
            return set()

        served_models = [HordeServedModel.model_validate(entry) for entry in models_response.json()]
        return {served_model.name for served_model in served_models if served_model.name}

    def interrogate(self, *, form: str, source_image_base64: str) -> CompletedInterrogation:
        """Submit one alchemy form for one image to the horde, and return the result.

        Each request has one form, because the API refuses the whole request if it does not accept
        one of its forms. In a batch, one refused form would also fail the accepted ones.

        Args:
            form: The form to run, as ``horde_sdk``'s vocabulary spells it.
            source_image_base64: The image, base64-encoded, for the API's ``source_image`` field.

        Returns:
            The finished form and its result.

        Raises:
            FormRefusedError: If the API's request schema does not accept the form.
            RuntimeError: If the horde could not run the form after every attempt.
        """
        last_error = ""
        for attempt in range(1, HordePacing.INTERROGATION_MAX_ATTEMPTS + 1):
            if not self._is_first_request:
                time.sleep(HordePacing.PAUSE_BETWEEN_JOBS_SECONDS)
            self._is_first_request = False

            try:
                return self._interrogate_once(form=form, source_image_base64=source_image_base64)
            except (requests.RequestException, TimeoutError) as request_error:
                last_error = str(request_error)
                logger.warning(
                    "Horde alchemy request failed, retrying: attempt=%s/%s form=%s error=%s",
                    attempt,
                    HordePacing.INTERROGATION_MAX_ATTEMPTS,
                    form,
                    last_error,
                )

        raise RuntimeError(f"the horde could not run the {form} form: {last_error}")

    def _interrogate_once(self, *, form: str, source_image_base64: str) -> CompletedInterrogation:
        """Submit one form and wait for its result.

        Raises:
            FormRefusedError: If the API's request schema does not accept the form.
            RuntimeError: If the horde faulted or cancelled the request.
            TimeoutError: If the ceiling is reached before the form finishes.
            requests.RequestException: On any transport or HTTP failure, which the caller retries.
        """
        started_at = time.monotonic()
        submit_response = self._session.post(
            f"{self._api_base_url}/interrogate/async",
            json={"forms": [{"name": form}], "source_image": source_image_base64},
            timeout=HordePacing.REQUEST_TIMEOUT_SECONDS,
        )
        if submit_response.status_code == HTTPStatus.BAD_REQUEST:
            raise FormRefusedError(refusal_reason(response=submit_response, form=form))
        submit_response.raise_for_status()
        accepted = HordeInterrogationAccepted.model_validate(submit_response.json())

        finished = self._wait_for_form(request_id=accepted.request_id, form=form)
        return CompletedInterrogation(
            request_id=accepted.request_id,
            form=finished.form or form,
            result=finished.result,
            wall_time_seconds=round(time.monotonic() - started_at, 1),
        )

    def _wait_for_form(self, *, request_id: str, form: str) -> HordeFormStatus:
        """Poll until the horde reports this form done, and return it.

        Raises:
            RuntimeError: If the horde faulted or cancelled the request.
            TimeoutError: If the ceiling is reached before the form finishes.
        """
        deadline = time.monotonic() + HordePacing.INTERROGATION_MAX_WAIT_SECONDS

        while time.monotonic() < deadline:
            status_response = self._session.get(
                f"{self._api_base_url}/interrogate/status/{request_id}",
                timeout=HordePacing.REQUEST_TIMEOUT_SECONDS,
            )
            status_response.raise_for_status()
            status = HordeInterrogationStatusResponse.model_validate(status_response.json())

            if status.state in ABANDONED_STATES:
                raise RuntimeError(f"the horde reported the {form} form {status.state}")

            for form_status in status.forms:
                if form_status.is_done:
                    return form_status

            logger.debug("Waiting on the horde: form=%s state=%s", form, status.state)
            time.sleep(HordePacing.POLL_INTERVAL_SECONDS)

        raise TimeoutError(
            f"the horde did not finish the {form} form within {HordePacing.INTERROGATION_MAX_WAIT_SECONDS:.0f}s",
        )

    def _generate_once(
        self, *, request_body: dict[str, Any], label: str, expected_image_count: int
    ) -> CompletedGeneration:
        """Submit one request, wait for it, and download its images.

        Raises:
            RuntimeError: If the horde reported the request as faulted, impossible, or returned
                fewer generations than were requested.
            requests.RequestException: On any transport or HTTP failure, which the caller retries.
        """
        started_at = time.monotonic()
        submit_response = self._session.post(
            f"{self._api_base_url}/generate/async",
            json=request_body,
            timeout=HordePacing.REQUEST_TIMEOUT_SECONDS,
        )
        submit_response.raise_for_status()
        accepted = HordeAsyncRequestAccepted.model_validate(submit_response.json())

        self._wait_for_completion(request_id=accepted.request_id, label=label)

        status_response = self._session.get(
            f"{self._api_base_url}/generate/status/{accepted.request_id}",
            timeout=HordePacing.REQUEST_TIMEOUT_SECONDS,
        )
        status_response.raise_for_status()
        status = HordeStatusResponse.model_validate(status_response.json())

        if len(status.generations) < expected_image_count:
            raise RuntimeError(
                f"the horde returned {len(status.generations)} of the {expected_image_count} generations"
                f" requested for {label}",
            )

        generations = status.generations[:expected_image_count]
        return CompletedGeneration(
            request_id=accepted.request_id,
            images=[self.download_image(generation.img) for generation in generations],
            generations=generations,
            kudos_estimate=accepted.kudos,
            kudos_charged=status.kudos,
            wall_time_seconds=round(time.monotonic() - started_at, 1),
        )

    def _wait_for_completion(self, *, request_id: str, label: str) -> None:
        """Poll until the horde reports this request done.

        Raises:
            RuntimeError: If the request faulted or no worker can serve it.
            TimeoutError: If the ceiling is reached before the request completes.
        """
        deadline = time.monotonic() + HordePacing.MAX_WAIT_SECONDS

        while time.monotonic() < deadline:
            check_response = self._session.get(
                f"{self._api_base_url}/generate/check/{request_id}",
                timeout=HordePacing.REQUEST_TIMEOUT_SECONDS,
            )
            check_response.raise_for_status()
            status = HordeStatusResponse.model_validate(check_response.json())

            if status.faulted:
                raise RuntimeError(f"the horde faulted the request for {label}")
            if not status.is_possible:
                raise RuntimeError(f"no worker can serve the request for {label}")
            if status.done:
                return

            logger.debug(
                "Waiting on the horde: label=%s queue_position=%s wait_time=%s",
                label,
                status.queue_position,
                status.wait_time,
            )
            time.sleep(HordePacing.POLL_INTERVAL_SECONDS)

        raise TimeoutError(f"the horde did not finish {label} within {HordePacing.MAX_WAIT_SECONDS:.0f}s")

    def download_image(self, image_url: str) -> Image.Image:
        """Download and decode one finished generation.

        Raises:
            RuntimeError: If the generation does not have a URL.
            requests.RequestException: On any transport or HTTP failure.
        """
        if not image_url:
            raise RuntimeError("the horde returned a generation with no image URL")

        image_response = self._session.get(image_url, timeout=HordePacing.REQUEST_TIMEOUT_SECONDS)
        image_response.raise_for_status()

        # An alpha channel is kept when present. The transparent workflow's output is the cutout,
        # and flattening it to RGB here would discard the result without any error.
        with Image.open(io.BytesIO(image_response.content)) as downloaded:
            return downloaded.convert("RGBA" if "A" in downloaded.getbands() else "RGB")
