# SPDX-FileCopyrightText: 2026 Tazlin
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Talk to the AI Horde API: submit one request, wait for it politely, and bring the images back.

This is the transport every part of the tool that reaches the horde shares, so the sweeps and the
tutorial flows queue, poll, retry and download identically. It knows nothing about specifications,
strips or flows: it takes a request body that is already built and returns what came back.

It is deliberately unhurried. One request is in flight at a time and the next is not submitted
until the previous images are in hand, because this is other people's hardware being asked for a
favour.

**No worker attribution is collected anywhere in this module.** The status models do not carry a
worker name or id, so nothing downstream can record one even by accident. Kudos, wall time and
generation ids are kept: they are what an honest disclosure of what an example cost needs.

The API key is held only in a session header. Nothing here prints, logs or returns it.

The horde's other half, alchemy, is served by the same transport under its own endpoints: a form
is submitted to ``/interrogate/async`` and polled at ``/interrogate/status`` exactly as an image
request is. ``horde_sdk`` models both sides of that exchange, and its vocabulary is used here for
the states and the form names; the answers themselves are kept as the mapping the API returned,
because the SDK models a result class only for the forms it already knew about and the describe
family has outgrown that list.

Critical public members:

- [`HordeApiClient`][param_sweeps.horde_client.HordeApiClient]: the client itself.
- [`CompletedGeneration`][param_sweeps.horde_client.CompletedGeneration]: what one request produced.
- [`CompletedInterrogation`][param_sweeps.horde_client.CompletedInterrogation]: what one form said.
- [`read_api_key`][param_sweeps.horde_client.read_api_key]: the ``.env`` and environment lookup.
- [`HordePacing`][param_sweeps.horde_client.HordePacing]: how politely the client talks.
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

logger = logging.getLogger(__name__)

API_KEY_VARIABLE = "AIHORDE_DEV_KEY"
"""The name of the key, both in the frontpage ``.env`` and in the environment."""

ENV_FILE_VARIABLE = "PARAM_SWEEPS_ENV_FILE"
"""Environment variable naming the ``.env`` to read the key from."""

DEFAULT_API_BASE_URL = "https://aihorde.net/api/v2"
"""The public AI Horde API."""

CLIENT_AGENT = "haidra-param-sweeps:1.0.0:https://github.com/Haidra-Org/haidra-assets"
"""Identifies this tool to the horde, as the API asks every client to do."""

ENV_ASSIGNMENT_PATTERN = re.compile(r"^\s*(?:export\s+)?(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<value>.*?)\s*$")

FINISHED_STATE: str = str(GENERATION_STATE.done)
"""The state the horde reports a piece of work in once its answer is ready."""

ABANDONED_STATES: frozenset[str] = frozenset({str(GENERATION_STATE.faulted), str(GENERATION_STATE.cancelled)})
"""The states that mean the work will not produce an answer, however long it is polled for."""


class HordePacing:
    """How politely the client talks to the horde.

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

    A form still queues for an alchemist worker to pick it up, so the ceiling is a queue's
    patience rather than a processor's; ten minutes is well past the point where waiting longer
    is more likely to mean nobody is serving the form than that the answer is nearly ready.
    """

    INTERROGATION_MAX_ATTEMPTS = 2
    """How many times one form is submitted before its failure is raised.

    A text form costs almost nothing in kudos, so a retry is cheap, but a form that failed twice
    is being refused rather than being unlucky and the run should say so rather than keep asking.
    """


class HordeGeneration(BaseModel):
    """Represents one finished generation in a status response.

    The worker that served the generation is deliberately not modelled. Examples published from
    this pipeline carry no worker attribution, and the surest way to keep that true is for the
    name never to be parsed in the first place.
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
    """Represents one model the horde's workers currently serve."""

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

    What is recorded is what the request cost and what identifies it: kudos, wall time and the
    generation ids. Who served it is not part of this record by design.
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
        """Return the model the first generation reports having been made with."""
        return self.generations[0].model_used

    @property
    def note(self) -> str | None:
        """Return a note worth keeping about the request, or None when there is nothing to say."""
        return "censored by the horde's safety checker" if any(gen.censored for gen in self.generations) else None


class HordeInterrogationAccepted(BaseModel):
    """Represents the response to a submitted alchemy request."""

    model_config = ConfigDict(frozen=True, extra="ignore", populate_by_name=True, protected_namespaces=())

    request_id: str = Field(alias="id")


class HordeFormStatus(BaseModel):
    """Represents one form's state, and its answer once it has one.

    The answer is kept as the mapping the API returned rather than parsed into a result class.
    ``horde_sdk`` models a class per form it knew about, and the forms the live alchemist has
    gained since answer in shapes none of those classes describe; keeping the mapping means a
    form the SDK has not caught up with is published rather than discarded.
    """

    model_config = ConfigDict(frozen=True, extra="ignore", protected_namespaces=())

    form: str = ""
    state: str = ""
    result: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_done(self) -> bool:
        """Return whether this form has finished and carries its answer."""
        return self.state == FINISHED_STATE


class HordeInterrogationStatusResponse(BaseModel):
    """Represents an ``/interrogate/status`` response."""

    model_config = ConfigDict(frozen=True, extra="ignore", protected_namespaces=())

    state: str = ""
    forms: list[HordeFormStatus] = Field(default_factory=list)


class CompletedInterrogation(BaseModel):
    """Represents what one alchemy form said about one image.

    As with a generation, who served it is deliberately not modelled.
    """

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    request_id: str
    form: str
    result: dict[str, Any]
    wall_time_seconds: float = 0.0


class FormRefusedError(RuntimeError):
    """The API refused the form itself, rather than failing to run it.

    This is what the deployed horde says about a form its request schema does not accept, which
    is a fact about what the live alchemist serves rather than a transport failure, so it is
    raised as its own type and never retried.
    """


class MissingApiKeyError(RuntimeError):
    """The API key could not be found where it was looked for."""


def read_api_key(*, env_file: Path | None) -> str:
    """Return the AI Horde API key, from the environment or the given ``.env``.

    The environment wins, so a run can be pointed at a different key without editing a file. The
    key itself is never logged: a failure reports only where it looked.

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
    """Return the ``.env`` to read by default: the environment's, else the frontpage repository's.

    This tool lives in the ``haidra-assets`` submodule of the frontpage repository, whose ``.env``
    holds the development key. That relative position is the default and nothing more: any other
    location is given with the flag or the environment variable.
    """
    from_environment = os.environ.get(ENV_FILE_VARIABLE, "").strip()
    if from_environment:
        return Path(from_environment)

    assets_repository_root = Path(__file__).resolve().parents[3]
    frontpage_env = assets_repository_root.parent.parent / ".env"
    return frontpage_env if frontpage_env.is_file() else None


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
    """Return, in one sentence, what the API said about a form it would not accept.

    The deployed API answers a form its request schema does not carry with a validation error
    naming the offending field and listing the forms it does accept. That list is the live
    alchemist's own account of what it serves, which is the most useful thing the published
    evidence can say about the gap, so it is kept verbatim rather than summarised.
    """
    try:
        body = response.json()
    except ValueError:
        return f"the API refused the {form} form: {response.text.strip()}"

    errors = body.get("errors") if isinstance(body, dict) else None
    stated = "; ".join(str(detail) for detail in errors.values()) if isinstance(errors, dict) else str(body)
    return f"the API refused the {form} form: {stated}"


class HordeApiClient:
    """Submits one generation request at a time to the AI Horde and returns what came back.

    Network Behavior:
        Each request is one ``POST /generate/async``, then ``/generate/check`` polled until it is
        done, then ``/generate/status`` once to collect the image URLs, then the downloads. A
        request that faults or times out is retried a small number of times before the failure is
        raised to the caller.

    Thread Safety:
        Not thread-safe, and deliberately so: the pacing between requests is per-client state.
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
            api_base_url: The API root to talk to.
            session: An existing session to use, which is how the tests substitute a transport.
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
        """Return the API root this client talks to."""
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
            label: What to call this request in logs and errors.
            expected_image_count: How many images the request asked for.

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
        """Return the names of the image models workers currently serve, or an empty set.

        An empty set means the question could not be answered, which the callers treat as "carry
        on without the availability check" rather than as "no models are served".
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
        """Ask the horde to run one alchemy form on one image, and return what it said.

        One form per request rather than several at once, because the API refuses the whole
        request over a form it does not accept, and a batch would take the forms that do work
        down with the one that does not.

        Args:
            form: The form to run, as ``horde_sdk``'s vocabulary spells it.
            source_image_base64: The image, base64-encoded, as the API's ``source_image`` takes it.

        Returns:
            The finished form and its answer.

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
        """Submit one form and see it through to its answer.

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
        """Submit one request and see it through to the downloaded images.

        Raises:
            RuntimeError: If the horde reported the request as faulted, impossible, or returned
                fewer generations than were asked for.
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
                f" asked for for {label}",
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
            RuntimeError: If the generation carries no URL.
            requests.RequestException: On any transport or HTTP failure.
        """
        if not image_url:
            raise RuntimeError("the horde returned a generation with no image URL")

        image_response = self._session.get(image_url, timeout=HordePacing.REQUEST_TIMEOUT_SECONDS)
        image_response.raise_for_status()

        # An alpha channel is kept when there is one: the transparent workflow's whole output is
        # the cutout, and flattening it to RGB here would silently throw the result away.
        with Image.open(io.BytesIO(image_response.content)) as downloaded:
            return downloaded.convert("RGBA" if "A" in downloaded.getbands() else "RGB")
