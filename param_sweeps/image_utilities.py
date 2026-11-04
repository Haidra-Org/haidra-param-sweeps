# SPDX-FileCopyrightText: 2026 Tazlin
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Draw the control maps through horde-image-utilities, the service that an alchemist worker runs.

The control maps are not produced by horde-engine or by the public API. A worker that serves the
``annotation`` form runs horde-image-utilities as a separate process, and requesting maps from
that service is the only way to publish a detector's output without approximating it with some
other installed tool. This module is the transport and the backend for that service: one
``POST /annotators/{control_type}`` per detector, with the source as a file upload.

The service reports its capabilities, so the published gaps are the service's messages instead
of this tool's guesses. ``GET /annotators`` reports every registered detector's runtime, whether
its dependencies import, and whether its weights are on disk. For a detector that it does not
run, the annotate call returns a status and a ``detail`` sentence. Both go into the manifest as
the reason that a control map is missing.

A detector's first call can be slow: weights are fetched on demand, and the segmentation and depth
models are large. The annotate timeout is sized for that rather than for a warm call.

Critical public members:

- [`HordeImageUtilitiesBackend`][param_sweeps.image_utilities.HordeImageUtilitiesBackend]: the
  backend that the alchemy campaign uses for the ``annotation`` form.
- [`resolve_service_url`][param_sweeps.image_utilities.resolve_service_url]: the service URL from
  the flag, the environment, or the default.
"""

from __future__ import annotations

import io
import logging
import os
from http import HTTPStatus
from typing import Any, override
from urllib.parse import urlsplit

import requests
from PIL import Image
from pydantic import BaseModel, ConfigDict

from param_sweeps.alchemy import AlchemyBackend, AlchemyOutcome

logger = logging.getLogger(__name__)

DEFAULT_SERVICE_URL = "http://127.0.0.1:7860"
"""Where horde-image-utilities listens when nothing else is configured."""

SERVICE_URL_ENVIRONMENT_VARIABLE = "PARAM_SWEEPS_HIU_URL"
"""Environment variable naming the horde-image-utilities service to draw the control maps on."""

BACKEND_NAME = "horde-image-utilities"
"""The backend name recorded in the sidecar beside every control map."""

ANNOTATORS_PATH = "/annotators"
OPENAPI_PATH = "/openapi.json"
"""The only document in which the service states its version."""

DETECTION_RESOLUTION_CEILING = 1024
"""The longest side, in pixels, that a detector is asked to work at. This is the size of the anchor source images."""


class ImageUtilitiesPacing:
    """The service's response timeouts."""

    REQUEST_TIMEOUT_SECONDS = 30.0
    """For the calls that only read the service's state."""

    ANNOTATE_TIMEOUT_SECONDS = 600.0
    """For one detection, which on a cold detector includes fetching the weights first."""


class AnnotatorReport(BaseModel):
    """Represents the service's report on one registered detector."""

    model_config = ConfigDict(frozen=True, extra="ignore", protected_namespaces=())

    name: str
    runtime: str
    available: bool
    weights_present: str
    loaded: bool


def resolve_service_url(*, requested: str | None) -> str:
    """Return the horde-image-utilities URL from the flag, the environment, or the default, in that order.

    Args:
        requested: The value given on the command line, or None.

    Returns:
        The service root, without a trailing slash.
    """
    candidate = requested if requested is not None else os.environ.get(SERVICE_URL_ENVIRONMENT_VARIABLE)
    if candidate is None or not candidate.strip():
        return DEFAULT_SERVICE_URL

    return candidate.strip().rstrip("/")


class HordeImageUtilitiesBackend(AlchemyBackend):
    """Draws control maps by requesting them from a horde-image-utilities service, and publishes the result.

    This backend handles only the ``annotation`` form. Every other form is served by
    horde-engine or by the live horde, and is reported unavailable here rather than approximated.

    Network Behavior:
        ``prepare`` reads the detector report once, and the version if the service states one. Each
        form is one ``POST /annotators/{control_type}`` with the source as a file upload. A
        status that the service explains is published as a gap with its explanation. A transport
        failure is raised, because a dropped connection does not show what the service serves.

    Thread Safety:
        Not thread-safe: the detector report is per-backend state, written once in ``prepare``.
    """

    def __init__(self, *, service_url: str, session: requests.Session | None = None) -> None:
        """Configure a control-map backend.

        Args:
            service_url: The service root, without a trailing slash.
            session: The session to use, or None to open one. ``close`` closes a session opened
                here. A session passed in is closed by its owner.
        """
        self._service_url = service_url
        self._session = session if session is not None else requests.Session()
        self._owns_session = session is None
        self._reports: dict[str, AnnotatorReport] = {}
        self._service_version: str | None = None

    @property
    @override
    def name(self) -> str:
        """Return the backend's name, as recorded in every sidecar it produces."""
        return BACKEND_NAME

    @property
    @override
    def service_host(self) -> str | None:
        """Return the host that drew the control maps."""
        return urlsplit(self._service_url).netloc or self._service_url

    @property
    @override
    def service_version(self) -> str | None:
        """Return the service's reported version, or None when it was not requested or not reported."""
        return self._service_version

    @override
    def prepare(self) -> None:
        """Read the detector report and the service's version.

        Raises:
            RuntimeError: If the service cannot be reached or does not report its detectors. Every
                control map comes from the service, so a run that cannot query it has no control
                maps to publish.
        """
        try:
            response = self._session.get(
                f"{self._service_url}{ANNOTATORS_PATH}",
                timeout=ImageUtilitiesPacing.REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            reported = response.json()
        except (requests.RequestException, ValueError) as request_error:
            raise RuntimeError(
                f"horde-image-utilities at {self._service_url} did not report its detectors: {request_error}",
            ) from request_error

        self._reports = {
            report.name: report for report in (AnnotatorReport.model_validate(entry) for entry in reported)
        }
        self._service_version = self._reported_version()
        logger.info(
            "Control-map service ready: url=%s detectors=%s version=%s",
            self._service_url,
            len(self._reports),
            self._service_version,
        )

    def _reported_version(self) -> str | None:
        """Return the service's stated version, or None when it does not state one.

        The version is only stated in the OpenAPI document, and it is worth having in the sidecar
        because a control map is the output of a particular build of a particular detector. It is
        not worth failing a run over, so a service that does not serve the document is recorded
        without a version instead of as unusable.
        """
        try:
            response = self._session.get(
                f"{self._service_url}{OPENAPI_PATH}",
                timeout=ImageUtilitiesPacing.REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            document = response.json()
        except (requests.RequestException, ValueError) as request_error:
            logger.info("The control-map service did not state a version: %s", request_error)
            return None

        information = document.get("info") if isinstance(document, dict) else None
        stated = information.get("version") if isinstance(information, dict) else None
        return str(stated) if stated else None

    @override
    def apply(self, *, form: str, control_type: str | None, source_image: Image.Image) -> AlchemyOutcome:
        """Draw one control map, and return the outcome.

        Raises:
            RuntimeError: If the service could not be reached at all.
        """
        if control_type is None:
            return AlchemyOutcome(
                unavailable_reason=(
                    f"the {form} form is not a control map, and horde-image-utilities only draws control maps"
                ),
            )

        refusal = self._reported_refusal(control_type)
        if refusal is not None:
            logger.warning("Control map unavailable: control_type=%s reason=%s", control_type, refusal)
            return AlchemyOutcome(unavailable_reason=refusal)

        response = self._annotated(control_type=control_type, source_image=source_image)
        if response.status_code != HTTPStatus.OK:
            stated = _stated_detail(response=response, control_type=control_type)
            logger.warning("Control map unavailable: control_type=%s reason=%s", control_type, stated)
            return AlchemyOutcome(unavailable_reason=stated)

        with Image.open(io.BytesIO(response.content)) as drawn:
            return AlchemyOutcome(image=drawn.convert("RGB"), model_name_used=control_type)

    def _reported_refusal(self, control_type: str) -> str | None:
        """Return why the detector report rules this detector out, or None when it does not."""
        report = self._reports.get(control_type)
        if report is None:
            return (
                f"horde-image-utilities does not register a {control_type} detector;"
                f" it registers {', '.join(sorted(self._reports))}"
            )
        if not report.available:
            return (
                f"horde-image-utilities reports the {control_type} detector unavailable:"
                f" its {report.runtime} runtime is not importable, and its weights are {report.weights_present}"
            )
        return None

    def _annotated(self, *, control_type: str, source_image: Image.Image) -> requests.Response:
        """Return the service's response for one detection, whatever that response is.

        Raises:
            RuntimeError: On any transport failure, which does not show what the service
                serves and so is not published as a gap.
        """
        upload = io.BytesIO()
        source_image.save(upload, format="PNG")
        try:
            return self._session.post(
                f"{self._service_url}{ANNOTATORS_PATH}/{control_type}",
                files={"file": (f"source-{control_type}.png", upload.getvalue(), "image/png")},
                data={"resolution": _detection_resolution(source_image)},
                timeout=ImageUtilitiesPacing.ANNOTATE_TIMEOUT_SECONDS,
            )
        except requests.RequestException as request_error:
            raise RuntimeError(
                f"the request to horde-image-utilities at {self._service_url} for the"
                f" {control_type} control map failed: {request_error}",
            ) from request_error

    @override
    def close(self) -> None:
        """Close the session, if this backend opened it."""
        if self._owns_session:
            self._session.close()


def _detection_resolution(source_image: Image.Image) -> int:
    """Return the resolution to detect at: the source's longer side, up to the ceiling.

    Detecting above the source's own size invents detail that the picture does not contain, and
    the service caps the resolution that it accepts, so the requested resolution is bounded on
    both sides.
    """
    return min(max(source_image.width, source_image.height), DETECTION_RESOLUTION_CEILING)


def _stated_detail(*, response: requests.Response, control_type: str) -> str:
    """Return, in one sentence, the service's message about a detection that it did not run.

    For a missing weight, an unimplemented detector or a full queue, the service responds with its
    own ``detail`` sentence. That is the most useful information about the published gap, so it
    is kept verbatim instead of summarised.
    """
    try:
        body: Any = response.json()
    except ValueError:
        return f"horde-image-utilities refused the {control_type} control map: {response.text.strip()}"

    stated = body.get("detail") if isinstance(body, dict) else None
    return (
        f"horde-image-utilities refused the {control_type} control map"
        f" ({response.status_code}): {stated if stated else body}"
    )
