# SPDX-FileCopyrightText: 2026 Tazlin
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Check the control-map backend: what it draws, what it publishes as a gap, and what it records."""

from __future__ import annotations

import io
import json
from http import HTTPStatus
from typing import Any, override

import pytest
import requests
from PIL import Image

from param_sweeps.image_utilities import (
    DEFAULT_SERVICE_URL,
    SERVICE_URL_ENVIRONMENT_VARIABLE,
    HordeImageUtilitiesBackend,
    resolve_service_url,
)

SERVICE_URL = "http://127.0.0.1:7860"
CONTROL_TYPE = "canny"
SERVICE_VERSION = "0.1.dev8+gdd3a836f0"

AVAILABLE_REPORT = [
    {"name": CONTROL_TYPE, "runtime": "none", "available": True, "weights_present": "present", "loaded": False},
    {"name": "openpose", "runtime": "torch", "available": True, "weights_present": "present", "loaded": True},
    {
        "name": "mediapipe_face",
        "runtime": "mediapipe",
        "available": False,
        "weights_present": "present",
        "loaded": False,
    },
]
"""What the service reports about its detectors, in the shape it reports it."""


class ScriptedService(requests.adapters.BaseAdapter):
    """Answers the backend from a script, so the whole exchange runs without a service.

    Mounted on a real session and handed to a real backend, which is what makes this a test of the
    transport rather than of a stand-in for it.
    """

    def __init__(self, *, annotate_status: int = 200, annotate_body: dict[str, Any] | None = None) -> None:
        """Script the answers the backend can ask for."""
        self.annotate_status = annotate_status
        self.annotate_body = annotate_body
        self.annotate_bodies: list[bytes] = []
        self.requested_paths: list[str] = []
        self.drawn_bytes = _png_bytes(Image.new("RGB", (8, 8), (255, 255, 255)))

    @override
    def send(self, request: requests.PreparedRequest, *args: Any, **kwargs: Any) -> requests.Response:
        url = request.url or ""
        self.requested_paths.append(url)
        if url.endswith("/openapi.json"):
            return _response(200, json.dumps({"info": {"version": SERVICE_VERSION}}).encode("utf-8"), url)
        if url.endswith("/annotators"):
            return _response(200, json.dumps(AVAILABLE_REPORT).encode("utf-8"), url)

        submitted = request.body if request.body is not None else b""
        self.annotate_bodies.append(submitted if isinstance(submitted, bytes) else submitted.encode("utf-8"))
        if self.annotate_body is not None:
            return _response(self.annotate_status, json.dumps(self.annotate_body).encode("utf-8"), url)
        return _response(self.annotate_status, self.drawn_bytes, url)

    @override
    def close(self) -> None:
        return


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _response(status_code: int, content: bytes, url: str) -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response.url = url
    response._content = content
    return response


def _backend(service: ScriptedService) -> HordeImageUtilitiesBackend:
    session = requests.Session()
    session.mount("http://", service)
    return HordeImageUtilitiesBackend(service_url=SERVICE_URL, session=session)


def test_a_control_map_is_drawn_by_the_service_and_published_as_the_picture_it_returned() -> None:
    """Assert the detector's PNG comes back as the entry's result, drawn at the source's own size."""
    service = ScriptedService()
    backend = _backend(service)
    backend.prepare()

    outcome = backend.apply(
        form="annotation",
        control_type=CONTROL_TYPE,
        source_image=Image.new("RGB", (1024, 768), (30, 40, 50)),
    )

    assert outcome.is_available
    assert outcome.image is not None
    assert outcome.model_name_used == CONTROL_TYPE
    assert f"/annotators/{CONTROL_TYPE}" in service.requested_paths[-1]
    assert b'name="resolution"' in service.annotate_bodies[0]
    assert b"1024" in service.annotate_bodies[0], "the detection runs at the source's longer side"


def test_the_sidecar_records_the_service_that_drew_the_map_and_the_version_it_reported() -> None:
    """Assert a control map can be traced to the build of the service that produced it."""
    backend = _backend(ScriptedService())
    backend.prepare()

    assert backend.name == "horde-image-utilities"
    assert backend.service_host == "127.0.0.1:7860"
    assert backend.service_version == SERVICE_VERSION


def test_a_detector_the_service_says_it_cannot_run_is_a_gap_carrying_its_own_words() -> None:
    """Assert an unavailable detector ends the entry rather than the run, and says why."""
    backend = _backend(ScriptedService())
    backend.prepare()

    outcome = backend.apply(form="annotation", control_type="mediapipe_face", source_image=Image.new("RGB", (64, 64)))

    assert not outcome.is_available
    assert outcome.unavailable_reason is not None
    assert "mediapipe" in outcome.unavailable_reason


def test_a_detector_the_service_does_not_register_is_a_gap_naming_what_it_does_register() -> None:
    """Assert a detector the SDK names and the service has never heard of is published as a gap."""
    backend = _backend(ScriptedService())
    backend.prepare()

    outcome = backend.apply(form="annotation", control_type="teed", source_image=Image.new("RGB", (64, 64)))

    assert not outcome.is_available
    assert outcome.unavailable_reason is not None
    assert CONTROL_TYPE in outcome.unavailable_reason


@pytest.mark.parametrize(
    ("status_code", "detail"),
    [
        (HTTPStatus.CONFLICT, "weights for 'depth' are missing and downloads are disabled"),
        (HTTPStatus.NOT_IMPLEMENTED, "the 'mediapipe_face' detector is registered but not implemented"),
    ],
)
def test_a_refused_detection_is_published_as_the_reason_the_service_gave(status_code: int, detail: str) -> None:
    """Assert the service's own sentence reaches the manifest rather than a status code alone."""
    service = ScriptedService(annotate_status=status_code, annotate_body={"detail": detail})
    backend = _backend(service)
    backend.prepare()

    outcome = backend.apply(form="annotation", control_type=CONTROL_TYPE, source_image=Image.new("RGB", (64, 64)))

    assert not outcome.is_available
    assert outcome.unavailable_reason is not None
    assert detail in outcome.unavailable_reason


def test_a_form_that_is_not_a_control_map_is_left_to_the_backends_that_serve_it() -> None:
    """Assert an upscaler asked of the control-map service is a gap rather than an approximation."""
    backend = _backend(ScriptedService())
    backend.prepare()

    outcome = backend.apply(form="RealESRGAN_x4plus", control_type=None, source_image=Image.new("RGB", (64, 64)))

    assert not outcome.is_available
    assert outcome.unavailable_reason is not None
    assert "control map" in outcome.unavailable_reason


def test_a_service_that_cannot_be_reached_ends_the_run_rather_than_publishing_a_gap() -> None:
    """Assert an unreachable service is an error, because it says nothing about what it serves.

    Publishing that as a gap would put a claim about the horde's control maps into the manifest
    that only this machine's network supports.
    """
    backend = HordeImageUtilitiesBackend(service_url="http://127.0.0.1:1")

    with pytest.raises(RuntimeError, match="did not report its detectors"):
        backend.prepare()

    backend.close()


def test_the_service_url_comes_from_the_flag_then_the_environment_then_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Assert a run can be pointed at another service without editing anything."""
    monkeypatch.delenv(SERVICE_URL_ENVIRONMENT_VARIABLE, raising=False)
    assert resolve_service_url(requested=None) == DEFAULT_SERVICE_URL

    monkeypatch.setenv(SERVICE_URL_ENVIRONMENT_VARIABLE, "http://gpu-box:7860/")
    assert resolve_service_url(requested=None) == "http://gpu-box:7860"
    assert resolve_service_url(requested="http://elsewhere:9000") == "http://elsewhere:9000"
