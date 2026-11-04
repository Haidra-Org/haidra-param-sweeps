# SPDX-FileCopyrightText: 2026 Tazlin
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Decode the QR codes that this tool renders, and check that every published one still scans.

A QR render is the one image in the library whose correctness is not a matter of taste. Either a
scanner reads the payload from it, or the picture is only decoration. The check is therefore a
real decode through ``zbar`` rather than a contrast heuristic, run against the image exactly as
it is published: the same pixels that a reader scans with a phone, without upscaling, sharpening
or repeated attempts at different scales. A check that preprocesses the image until it decodes
only tests the preprocessing.

The check runs in two places. A single-image render is decoded when the runner returns it, and
the job fails if it does not scan, so an unreadable code is not written into the library. A
curation batch is not failed that way, because a human picks one of its candidates afterwards.
Every candidate's decode is reported instead, and
[`check_library`][param_sweeps.qr_decode.check_library] checks what was published. It runs from
``param-sweeps qr-check`` once the picks are made.

Critical public members:

- [`decode_qr`][param_sweeps.qr_decode.decode_qr]: the decode itself, text or None.
- [`payload_in`][param_sweeps.qr_decode.payload_in]: the payload in an ``extra_texts`` list.
- [`decode_candidates`][param_sweeps.qr_decode.decode_candidates]: what a job's renders decoded to.
- [`check_library`][param_sweeps.qr_decode.check_library]: decode every published QR image.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import BaseModel, ConfigDict
from pyzbar.pyzbar import ZBarSymbol
from pyzbar.pyzbar import decode as zbar_decode

from param_sweeps.flows import TUTORIALS_DIRECTORY_NAME
from param_sweeps.outputs import INDEX_FILENAME, ImageSidecar, SweepIndex
from param_sweeps.spec import QR_TEXT_REFERENCES, ExtraText, RenderJob, WorkflowName

logger = logging.getLogger(__name__)

FLOW_REQUEST_FILENAME = "request.json"
FLOW_RESULT_FILENAME = "result.webp"

FROZEN_MODEL_CONFIG = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())


class QrCheck(BaseModel):
    """Represents one published image and the scanner's result for it."""

    model_config = FROZEN_MODEL_CONFIG

    path: str
    """The image, relative to the output root."""

    expected: str
    """The payload that the request specified for encoding."""

    decoded: str | None
    """What the scanner read, or None when it did not read anything."""

    @property
    def scans(self) -> bool:
        """Return whether this image decodes to exactly the requested payload."""
        return self.decoded == self.expected


class QrReport(BaseModel):
    """Represents every published QR image that was checked, and the results."""

    model_config = FROZEN_MODEL_CONFIG

    checks: list[QrCheck]

    @property
    def failures(self) -> list[QrCheck]:
        """Return the images that did not decode to their payload."""
        return [check for check in self.checks if not check.scans]

    @property
    def is_intact(self) -> bool:
        """Return whether every checked image scanned."""
        return not self.failures


def decode_qr(image: Image.Image) -> str | None:
    """Return the text that a scanner reads from this image, or None.

    Args:
        image: The image to scan, at its published size.

    Returns:
        The decoded text, or None when no QR symbol was found. A symbol whose bytes are not
        UTF-8 is reported as unreadable, because a payload that this tool can write is text.
    """
    symbols = zbar_decode(image.convert("L"), symbols=[ZBarSymbol.QRCODE])
    if not symbols:
        return None

    try:
        return str(symbols[0].data.decode("utf-8"))
    except UnicodeDecodeError:
        logger.warning("A QR symbol decoded to bytes that are not UTF-8, and every payload here is UTF-8 text")
        return None


def decode_image_file(image_path: Path) -> str | None:
    """Return what a scanner reads from the image at this path, or None."""
    with Image.open(image_path) as opened:
        return decode_qr(opened)


def payload_in(extra_texts: Iterable[ExtraText]) -> str | None:
    """Return the payload that these extra texts specify for the QR workflow to encode, or None."""
    for extra_text in extra_texts:
        if extra_text.reference in QR_TEXT_REFERENCES:
            return extra_text.text
    return None


def decode_candidates(*, job: RenderJob, images: Sequence[Image.Image]) -> list[str | None]:
    """Return what each of a job's rendered candidates decodes to, in render order.

    Args:
        job: The job that produced them.
        images: Its candidates, in the backend's order.

    Returns:
        One entry per candidate: the decoded text, or None. An empty list when the job is not a
        QR render, so a caller can tell "nothing to check" apart from "nothing passed".
    """
    if job.workflow is not WorkflowName.QR_CODE:
        return []

    return [decode_qr(image) for image in images]


def failed_gate_reason(*, job: RenderJob, images: Sequence[Image.Image]) -> str | None:
    """Return why this render must not be published, or None when it may be.

    A single-image QR render is checked here because it is published as it is, so a code that
    does not scan is a failed render. A curation batch is not checked here, because a human has
    not yet chosen which candidate to publish. Its decodes are logged to inform that choice.

    Args:
        job: The job that was rendered.
        images: The images that the runner returned for it.

    Returns:
        A one-line reason that the campaign can report, or None when the render passes.
    """
    decoded = decode_candidates(job=job, images=images)
    if not decoded:
        return None

    expected = payload_in(job.extra_texts)
    scanned = [number for number, text in enumerate(decoded, start=1) if text == expected]
    logger.info(
        "QR decode: job=%s scanned=%s of %s candidate(s)",
        job.relative_image_path,
        len(scanned),
        len(decoded),
    )

    if job.batch_size > 1:
        logger.info("QR candidates that scanned: job=%s candidates=%s", job.relative_image_path, scanned)
        return None

    if scanned:
        return None

    return f"the QR code does not scan: expected {expected!r}, a scanner read {decoded[0]!r}"


def check_library(*, output_root: Path) -> QrReport:
    """Decode every published QR image under this output root, and report each result.

    Both trees are covered, because both publish QR images that a reader scans with a phone: the
    sweep library, whose sidecars state each image's request, and the tutorial flows, whose
    ``request.json`` does the same. Only images in the index and in the tutorial directories count
    as published, so a candidate that a human did not pick is not checked, and neither is a stop
    that is not in the index.

    Args:
        output_root: The root of the output tree.

    Returns:
        One check per published QR image, in the order they were found.
    """
    return QrReport(checks=[*_library_checks(output_root), *_tutorial_checks(output_root)])


def _library_checks(output_root: Path) -> list[QrCheck]:
    """Return a check for every QR image published in the sweep index."""
    index_path = output_root / INDEX_FILENAME
    if not index_path.is_file():
        logger.info("No index to check QR images against: %s", index_path)
        return []

    index = SweepIndex.model_validate_json(index_path.read_text(encoding="utf-8"))
    checks: list[QrCheck] = []

    for strip in index.strips:
        for stop in strip.stops:
            sidecar_path = output_root / stop.sidecar
            if not sidecar_path.is_file():
                continue

            sidecar = ImageSidecar.model_validate_json(sidecar_path.read_text(encoding="utf-8"))
            if sidecar.features.workflow is not WorkflowName.QR_CODE:
                continue

            expected = payload_in(sidecar.features.extra_texts)
            if expected is None:
                continue

            checks.append(_checked(output_root=output_root, relative_path=stop.file, expected=expected))

    return checks


def _tutorial_checks(output_root: Path) -> list[QrCheck]:
    """Return a check for every QR image that a tutorial flow published."""
    tutorials_root = output_root / TUTORIALS_DIRECTORY_NAME
    if not tutorials_root.is_dir():
        return []

    checks: list[QrCheck] = []
    for request_path in sorted(tutorials_root.glob(f"*/{FLOW_REQUEST_FILENAME}")):
        request: dict[str, Any] = json.loads(request_path.read_text(encoding="utf-8"))
        parameters = request.get("params", {})
        if parameters.get("workflow") != str(WorkflowName.QR_CODE):
            continue

        expected = payload_in(ExtraText.model_validate(entry) for entry in parameters.get("extra_texts", []))
        result_path = request_path.parent / FLOW_RESULT_FILENAME
        if expected is None or not result_path.is_file():
            continue

        checks.append(
            _checked(
                output_root=output_root,
                relative_path=result_path.relative_to(output_root).as_posix(),
                expected=expected,
            ),
        )

    return checks


def _checked(*, output_root: Path, relative_path: str, expected: str) -> QrCheck:
    """Return one image decoded and compared against the requested payload."""
    return QrCheck(
        path=relative_path,
        expected=expected,
        decoded=decode_image_file(output_root / relative_path),
    )
