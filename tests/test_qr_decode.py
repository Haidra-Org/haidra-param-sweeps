# SPDX-FileCopyrightText: 2026 Tazlin
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Check the QR decode gate: that it reads a real code, refuses anything else, and gates renders."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import qrcode
from PIL import Image
from qrcode.image.pil import PilImage

from param_sweeps.qr_decode import (
    QrCheck,
    QrReport,
    check_library,
    decode_candidates,
    decode_qr,
    failed_gate_reason,
    payload_in,
)
from param_sweeps.spec import ExtraText, RenderJob, SweepParameter, SweepSpec, load_spec

SPEC_PATH = Path(__file__).resolve().parents[1] / "sweep_spec.json"

PAYLOAD = "https://aihorde.net"


@pytest.fixture(scope="module")
def spec() -> SweepSpec:
    return load_spec(SPEC_PATH)


def _code(payload: str = PAYLOAD) -> Image.Image:
    """Return a plain, machine-drawn QR code: what a scanner is supposed to have no trouble with."""
    drawn: PilImage = qrcode.make(payload, image_factory=PilImage)
    return drawn.get_image().convert("RGB")


def _qr_job(*, batch_size: int = 1) -> RenderJob:
    return RenderJob(
        parameter=SweepParameter.QR_STYLE,
        model_slug="probe-model",
        horde_model_name="Probe",
        scene_id="probe",
        prompt="a probe",
        stop_index=1,
        stop_label="probe",
        stop_value="probe",
        caption="a probe",
        width=1024,
        height=1024,
        steps=25,
        cfg_scale=7.5,
        sampler_name="k_euler",
        clip_skip=1,
        hires_fix=False,
        tiling=False,
        karras=False,
        denoising_strength=1.0,
        seed=1,
        workflow="qr_code",
        control_strength=1.3,
        batch_size=batch_size,
        extra_texts=[ExtraText(text=PAYLOAD, reference="qr_code")],
    )


def test_a_real_code_decodes_to_its_payload() -> None:
    assert decode_qr(_code()) == PAYLOAD


def test_a_picture_with_no_code_in_it_decodes_to_nothing() -> None:
    assert decode_qr(Image.new("RGB", (512, 512), "white")) is None


def test_the_payload_is_read_from_either_reference_spelling() -> None:
    assert payload_in([ExtraText(text=PAYLOAD, reference="qr_text")]) == PAYLOAD
    assert payload_in([ExtraText(text="Circle", reference="module_drawer")]) is None


def test_a_render_that_is_not_a_qr_job_is_not_gated() -> None:
    job = _qr_job().model_copy(update={"workflow": None, "extra_texts": []})

    assert decode_candidates(job=job, images=[Image.new("RGB", (64, 64))]) == []
    assert failed_gate_reason(job=job, images=[Image.new("RGB", (64, 64))]) is None


def test_a_single_qr_render_that_does_not_scan_is_rejected() -> None:
    reason = failed_gate_reason(job=_qr_job(), images=[Image.new("RGB", (512, 512), "white")])

    assert reason is not None
    assert PAYLOAD in reason


def test_a_single_qr_render_that_scans_is_published() -> None:
    assert failed_gate_reason(job=_qr_job(), images=[_code()]) is None


def test_a_curation_batch_is_reported_rather_than_rejected() -> None:
    """Assert a batch survives candidates that do not scan, because its pick is still to be made.

    Rejecting the whole job would throw away the candidates a reviewer is meant to choose between,
    and the choice is what decides which image is published; ``check_library`` gates that.
    """
    job = _qr_job(batch_size=2)
    images = [Image.new("RGB", (512, 512), "white"), _code()]

    assert decode_candidates(job=job, images=images) == [None, PAYLOAD]
    assert failed_gate_reason(job=job, images=images) is None


def test_a_check_only_passes_on_the_exact_payload() -> None:
    assert QrCheck(path="a.webp", expected=PAYLOAD, decoded=PAYLOAD).scans
    assert not QrCheck(path="a.webp", expected=PAYLOAD, decoded="https://example.com").scans
    assert not QrCheck(path="a.webp", expected=PAYLOAD, decoded=None).scans
    assert not QrReport(checks=[QrCheck(path="a.webp", expected=PAYLOAD, decoded=None)]).is_intact


def test_checking_an_empty_tree_finds_nothing_to_scan(tmp_path: Path) -> None:
    assert check_library(output_root=tmp_path).checks == []


def test_a_published_flow_is_scanned_against_the_payload_its_request_states(tmp_path: Path) -> None:
    flow_directory = tmp_path / "tutorials" / "qr_code"
    flow_directory.mkdir(parents=True)
    request = {"params": {"workflow": "qr_code", "extra_texts": [{"text": PAYLOAD, "reference": "qr_code"}]}}
    (flow_directory / "request.json").write_text(json.dumps(request), encoding="utf-8")
    _code().save(flow_directory / "result.webp", format="WEBP")

    report = check_library(output_root=tmp_path)

    assert [check.path for check in report.checks] == ["tutorials/qr_code/result.webp"]
    assert report.is_intact


def test_every_qr_stop_in_the_specification_carries_the_payload(spec: SweepSpec) -> None:
    """Assert no QR stop was written without the text it exists to encode."""
    qr_jobs = [job for job in spec.expand_jobs() if job.workflow is not None]

    assert qr_jobs, "the specification is supposed to render QR strips"
    for job in qr_jobs:
        assert payload_in(job.extra_texts) == PAYLOAD, f"{job.relative_image_path} encodes nothing"
