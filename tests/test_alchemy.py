# SPDX-FileCopyrightText: 2026 Tazlin
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Check the alchemy campaign: form coverage, the manifest's shape, text results and the strip."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, override

import pytest
import requests
from PIL import Image

from param_sweeps.alchemy import (
    ANNOTATION_FORM,
    MANIFEST_FILENAME,
    RESULT_IMAGE_FILENAME,
    RESULT_JSON_FILENAME,
    RESULT_TEXT_FILENAME,
    AlchemyBackend,
    AlchemyOutcome,
    AlchemySpec,
    AlchemyUseCase,
    LiveHordeAlchemyBackend,
    alchemy_root,
    build_manifest,
    classify_form,
    known_alchemy_forms,
    known_annotation_control_types,
    load_alchemy_spec,
    render_form_answer,
    run_alchemy,
    upscaler_form_strip,
)
from param_sweeps.cli import build_parser
from param_sweeps.control_maps import draw_control_maps
from param_sweeps.horde_client import HordeApiClient
from param_sweeps.outputs import WebpEncoding
from param_sweeps.spec import SweepParameter, SweepSpec, load_spec

TOOL_ROOT = Path(__file__).resolve().parents[1]
ALCHEMY_SPEC_PATH = TOOL_ROOT / "alchemy_spec.json"
SWEEP_SPEC_PATH = TOOL_ROOT / "sweep_spec.json"

UPSCALER_FORM = "RealESRGAN_x4plus"
"""An upscaler the strip declares a stop for, which is what makes it the one worth asserting on."""

TEXT_FORM = "caption"
"""A describe-family form, whose result is words rather than pixels."""

CUT_OUT_FORM = "strip_background"

CONTROL_TYPE = "canny"
"""One control map, which is the whole result for the form that all thirty-three of them share."""

REFUSED_FORM = "palette"
"""A form to script a refusal for, standing in for whatever the deployed API's schema lacks."""

PICTURE_ANSWERING_FORM = "vectorize"
"""A form whose result is a picture, which the API returns as a URL rather than as bytes."""


@pytest.fixture(autouse=True)
def fast_webp_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Encode with the fastest WebP method while these tests run.

    The scripted campaigns write dozens of results enlarged from the real 1024px anchor, and the
    library's slowest-method encoder turns each into seconds. Nothing here asserts on encoder
    quality, only on what is written where.
    """
    monkeypatch.setattr(WebpEncoding, "METHOD", 0)


@pytest.fixture(scope="module")
def alchemy_spec() -> AlchemySpec:
    """Return the checked-in alchemy specification."""
    return load_alchemy_spec(ALCHEMY_SPEC_PATH)


@pytest.fixture(scope="module")
def sweep_spec() -> SweepSpec:
    """Return the checked-in sweep specification."""
    return load_spec(SWEEP_SPEC_PATH)


class FakeAlchemyBackend(AlchemyBackend):
    """A backend that returns scripted results, so the campaign can be driven without a GPU.

    Every entry named in ``texts`` returns words, every entry named in ``unavailable`` returns a
    reason, and every other entry returns a small image. An entry is named by its
    control type when it has one, so the control maps can be scripted apart from each other.
    """

    def __init__(
        self,
        *,
        texts: dict[str, str] | None = None,
        svgs: dict[str, str] | None = None,
        unavailable: dict[str, str] | None = None,
    ) -> None:
        """Script the backend's results."""
        self.texts = texts or {}
        self.svgs = svgs or {}
        self.unavailable = unavailable or {}
        self.applied_forms: list[str] = []
        self.prepared = False
        self.closed = False

    @property
    @override
    def name(self) -> str:
        return "fake"

    @override
    def prepare(self) -> None:
        self.prepared = True

    @override
    def apply(self, *, form: str, control_type: str | None, source_image: Image.Image) -> AlchemyOutcome:
        applied = control_type if control_type is not None else form
        self.applied_forms.append(applied)
        if applied in self.unavailable:
            return AlchemyOutcome(unavailable_reason=self.unavailable[applied])
        if applied in self.texts:
            return AlchemyOutcome(
                text=self.texts[applied],
                structured={applied: self.texts[applied]},
                model_name_used=f"{applied}-model",
            )
        if applied in self.svgs:
            return AlchemyOutcome(svg=self.svgs[applied])

        enlarged = source_image.resize((source_image.width * 2, source_image.height * 2))
        return AlchemyOutcome(image=enlarged, model_name_used=applied)

    @override
    def close(self) -> None:
        self.closed = True


def test_every_known_form_is_placed_in_exactly_one_use_case(alchemy_spec: AlchemySpec) -> None:
    """Assert the specification covers the shared vocabulary once each and adds nothing to it.

    The loader enforces this, so reaching this assertion at all means the file is complete. The
    counts are asserted separately because a form silently moving between use cases would still
    load and would quietly change which page a reader finds it on. The ``annotation`` form is held
    to the same rule one level down, by the control-map coverage test.
    """
    placed = [entry.form for entry in alchemy_spec.entries if entry.form != ANNOTATION_FORM]
    assert sorted(placed) == sorted(form for form in known_alchemy_forms() if form != ANNOTATION_FORM)
    assert len(placed) == len(set(placed))
    assert ANNOTATION_FORM in {entry.form for entry in alchemy_spec.entries}

    for entry in alchemy_spec.entries:
        assert entry.use_case in set(AlchemyUseCase)


def test_a_form_the_vocabulary_names_and_the_specification_omits_fails_to_load() -> None:
    """Assert an incomplete specification is a read error rather than a quiet gap in the evidence."""
    complete = json.loads(ALCHEMY_SPEC_PATH.read_text(encoding="utf-8"))
    complete["entries"] = complete["entries"][1:]

    with pytest.raises(ValueError, match="not placed in any use case"):
        AlchemySpec.model_validate(complete)


def test_a_form_placed_twice_fails_to_load() -> None:
    """Assert a form cannot be placed on two pages."""
    duplicated = json.loads(ALCHEMY_SPEC_PATH.read_text(encoding="utf-8"))
    duplicated["entries"].append(duplicated["entries"][0])

    with pytest.raises(ValueError, match="placed more than once"):
        AlchemySpec.model_validate(duplicated)


def test_forms_are_classified_by_the_vocabulary_rather_than_by_name() -> None:
    """Assert the execution kind comes from the SDK's own membership tests."""
    assert classify_form(UPSCALER_FORM) is not classify_form(CUT_OUT_FORM)
    assert classify_form("GFPGAN") is not classify_form(UPSCALER_FORM)
    assert str(classify_form(TEXT_FORM)) == "unsupported"


def test_the_upscaler_strip_holds_only_forms_the_manifest_also_lists(
    sweep_spec: SweepSpec,
    alchemy_spec: AlchemySpec,
) -> None:
    """Assert every stop is an upscaler the manifest places, and that the rest stay disclosed.

    The strip is a scrubber that the page hides unless it is whole, so it includes the upscalers
    that have weights behind them and no others. The upscalers that the vocabulary names and that
    no weights back are not dropped: they stay in the specification, which is what puts them in the manifest
    as entries that say they could not be run.
    """
    strip = upscaler_form_strip(sweep_spec=sweep_spec)
    assert strip.parameter is SweepParameter.UPSCALER_FORM
    assert strip.source_image is not None

    stop_forms = {str(job.stop_value) for job in strip.jobs}
    upscaler_entries = {
        entry.form
        for entry in alchemy_spec.entries
        if entry.use_case is AlchemyUseCase.ENLARGE and str(classify_form(entry.form)) == "upscale"
    }
    assert stop_forms <= upscaler_entries
    assert stop_forms, "the strip has to compare something"
    assert upscaler_entries - stop_forms, "an upscaler off the strip is still owed a manifest entry"

    for job in strip.jobs:
        assert job.stop_label == job.stop_label.lower()
        assert "." not in job.stop_label

    sources = {entry.source for entry in alchemy_spec.entries if entry.form in stop_forms}
    assert sources == {strip.source_image}, "the strip and its manifest entries must share one source"


def test_a_run_writes_the_manifest_the_frontpage_reads(
    tmp_path: Path,
    alchemy_spec: AlchemySpec,
    sweep_spec: SweepSpec,
) -> None:
    """Assert the manifest lists every entry, its shared source, and the right kind of result."""
    backend = FakeAlchemyBackend(
        texts={TEXT_FORM: "a woman on the deck of an airship at sunset"},
        unavailable={"4xNomos8kSC": "not in the model reference"},
    )
    results = run_alchemy(
        spec=alchemy_spec,
        sweep_spec=sweep_spec,
        backend=backend,
        output_root=tmp_path,
    )

    assert backend.prepared and backend.closed
    assert len(results) == len(alchemy_spec.entries)

    manifest = json.loads((alchemy_root(tmp_path) / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["generated_against"] == alchemy_spec.generated_against
    assert [entry["form"] for entry in manifest["entries"]] == [entry.form for entry in alchemy_spec.entries]

    by_form = {entry["form"]: entry for entry in manifest["entries"]}

    caption_entry = by_form[TEXT_FORM]
    assert caption_entry["files"]["result_text"].endswith(RESULT_TEXT_FILENAME)
    assert "result" not in caption_entry["files"], "a text answer must not claim an image result"
    text_path = alchemy_root(tmp_path) / caption_entry["files"]["result_text"]
    assert text_path.read_text(encoding="utf-8").strip() == "a woman on the deck of an airship at sunset"
    assert caption_entry["files"]["result_json"].endswith(RESULT_JSON_FILENAME)
    json_path = alchemy_root(tmp_path) / caption_entry["files"]["result_json"]
    assert json.loads(json_path.read_text(encoding="utf-8")) == {
        TEXT_FORM: "a woman on the deck of an airship at sunset",
    }

    upscale_entry = by_form[UPSCALER_FORM]
    assert upscale_entry["files"]["result"].endswith(RESULT_IMAGE_FILENAME)
    assert "result_text" not in upscale_entry["files"]
    assert "result_json" not in upscale_entry["files"]
    assert upscale_entry["model"] == UPSCALER_FORM
    assert (alchemy_root(tmp_path) / upscale_entry["files"]["result"]).is_file()

    unavailable_entry = by_form["4xNomos8kSC"]
    assert "result" not in unavailable_entry["files"]
    assert "result_text" not in unavailable_entry["files"]
    assert "not in the model reference" in unavailable_entry["notes"]
    assert (alchemy_root(tmp_path) / unavailable_entry["files"]["source"]).is_file()


def test_the_shared_sources_are_published_once_and_pointed_at_by_every_entry_using_them(
    tmp_path: Path,
    alchemy_spec: AlchemySpec,
    sweep_spec: SweepSpec,
) -> None:
    """Assert forms in one use case compare against the same published file rather than copies."""
    run_alchemy(spec=alchemy_spec, sweep_spec=sweep_spec, backend=FakeAlchemyBackend(), output_root=tmp_path)

    manifest = json.loads((alchemy_root(tmp_path) / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    sources_by_use_case: dict[str, set[str]] = {}
    for entry in manifest["entries"]:
        sources_by_use_case.setdefault(entry["use_case"], set()).add(entry["files"]["source"])

    for use_case, sources in sources_by_use_case.items():
        assert len(sources) == 1, f"{use_case} must run every form on one source"


def test_a_run_publishes_the_upscaler_strip_into_the_sweep_library(
    tmp_path: Path,
    alchemy_spec: AlchemySpec,
    sweep_spec: SweepSpec,
) -> None:
    """Assert each upscaler's result is written at the stop that the sweep specification declares for it."""
    run_alchemy(spec=alchemy_spec, sweep_spec=sweep_spec, backend=FakeAlchemyBackend(), output_root=tmp_path)

    strip = upscaler_form_strip(sweep_spec=sweep_spec)
    for job in strip.jobs:
        assert (tmp_path / job.relative_image_path).is_file(), job.relative_image_path
        assert (tmp_path / job.relative_sidecar_path).is_file()

    index = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    indexed = [entry for entry in index["strips"] if entry["parameter"] == str(SweepParameter.UPSCALER_FORM)]
    assert len(indexed) == 1
    assert [stop["value"] for stop in indexed[0]["stops"]] == [str(job.stop_value) for job in strip.jobs]
    assert indexed[0]["complete"] is True


def test_an_unavailable_form_publishes_no_strip_stop(
    tmp_path: Path,
    alchemy_spec: AlchemySpec,
    sweep_spec: SweepSpec,
) -> None:
    """Assert a form the backend could not run leaves its stop empty and the strip incomplete.

    The strip is allowed to be short. It is not allowed to be short and say it is whole.
    """
    missing_form = "NMKD_Siax"
    run_alchemy(
        spec=alchemy_spec,
        sweep_spec=sweep_spec,
        backend=FakeAlchemyBackend(unavailable={missing_form: "no weights here"}),
        output_root=tmp_path,
    )

    strip = upscaler_form_strip(sweep_spec=sweep_spec)
    absent = [job for job in strip.jobs if str(job.stop_value) == missing_form]
    assert len(absent) == 1
    assert not (tmp_path / absent[0].relative_image_path).is_file()

    index = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    indexed = next(entry for entry in index["strips"] if entry["parameter"] == str(SweepParameter.UPSCALER_FORM))
    assert indexed["complete"] is False
    assert missing_form not in [stop["value"] for stop in indexed["stops"]]


def test_a_second_run_reuses_what_is_already_on_disk(
    tmp_path: Path,
    alchemy_spec: AlchemySpec,
    sweep_spec: SweepSpec,
) -> None:
    """Assert an entry that already has a result is left alone, and ``--force`` applies it again."""
    run_alchemy(spec=alchemy_spec, sweep_spec=sweep_spec, backend=FakeAlchemyBackend(), output_root=tmp_path)

    second_backend = FakeAlchemyBackend()
    reused = run_alchemy(
        spec=alchemy_spec,
        sweep_spec=sweep_spec,
        backend=second_backend,
        output_root=tmp_path,
        forms=(UPSCALER_FORM,),
    )
    assert second_backend.applied_forms == []
    assert [result.reused for result in reused] == [True]

    forced_backend = FakeAlchemyBackend()
    run_alchemy(
        spec=alchemy_spec,
        sweep_spec=sweep_spec,
        backend=forced_backend,
        output_root=tmp_path,
        forms=(UPSCALER_FORM,),
        force=True,
    )
    assert forced_backend.applied_forms == [UPSCALER_FORM]


def test_a_form_that_stops_being_available_loses_the_result_it_used_to_have(
    tmp_path: Path,
    alchemy_spec: AlchemySpec,
    sweep_spec: SweepSpec,
) -> None:
    """Assert a stale result cannot outlive the availability that produced it.

    A manifest built from the files on disk would otherwise list a result beside a note saying
    the form could not be run, which is the one contradiction that the gap reporting is meant to
    prevent.
    """
    run_alchemy(spec=alchemy_spec, sweep_spec=sweep_spec, backend=FakeAlchemyBackend(), output_root=tmp_path)
    run_alchemy(
        spec=alchemy_spec,
        sweep_spec=sweep_spec,
        backend=FakeAlchemyBackend(unavailable={CUT_OUT_FORM: "rembg is not installed"}),
        output_root=tmp_path,
        forms=(CUT_OUT_FORM,),
        force=True,
    )

    manifest = build_manifest(spec=alchemy_spec, output_root=tmp_path)
    entry = next(candidate for candidate in manifest.entries if candidate.form == CUT_OUT_FORM)
    assert entry.files.result is None
    assert entry.files.result_text is None
    assert entry.notes is not None
    assert "rembg is not installed" in entry.notes


REFUSAL_BODY = {
    "errors": {"forms.0.name": "'palette' is not one of ['caption', 'interrogation', 'nsfw']"},
    "message": "Input payload validation failed",
}
"""The deployed API's response to a form that its request schema does not include."""

SCRIPTED_REQUEST_ID = "0f8b1f16-0000-4000-8000-000000000000"


class ScriptedTransport(requests.adapters.BaseAdapter):
    """Responds to the client from a script, so the whole submit-and-poll path runs without a network.

    Mounted on a real session and handed to a real client, which is what makes this a test of the
    transport rather than of a stand-in for it.
    """

    def __init__(self, *, submit_status: int, submit_body: dict[str, Any], status_body: dict[str, Any]) -> None:
        """Script the responses to the requests that the client can make."""
        self.submit_status = submit_status
        self.submit_body = submit_body
        self.status_body = status_body
        self.image_bytes = _webp_bytes(Image.new("RGB", (8, 8), (10, 20, 30)))
        self.submitted_bodies: list[dict[str, Any]] = []

    @override
    def send(self, request: requests.PreparedRequest, *args: Any, **kwargs: Any) -> requests.Response:
        url = request.url or ""
        if url.endswith("/interrogate/async"):
            self.submitted_bodies.append(json.loads(request.body or b"{}"))
            return _scripted_response(status_code=self.submit_status, content=_encoded(self.submit_body), url=url)
        if "/interrogate/status/" in url:
            return _scripted_response(status_code=200, content=_encoded(self.status_body), url=url)
        return _scripted_response(status_code=200, content=self.image_bytes, url=url)

    @override
    def close(self) -> None:
        return


def _encoded(body: dict[str, Any]) -> bytes:
    return json.dumps(body).encode("utf-8")


def _webp_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="WEBP")
    return buffer.getvalue()


def _scripted_response(*, status_code: int, content: bytes, url: str) -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response.url = url
    response._content = content
    return response


def _live_backend(transport: ScriptedTransport) -> LiveHordeAlchemyBackend:
    session = requests.Session()
    session.mount("https://", transport)
    return LiveHordeAlchemyBackend(client=HordeApiClient(api_key="not-a-real-key", session=session))


def test_the_live_backend_publishes_what_the_horde_said() -> None:
    """Assert a text result comes back as text, and that the form and the image were submitted."""
    said = "a woman on the deck of an airship at sunset"
    transport = ScriptedTransport(
        submit_status=202,
        submit_body={"id": SCRIPTED_REQUEST_ID},
        status_body={
            "state": "done",
            "forms": [{"form": TEXT_FORM, "state": "done", "result": {TEXT_FORM: said}}],
        },
    )
    backend = _live_backend(transport)
    backend.prepare()

    outcome = backend.apply(form=TEXT_FORM, control_type=None, source_image=Image.new("RGB", (16, 16), (200, 100, 50)))

    assert outcome.is_available
    assert outcome.text == said
    assert outcome.structured == {TEXT_FORM: said}, "the answer is published as it arrived as well as as text"
    assert outcome.image is None
    assert outcome.model_name_used is None, "the response names no weights, so the manifest claims none"

    submitted = transport.submitted_bodies[0]
    assert submitted["forms"] == [{"name": TEXT_FORM}]
    assert submitted["source_image"], "the image travels as the base64 the API's source_image takes"


def test_a_form_the_api_refuses_is_published_as_a_gap_carrying_the_reason_it_gave() -> None:
    """Assert a form the live alchemist does not serve ends the entry rather than the run.

    The reason is the API's own, listing what it does accept, because a manifest that says only
    "unavailable" tells a reader nothing about whether the horde ever offered the form.
    """
    transport = ScriptedTransport(submit_status=400, submit_body=REFUSAL_BODY, status_body={})
    backend = _live_backend(transport)

    outcome = backend.apply(form=REFUSED_FORM, control_type=None, source_image=Image.new("RGB", (16, 16)))

    assert not outcome.is_available
    assert outcome.unavailable_reason is not None
    assert "is not one of" in outcome.unavailable_reason
    assert REFUSED_FORM in outcome.unavailable_reason


def test_an_answer_that_is_a_picture_is_downloaded_rather_than_written_down() -> None:
    """Assert a form that responds with an image URL is written as a result image, not as its URL in text."""
    transport = ScriptedTransport(
        submit_status=202,
        submit_body={"id": SCRIPTED_REQUEST_ID},
        status_body={
            "state": "done",
            "forms": [
                {
                    "form": PICTURE_ANSWERING_FORM,
                    "state": "done",
                    "result": {PICTURE_ANSWERING_FORM: "https://example.invalid/vectorized.webp"},
                },
            ],
        },
    )
    backend = _live_backend(transport)

    outcome = backend.apply(form=PICTURE_ANSWERING_FORM, control_type=None, source_image=Image.new("RGB", (16, 16)))

    assert outcome.is_available
    assert outcome.image is not None
    assert outcome.text is None


def test_an_answer_with_nothing_in_it_is_a_gap_rather_than_an_empty_result() -> None:
    """Assert a form that finishes without a result does not publish a result and reports why."""
    transport = ScriptedTransport(
        submit_status=202,
        submit_body={"id": SCRIPTED_REQUEST_ID},
        status_body={"state": "done", "forms": [{"form": TEXT_FORM, "state": "done", "result": {}}]},
    )

    outcome = _live_backend(transport).apply(
        form=TEXT_FORM, control_type=None, source_image=Image.new("RGB", (16, 16))
    )

    assert not outcome.is_available
    assert outcome.unavailable_reason is not None
    assert TEXT_FORM in outcome.unavailable_reason


def test_a_sentence_answer_is_published_as_the_sentence() -> None:
    """Assert the captioner's result is not dressed up with the field name that it arrived under."""
    said = "a woman standing on a wooden deck next to a metal railing"
    assert render_form_answer(form="caption", result={"caption": said}) == said


def test_a_judgement_answer_keeps_the_spelling_the_horde_used() -> None:
    """Assert a boolean reads as the JSON it arrived as, and a score keeps every digit that it had."""
    assert render_form_answer(form="nsfw", result={"nsfw": False}) == "nsfw: false"
    assert render_form_answer(form="aesthetic", result={"aesthetic": 6.3933}) == "aesthetic: 6.3933"


def test_a_data_answer_is_published_a_field_to_a_line() -> None:
    """Assert the technical description reads as its own fields rather than as a JSON blob."""
    rendered = render_form_answer(
        form="describe",
        result={"describe": {"width": 1024, "format": "WEBP", "has_alpha": False}},
    )

    assert rendered.splitlines() == ["width: 1024", "format: WEBP", "has_alpha: false"]


def test_a_listed_answer_names_each_item_and_how_strongly_it_holds() -> None:
    """Assert the two listed shapes, CLIP's phrases and the palette's colours, both read plainly."""
    interrogation = render_form_answer(
        form="interrogation",
        result={
            "interrogation": {
                "techniques": [{"text": "Anamorphic", "confidence": 54.0709}],
                "sites": [{"text": "behance", "confidence": 59.0001}],
            },
        },
    )
    assert interrogation.splitlines() == ["techniques: Anamorphic (54.0709)", "sites: behance (59.0001)"]

    palette = render_form_answer(
        form="palette",
        result={"palette": {"colors": [{"hex": "#0d1c1e", "proportion": 0.2389}]}},
    )
    assert palette == "colors: #0d1c1e (0.2389)"


def test_a_vector_answer_publishes_as_a_drawable_result(
    alchemy_spec: AlchemySpec,
    sweep_spec: SweepSpec,
    tmp_path: Path,
) -> None:
    """Assert an SVG result is written as the entry's result rather than as a transcript.

    The vectorizer returns the picture itself, as markup. A page handed that markup as
    result_text would print a wall of path data. Filed as the result, a browser draws it.
    """
    markup = '<svg xmlns="http://www.w3.org/2000/svg" width="8" height="8"><rect width="8" height="8"/></svg>'
    run_alchemy(
        spec=alchemy_spec,
        sweep_spec=sweep_spec,
        backend=FakeAlchemyBackend(svgs={"vectorize": markup}),
        output_root=tmp_path,
    )

    manifest = json.loads((tmp_path / "alchemy" / "manifest.json").read_text(encoding="utf-8"))
    entry = next(e for e in manifest["entries"] if e["form"] == "vectorize")
    assert entry["files"]["result"].endswith("result.svg")
    assert "result_text" not in entry["files"]
    svg_path = tmp_path / "alchemy" / entry["files"]["result"]
    assert svg_path.read_text(encoding="utf-8").strip() == markup


def test_every_control_map_the_vocabulary_names_is_placed_exactly_once(alchemy_spec: AlchemySpec) -> None:
    """Assert the control-map page shows every detector the horde knows, on one shared source."""
    placed = [entry.control_type for entry in alchemy_spec.entries if entry.control_type is not None]

    assert sorted(placed) == sorted(known_annotation_control_types())
    assert len(placed) == len(set(placed))

    control_map_entries = [entry for entry in alchemy_spec.entries if entry.use_case is AlchemyUseCase.CONTROL_MAP]
    assert [entry.control_type for entry in control_map_entries] == placed
    assert {entry.form for entry in control_map_entries} == {ANNOTATION_FORM}
    assert len({entry.source for entry in control_map_entries}) == 1


def test_a_control_map_the_vocabulary_names_and_the_specification_omits_fails_to_load() -> None:
    """Assert a detector the SDK gains cannot quietly go missing from the published evidence."""
    incomplete = json.loads(ALCHEMY_SPEC_PATH.read_text(encoding="utf-8"))
    incomplete["entries"] = [entry for entry in incomplete["entries"] if entry.get("control_type") != CONTROL_TYPE]

    with pytest.raises(ValueError, match="control types are not placed"):
        AlchemySpec.model_validate(incomplete)


def test_a_control_map_the_vocabulary_does_not_name_fails_to_load() -> None:
    """Assert the specification cannot invent a detector the horde has no name for."""
    invented = json.loads(ALCHEMY_SPEC_PATH.read_text(encoding="utf-8"))
    placed = next(entry for entry in invented["entries"] if entry.get("control_type") == CONTROL_TYPE)
    invented["entries"].append({**placed, "control_type": "sobel", "title": "sobel"})

    with pytest.raises(ValueError, match="are not known to horde_sdk"):
        AlchemySpec.model_validate(invented)


def test_a_control_type_on_a_form_that_is_not_the_annotation_form_fails_to_load() -> None:
    """Assert a control type belongs to the one form whose result identifies which detector ran."""
    misplaced = json.loads(ALCHEMY_SPEC_PATH.read_text(encoding="utf-8"))
    for entry in misplaced["entries"]:
        if entry["form"] == UPSCALER_FORM:
            entry["control_type"] = CONTROL_TYPE

    with pytest.raises(ValueError, match="disagree"):
        AlchemySpec.model_validate(misplaced)


def test_the_control_maps_are_told_apart_by_their_detector_in_the_tree_and_in_the_manifest(
    tmp_path: Path,
    alchemy_spec: AlchemySpec,
    sweep_spec: SweepSpec,
) -> None:
    """Assert each detector gets its own directory and its own manifest entry naming it.

    Every control map is the same form, so the form cannot tell the entries apart. The detector
    is what distinguishes them, and a consumer that cannot read which one drew a map cannot
    label it.
    """
    run_alchemy(
        spec=alchemy_spec,
        sweep_spec=sweep_spec,
        backend=FakeAlchemyBackend(unavailable={"openpose": "no weights for the pose detector here"}),
        output_root=tmp_path,
        forms=(ANNOTATION_FORM,),
    )

    manifest = json.loads((alchemy_root(tmp_path) / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    control_maps = [entry for entry in manifest["entries"] if entry["use_case"] == str(AlchemyUseCase.CONTROL_MAP)]
    drawn_by = [entry["control_type"] for entry in control_maps]
    assert sorted(drawn_by) == sorted(known_annotation_control_types())
    placed = [entry.control_type for entry in alchemy_spec.entries if entry.control_type is not None]
    assert placed == drawn_by, "the manifest lists the detectors in the order the specification groups them"

    drawn = next(entry for entry in control_maps if entry["control_type"] == CONTROL_TYPE)
    assert drawn["files"]["result"] == f"{AlchemyUseCase.CONTROL_MAP}/{CONTROL_TYPE}/{RESULT_IMAGE_FILENAME}"
    assert (alchemy_root(tmp_path) / drawn["files"]["result"]).is_file()

    missing = next(entry for entry in control_maps if entry["control_type"] == "openpose")
    assert "result" not in missing["files"]
    assert "no weights for the pose detector here" in missing["notes"]


def test_a_run_can_be_restricted_to_one_detector(
    tmp_path: Path,
    alchemy_spec: AlchemySpec,
    sweep_spec: SweepSpec,
) -> None:
    """Assert naming a control type runs that detector and nothing else."""
    backend = FakeAlchemyBackend()
    results = run_alchemy(
        spec=alchemy_spec,
        sweep_spec=sweep_spec,
        backend=backend,
        output_root=tmp_path,
        control_types=(CONTROL_TYPE,),
    )

    assert backend.applied_forms == [CONTROL_TYPE]
    assert [result.control_type for result in results] == [CONTROL_TYPE]

    parsed = build_parser().parse_args(["alchemy", "--control-type", CONTROL_TYPE, "--backend", "hiu"])
    assert parsed.control_type == [CONTROL_TYPE]
    assert parsed.hiu_url is None


def test_a_strength_strip_draws_one_map_and_a_strip_with_no_detector_draws_none(
    tmp_path: Path,
    sweep_spec: SweepSpec,
) -> None:
    """Assert a strip held to one detector publishes one map, beside its source, and only one.

    Every stop of a strength strip is held to the same map, so drawing it per stop would request the
    same picture from the service a dozen times and publish a dozen copies of it. A strip that
    conditions on nothing has nothing for a detector to draw.
    """
    backend = FakeAlchemyBackend()
    held_strips = [strip for strip in sweep_spec.expand_strips() if strip.relative_control_map_path is not None]

    outcomes = draw_control_maps(
        spec=sweep_spec,
        output_root=tmp_path,
        backend=backend,
        parameters=(SweepParameter.CONTROL_STRENGTH,),
    )

    assert [outcome.relative_path for outcome in outcomes] == [
        strip.relative_control_map_path for strip in held_strips
    ]
    assert all(outcome.drawn and not outcome.skipped for outcome in outcomes)
    assert backend.applied_forms == [strip.jobs[0].control_type for strip in held_strips]
    assert len(list(tmp_path.rglob("*.webp"))) == len(held_strips), "one map per strip and nothing else"

    first = held_strips[0]
    published = first.relative_control_map_path
    assert published is not None
    assert (tmp_path / published).is_file()
    sidecar = json.loads((tmp_path / published).with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["control_type"] == first.jobs[0].control_type
    assert sidecar["source"] == first.source_image
    assert sidecar["render"] is None, "the map stands for the whole strip rather than for one render"

    reused = draw_control_maps(
        spec=sweep_spec,
        output_root=tmp_path,
        backend=backend,
        parameters=(SweepParameter.CONTROL_STRENGTH,),
    )
    assert all(outcome.skipped for outcome in reused)

    redrawn = draw_control_maps(
        spec=sweep_spec,
        output_root=tmp_path,
        backend=backend,
        parameters=(SweepParameter.CONTROL_STRENGTH,),
        force=True,
    )
    assert all(outcome.drawn and not outcome.skipped for outcome in redrawn)

    assert (
        draw_control_maps(
            spec=sweep_spec,
            output_root=tmp_path,
            backend=backend,
            parameters=(SweepParameter.QR_STYLE,),
        )
        == []
    )


def test_a_backend_that_does_not_draw_control_maps_says_where_they_are_drawn() -> None:
    """Assert the live-horde backend points at the service that serves the form it cannot."""
    outcome = _live_backend(ScriptedTransport(submit_status=202, submit_body={}, status_body={})).apply(
        form=ANNOTATION_FORM,
        control_type=CONTROL_TYPE,
        source_image=Image.new("RGB", (16, 16)),
    )

    assert not outcome.is_available
    assert outcome.unavailable_reason is not None
    assert "--backend hiu" in outcome.unavailable_reason
