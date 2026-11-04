# SPDX-FileCopyrightText: 2026 Tazlin
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Check the promo gallery: model requirements, the manifest rebuild, refusals and conversations."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, override

import pytest
import requests
from PIL import Image

from param_sweeps.cli import build_parser
from param_sweeps.gallery import (
    CENSORED_NAME,
    CONVERSATION_DIRECTORY,
    MANIFEST_NAME,
    META_NAME,
    TEXT_DIRECTORY,
    TEXT_FORMAT,
    TEXT_FRAME,
    GalleryHorde,
    apply_requirements,
    build_manifest,
    run_gallery,
)

API_ROOT = "https://fake.horde.test/api/v2"
MODEL_REFERENCE_HOST = "https://models.aihorde.net"

FLUX_MODEL = "Flux.1-Schnell fp8 (Compact)"
"""The model whose requirements are the strictest: three to eight steps, cfg 1 to 4, one sampler."""

CLIP_SKIP_MODEL = "Anime Model"
PINNED_CFG_MODEL = "Pinned Model"
PLAIN_MODEL = "Deliberate"

MODEL_REFERENCE: dict[str, dict[str, Any]] = {
    FLUX_MODEL: {
        "requirements": {
            "min_steps": 3,
            "max_steps": 8,
            "min_cfg_scale": 1,
            "max_cfg_scale": 4,
            "samplers": ["k_euler"],
        }
    },
    CLIP_SKIP_MODEL: {"requirements": {"clip_skip": 2, "schedulers": ["simple"]}},
    PINNED_CFG_MODEL: {"requirements": {"cfg_scale": 2.5}},
    PLAIN_MODEL: {},
}

SPEC: dict[str, Any] = {
    "api": API_ROOT,
    "client_agent": "gallery-test:1.0:tests",
    "defaults": {
        "flux": {"width": 1024, "height": 1024, "steps": 25, "cfg_scale": 7, "sampler_name": "k_dpmpp_2m"},
        "sd15": {"width": 512, "height": 768, "steps": 25, "cfg_scale": 7, "sampler_name": "k_euler_a"},
    },
    "families": {"flux": [FLUX_MODEL], "sd15": [PLAIN_MODEL]},
    "images": [
        {
            "id": "fox-snow",
            "prompt": "a red fox asleep in fresh snow",
            "family": "sd15",
            "models": [PLAIN_MODEL],
            "seed": 41001,
            "alchemy": ["caption"],
            "tags": ["everyday", "photo"],
        },
        {
            "id": "airship",
            "prompt": "an airship over a harbour at dawn",
            "family": "flux",
            "models": [FLUX_MODEL],
            "seed": 41002,
            "alchemy": [],
            "tags": ["art", "flux"],
        },
    ],
    "texts": [{"id": "lighthouse-poem", "prompt": "Write a four-line poem about a lighthouse.", "max_length": 80}],
    "conversations": [
        {
            "id": "thread-fox",
            "format": "instruct",
            "turns": ["Name a fox who runs a bakery.", "What goes wrong on opening day?"],
            "max_length": 90,
        },
        {
            "id": "story-market",
            "format": "raw",
            "turns": ["Every stall in the night market sold something impossible. The first sold"],
            "max_length": 120,
        },
    ],
}

REQUEST_ID = "0f8b1f16-0000-4000-8000-000000000000"


class ScriptedTransport(requests.adapters.BaseAdapter):
    """Responds to the whole submit-and-poll path from a script, so a run needs no network.

    Mounted on the session a real ``GalleryHorde`` opened, which is what makes this a test of the
    submitting code rather than of a stand-in for it. Every request finishes on its first poll:
    the waiting is what the live horde does, and nothing here asserts on it.
    """

    def __init__(
        self,
        *,
        served: dict[str, int] | None = None,
        censored: frozenset[str] = frozenset(),
        replies: list[str] | None = None,
    ) -> None:
        """Script what is served, which prompts come back censored, and what the text models return."""
        self.served = served or dict.fromkeys(MODEL_REFERENCE, 1)
        self.censored = censored
        self.replies = list(replies or [])
        self.image_submissions: list[dict[str, Any]] = []
        self.text_submissions: list[dict[str, Any]] = []
        self.alchemy_submissions: list[dict[str, Any]] = []
        self._by_request: dict[str, tuple[str, dict[str, Any]]] = {}
        self._text_replies: dict[str, str] = {}
        self._picture = _png_bytes(Image.new("RGB", (32, 24), (120, 90, 60)))

    @override
    def send(self, request: requests.PreparedRequest, *args: Any, **kwargs: Any) -> requests.Response:
        url = request.url or ""
        if url.startswith(MODEL_REFERENCE_HOST):
            return _response(MODEL_REFERENCE, url)
        if "/status/models" in url:
            return _response([{"name": name, "count": count} for name, count in self.served.items()], url)
        if url.endswith("/generate/async"):
            return self._accept(url, "image", json.loads(request.body or b"{}"), self.image_submissions)
        if url.endswith("/generate/text/async"):
            return self._accept(url, "text", json.loads(request.body or b"{}"), self.text_submissions)
        if url.endswith("/interrogate/async"):
            return self._accept(url, "alchemy", json.loads(request.body or b"{}"), self.alchemy_submissions)
        if "/generate/check/" in url or "/generate/status/" in url:
            return _response(self._image_result(url), url)
        if "/generate/text/status/" in url:
            return _response(self._text_result(url), url)
        if "/interrogate/status/" in url:
            return _response(
                {
                    "state": "done",
                    "forms": [{"form": "caption", "state": "done", "result": {"caption": "a fox in snow"}}],
                },
                url,
            )
        return _response_bytes(self._picture, url)

    @override
    def close(self) -> None:
        return

    def _accept(
        self,
        url: str,
        kind: str,
        body: dict[str, Any],
        recorded: list[dict[str, Any]],
    ) -> requests.Response:
        recorded.append(body)
        request_id = f"{REQUEST_ID[:-2]}{len(self._by_request):02d}"
        self._by_request[request_id] = (kind, body)
        return _response({"id": request_id}, url)

    def _image_result(self, url: str) -> dict[str, Any]:
        _, body = self._by_request[url.rsplit("/", 1)[-1]]
        censored = body["prompt"] in self.censored
        return {
            "done": True,
            "kudos": 4.5,
            "generations": [
                {
                    "img": "https://fake.horde.test/r2/result.png",
                    "model": body["models"][0],
                    "seed": body["params"]["seed"],
                    "censored": censored,
                },
            ],
        }

    def _text_result(self, url: str) -> dict[str, Any]:
        # The text endpoint serves progress and result at one URL, so a request is read twice and
        # the same reply has to come back both times.
        request_id = url.rsplit("/", 1)[-1]
        if request_id not in self._text_replies:
            self._text_replies[request_id] = (
                self.replies.pop(0) if self.replies else f"answer {len(self._text_replies) + 1}"
            )
        reply = self._text_replies[request_id]
        return {
            "done": True,
            "kudos": 1.25,
            "generations": [{"text": reply, "model": "koboldcpp/a-text-model"}],
        }


def _response(payload: Any, url: str) -> requests.Response:
    return _response_bytes(json.dumps(payload).encode("utf-8"), url)


def _response_bytes(content: bytes, url: str) -> requests.Response:
    response = requests.Response()
    response.status_code = 200
    response.url = url
    response._content = content
    return response


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _horde(transport: ScriptedTransport) -> GalleryHorde:
    horde = GalleryHorde(API_ROOT, "not-a-real-key", "gallery-test:1.0:tests")
    horde.session.mount("https://", transport)
    return horde


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch: pytest.MonkeyPatch) -> None:
    """Take the pacing out of a scripted run: every request here finishes on its first poll."""
    monkeypatch.setattr("param_sweeps.gallery.time.sleep", lambda _seconds: None)


def test_flux_requirements_clamp_steps_cfg_and_the_sampler() -> None:
    """Assert a request the model would refuse is brought inside what the reference states."""
    clamped = apply_requirements(
        {"steps": 25, "cfg_scale": 7, "sampler_name": "k_dpmpp_2m"},
        MODEL_REFERENCE[FLUX_MODEL]["requirements"],
    )

    assert clamped["steps"] == 8, "the reference caps the model at eight steps"
    assert clamped["cfg_scale"] == 4
    assert clamped["sampler_name"] == "k_euler", "the only sampler the model states"


def test_a_model_that_pins_clip_skip_and_a_scheduler_has_both_applied() -> None:
    """Assert a pinned clip_skip is set, and that a scheduler list decides whether karras is on."""
    clamped = apply_requirements(
        {"steps": 25, "cfg_scale": 7, "sampler_name": "k_euler_a", "karras": True},
        MODEL_REFERENCE[CLIP_SKIP_MODEL]["requirements"],
    )

    assert clamped["clip_skip"] == 2
    assert clamped["karras"] is False, "the model states only the simple scheduler"
    assert clamped["steps"] == 25, "a model that states no step range keeps what was asked for"


def test_a_pinned_cfg_scale_replaces_what_was_asked_for() -> None:
    """Assert one stated cfg value wins over the specification's, rather than being clamped to it."""
    clamped = apply_requirements({"steps": 20, "cfg_scale": 7}, MODEL_REFERENCE[PINNED_CFG_MODEL]["requirements"])

    assert clamped["cfg_scale"] == 2.5


def test_a_run_writes_the_pictures_texts_conversations_and_a_manifest(tmp_path: Path) -> None:
    """Assert one full scripted run writes every kind of entry, with the clamped request beside it."""
    transport = ScriptedTransport()
    manifest_path = run_gallery(horde=_horde(transport), spec=SPEC, out=tmp_path / "gallery")

    assert manifest_path is not None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert [entry["id"] for entry in manifest["images"]] == ["fox-snow", "airship"], "specification order"
    assert manifest["images"][0]["tags"] == ["everyday", "photo"], "tags travel from the specification"
    assert manifest["images"][0]["file"] == "gallery/fox-snow/result.webp"
    assert manifest["images"][0]["alchemy"]["caption"]["file"] == "gallery/fox-snow/caption.txt"
    assert manifest["texts"][0]["format"] == TEXT_FORMAT
    assert manifest["censored"] == []

    airship = json.loads((tmp_path / "gallery" / "airship" / META_NAME).read_text(encoding="utf-8"))
    assert airship["request"]["params"]["steps"] == 8, "the request recorded is the one that was sent"
    assert airship["request"]["params"]["sampler_name"] == "k_euler"

    text_meta = json.loads((tmp_path / "gallery" / TEXT_DIRECTORY / "lighthouse-poem" / META_NAME).read_text("utf-8"))
    assert text_meta["request"]["prompt"] == TEXT_FRAME.format(instruction=SPEC["texts"][0]["prompt"])
    assert text_meta["format"] == TEXT_FORMAT
    assert "worker" not in json.dumps(manifest), "the worker that served an entry is never recorded"


def test_the_manifest_is_rebuilt_from_disk_and_keeps_curation(tmp_path: Path) -> None:
    """Assert a narrowed run republishes everything on disk, with the curation that the manifest held.

    A rebuild from what one run touched would drop every entry that run passed over, which is how a
    curated gallery loses its curation and half its pictures at once.
    """
    out = tmp_path / "gallery"
    run_gallery(horde=_horde(ScriptedTransport()), spec=SPEC, out=out)

    manifest = json.loads((out / MANIFEST_NAME).read_text(encoding="utf-8"))
    manifest["images"][0]["curated"] = True
    manifest["images"][0]["note"] = "the fox is the header picture"
    (out / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    transport = ScriptedTransport()
    run_gallery(
        horde=_horde(transport),
        spec=SPEC,
        out=out,
        only={"airship"},
        skip_text=True,
        skip_conversations=True,
    )

    assert transport.image_submissions == [], "everything asked for is already on disk"
    rebuilt = json.loads((out / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert [entry["id"] for entry in rebuilt["images"]] == ["fox-snow", "airship"]
    assert [entry["id"] for entry in rebuilt["texts"]] == ["lighthouse-poem"], "a skipped section is still published"
    assert len(rebuilt["conversations"]) == 2
    assert rebuilt["images"][0]["curated"] is True
    assert rebuilt["images"][0]["note"] == "the fox is the header picture"
    assert "curated" not in rebuilt["images"][1], "curation is absent until a curator writes it"
    assert rebuilt["images"][0]["request"]["prompt"] == SPEC["images"][0]["prompt"], (
        "the request that made a picture is published with it"
    )
    assert "params" in rebuilt["images"][0]["request"]


def test_a_censored_entry_is_recorded_and_left_alone_until_it_is_retried(tmp_path: Path) -> None:
    """Assert a refusal costs the run once: it is written down, listed, and not offered again."""
    out = tmp_path / "gallery"
    refused = frozenset({SPEC["images"][0]["prompt"]})
    run_gallery(
        horde=_horde(ScriptedTransport(censored=refused)),
        spec=SPEC,
        out=out,
        skip_text=True,
        skip_conversations=True,
    )

    refusal = json.loads((out / "fox-snow" / CENSORED_NAME).read_text(encoding="utf-8"))
    assert refusal["request"]["prompt"] == SPEC["images"][0]["prompt"]
    assert refusal["time"], "the refusal records when it happened"
    assert not (out / "fox-snow" / META_NAME).is_file(), "a refusal is not a result"
    manifest = json.loads((out / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["censored"] == ["fox-snow"]
    assert [entry["id"] for entry in manifest["images"]] == ["airship"]

    passed_over = ScriptedTransport(censored=refused)
    run_gallery(horde=_horde(passed_over), spec=SPEC, out=out, skip_text=True, skip_conversations=True)
    assert passed_over.image_submissions == [], "a refusal already on disk is not paid for twice"

    retried = ScriptedTransport()
    run_gallery(
        horde=_horde(retried),
        spec=SPEC,
        out=out,
        skip_text=True,
        skip_conversations=True,
        retry_censored=True,
    )
    assert [body["prompt"] for body in retried.image_submissions] == [SPEC["images"][0]["prompt"]]
    assert not (out / "fox-snow" / CENSORED_NAME).is_file(), "the refusal is cleared once the entry renders"
    assert json.loads((out / MANIFEST_NAME).read_text(encoding="utf-8"))["censored"] == []


def test_an_instruct_conversation_carries_its_earlier_turns(tmp_path: Path) -> None:
    """Assert the second turn is sent with the first turn and its reply inside the frame."""
    out = tmp_path / "gallery"
    transport = ScriptedTransport(replies=["Ember, who bakes at dawn.", "The oven will not light."])
    run_gallery(
        horde=_horde(transport),
        spec=SPEC,
        out=out,
        only={"thread-fox"},
        skip_text=True,
    )

    first, second = (body["prompt"] for body in transport.text_submissions)
    assert first == TEXT_FRAME.format(instruction="Name a fox who runs a bakery.")
    assert second.startswith(first + "Ember, who bakes at dawn.\n\n")
    assert second.endswith(TEXT_FRAME.format(instruction="What goes wrong on opening day?"))
    assert transport.text_submissions[1]["params"]["stop_sequence"] == ["###", "\n\n\n", "</s>", "<|"]

    meta = json.loads((out / CONVERSATION_DIRECTORY / "thread-fox" / META_NAME).read_text(encoding="utf-8"))
    assert [turn["reply"] for turn in meta["turns"]] == ["Ember, who bakes at dawn.", "The oven will not light."]
    assert meta["turns"][1]["request"]["prompt"] == second, "every turn keeps the body it was sent as"
    assert meta["turns"][0]["model"] == "koboldcpp/a-text-model"
    assert "The oven will not light." in (out / CONVERSATION_DIRECTORY / "thread-fox" / "transcript.txt").read_text(
        encoding="utf-8",
    )

    entry = json.loads((out / MANIFEST_NAME).read_text(encoding="utf-8"))["conversations"][0]
    assert entry["format"] == "instruct"
    assert entry["file"] == "gallery/conversations/thread-fox/transcript.txt"
    assert [turn["prompt"] for turn in entry["turns"]] == SPEC["conversations"][0]["turns"]
    assert "request" not in entry["turns"][0], "the manifest carries the reading of a turn, not its body"


def test_a_raw_conversation_is_sent_without_a_frame(tmp_path: Path) -> None:
    """Assert a raw entry goes out as the text itself, stopping only on a paragraph break."""
    transport = ScriptedTransport(replies=["bottled thunderstorms."])
    run_gallery(
        horde=_horde(transport),
        spec=SPEC,
        out=tmp_path / "gallery",
        only={"story-market"},
        skip_text=True,
    )

    submitted = transport.text_submissions[0]
    assert submitted["prompt"] == SPEC["conversations"][1]["turns"][0]
    assert "### Instruction:" not in submitted["prompt"]
    assert submitted["params"]["stop_sequence"] == ["\n\n\n"]
    assert submitted["params"]["max_length"] == 120


def test_a_dry_run_submits_nothing_and_prints_the_model_and_settings(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Assert the plan names each picture's resolved model and clamped settings, and writes no files."""
    out = tmp_path / "gallery"
    transport = ScriptedTransport()

    assert run_gallery(horde=_horde(transport), spec=SPEC, out=out, dry_run=True) is None

    printed = capsys.readouterr().out
    assert f"image airship: {FLUX_MODEL}" in printed
    assert "steps=8" in printed and "sampler_name=k_euler" in printed
    assert "text lighthouse-poem" in printed
    assert "conversation thread-fox: instruct, 2 turns" in printed
    assert transport.image_submissions == []
    assert transport.text_submissions == []
    assert not (out / MANIFEST_NAME).is_file()
    assert list(out.glob(f"*/{META_NAME}")) == []


def test_the_manifest_reads_an_entry_no_specification_lists(tmp_path: Path) -> None:
    """Assert a picture on disk is published even when the specification no longer names it.

    What is on disk is the gallery: an entry dropped from the specification is still a render that
    was paid for, and silently losing it from the manifest is worse than listing it at the end.
    """
    out = tmp_path / "gallery"
    run_gallery(horde=_horde(ScriptedTransport()), spec=SPEC, out=out, skip_text=True, skip_conversations=True)
    narrowed = {**SPEC, "images": [SPEC["images"][1]]}

    manifest = build_manifest(spec=narrowed, out=out)

    assert [entry["id"] for entry in manifest["images"]] == ["airship", "fox-snow"], "the stranger sorts last"
    assert manifest["images"][1]["tags"] == ["everyday", "photo"], "its own metadata still carries its tags"


def test_the_subcommand_is_wired_with_its_defaults() -> None:
    """Assert the parser accepts ``gallery`` and has the flags that the runbook documents."""
    arguments = build_parser().parse_args(["gallery", "--dry-run", "--only", "fox-snow", "--retry-censored"])

    assert arguments.command == "gallery"
    assert arguments.dry_run is True
    assert arguments.only == "fox-snow"
    assert arguments.retry_censored is True
    assert arguments.spec.name == "gallery_spec.json"
    assert arguments.spec.is_file(), "the default gallery specification is tracked in this repository"
    assert arguments.out.name == "gallery"
    assert (arguments.skip_alchemy, arguments.skip_text, arguments.skip_conversations) == (False, False, False)
