# SPDX-FileCopyrightText: 2026 Tazlin
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Make the promo gallery through the live horde: pictures, post-processing pairs, texts and conversations.

Every entry is recorded with what a receipt shows: the seconds it took, the kudos it cost, the
model that made it, and the request itself. The worker that served it is never written down, so
nothing here can name a volunteer. Runs are resumable: an entry whose files are on disk is kept,
and the manifest is rebuilt from the whole tree rather than from the entries one run touched, so a
narrowed run can never drop what an earlier one made.

The command that drives this module is ``param-sweeps gallery``; see the README.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
from PIL import Image

MANIFEST_NAME = "gallery.json"
"""The manifest at the gallery root, which is what the site and the publishing step read."""

MODEL_REFERENCE_URL = "https://models.aihorde.net/api/model_references/v2/image_generation"
"""The image model reference, whose ``requirements`` say what each model will accept."""

CENSORED_NAME = "censored.json"
"""What an entry the safety checker refused leaves behind, so a later run does not pay for it again."""

TEXT_DIRECTORY = "text"
CONVERSATION_DIRECTORY = "conversations"
TRANSCRIPT_NAME = "transcript.txt"
META_NAME = "meta.json"
RESULT_IMAGE_NAME = "result.webp"
RESULT_TEXT_NAME = "result.txt"

POLL_SECONDS = 6
MAX_WAIT_SECONDS = 20 * 60

MAX_IN_FLIGHT = 25
"""A user may hold up to 30 requests at once; a few are left for whatever else the key is doing."""

TEXT_FRAME = "### Instruction:\n{instruction}\n\n### Response:\n"
"""The Alpaca frame, which is the one the widest range of the horde's instruct models were tuned on."""

TEXT_FORMAT = "instruct:alpaca"
"""What a single-turn text entry records about how it was asked, since the frame shapes the answer."""

INSTRUCT_STOP_SEQUENCES = ["###", "\n\n\n", "</s>", "<|"]
"""Cut the answer where the frame would start another turn, or where a chat model closes one."""

RAW_STOP_SEQUENCES = ["\n\n\n"]
"""A raw completion carries no frame, so only a paragraph break can end it."""

INSTRUCT_FORMAT = "instruct"
RAW_FORMAT = "raw"

TEXT_CONTEXT_LENGTH = 1024
CONVERSATION_CONTEXT_LENGTH = 2048
"""A thread carries every turn before it in its prompt, so it needs more room than one question."""

logger = logging.getLogger("param_sweeps.gallery")


class GalleryHorde:
    """The public horde's endpoints, over one session carrying the key and the client agent."""

    def __init__(self, api: str, key: str, agent: str) -> None:
        """Open the session this run submits through.

        Args:
            api: The API root, such as ``https://aihorde.net/api/v2``.
            key: The API key to submit with.
            agent: The Client-Agent string identifying this tool to the horde.
        """
        self.api = api.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"apikey": key, "Client-Agent": agent})

    def model_reference(self) -> dict[str, Any]:
        """Return the image model reference, keyed by model name."""
        response = self.session.get(MODEL_REFERENCE_URL, timeout=120)
        response.raise_for_status()
        reference: dict[str, Any] = response.json()
        return reference

    def served_models(self, kind: str) -> dict[str, int]:
        """Return how many workers serve each model of the given kind (``image`` or ``text``)."""
        response = self.session.get(f"{self.api}/status/models", params={"type": kind}, timeout=60)
        response.raise_for_status()
        return {entry["name"]: int(entry.get("count", 0)) for entry in response.json()}

    def download(self, url: str) -> bytes:
        """Return the bytes of a result the horde parked in object storage."""
        response = self.session.get(url, timeout=120)
        response.raise_for_status()
        return response.content


def load_gallery_spec(path: Path) -> dict[str, Any]:
    """Return the gallery specification, which belongs to the consumer rather than to this tool."""
    return _read_json(path)


def save_webp(data: bytes, destination: Path) -> tuple[int, int]:
    """Write image bytes out as WebP, and return the size written."""
    with Image.open(io.BytesIO(data)) as picture:
        converted = picture.convert("RGBA") if picture.mode in {"RGBA", "LA", "P"} else picture.convert("RGB")
        converted.save(destination, format="WEBP", quality=92, method=6)
        return converted.width, converted.height


def to_base64(path: Path) -> str:
    """Return a file's bytes as the base64 the alchemy endpoint's ``source_image`` takes."""
    return base64.b64encode(path.read_bytes()).decode("ascii")


def apply_requirements(params: dict[str, Any], requirements: dict[str, Any] | None) -> dict[str, Any]:
    """Bring a request's parameters inside what the model reference says the model accepts.

    Steps and cfg are clamped to the stated range (or pinned where one value is given); a sampler
    or scheduler outside the allowed list is replaced by the first allowed one; clip_skip is
    pinned. A request that ignores these is refused by the horde or renders wrong, and a promo
    picture rendered wrong is worse than none.

    Args:
        params: The parameters the specification asked for.
        requirements: The model's ``requirements`` from the reference, or None when it states none.

    Returns:
        The parameters to submit, which are the ones given when the model states no requirements.
    """
    if not requirements:
        return params
    out = dict(params)
    if "min_steps" in requirements:
        out["steps"] = max(int(requirements["min_steps"]), out["steps"])
    if "max_steps" in requirements:
        out["steps"] = min(int(requirements["max_steps"]), out["steps"])
    if "cfg_scale" in requirements:
        out["cfg_scale"] = requirements["cfg_scale"]
    if "min_cfg_scale" in requirements:
        out["cfg_scale"] = max(requirements["min_cfg_scale"], out["cfg_scale"])
    if "max_cfg_scale" in requirements:
        out["cfg_scale"] = min(requirements["max_cfg_scale"], out["cfg_scale"])
    if "clip_skip" in requirements:
        out["clip_skip"] = int(requirements["clip_skip"])
    samplers = requirements.get("samplers")
    if samplers and out.get("sampler_name") not in samplers:
        out["sampler_name"] = samplers[0]
    schedulers = requirements.get("schedulers")
    if schedulers:
        out["karras"] = "karras" in schedulers
    return out


def pick_model(preferences: list[str], served: dict[str, int]) -> str | None:
    """Return the first preferred model a worker is serving right now, or None when none is."""
    for name in preferences:
        if served.get(name, 0) > 0:
            return name
    return None


class Pending:
    """One submitted request, and where its progress and its result are read from."""

    def __init__(self, label: str, request_id: str, check_url: str, status_url: str) -> None:
        """Record the request, its two URLs, and when it went out."""
        self.label = label
        self.request_id = request_id
        self.check_url = check_url
        self.status_url = status_url
        self.started = time.monotonic()


def submit_and_collect(
    horde: GalleryHorde,
    jobs: list[tuple[str, dict[str, Any]]],
    *,
    endpoint: str,
    check_path: str,
    status_path: str,
) -> dict[str, tuple[dict[str, Any], float]]:
    """Submit every job, keeping at most MAX_IN_FLIGHT open, and return each result with its seconds.

    Args:
        horde: The session to submit through.
        jobs: The label and request body of each job.
        endpoint: The path submissions are posted to, relative to the API root.
        check_path: The path a request's progress is read from, without its id.
        status_path: The path a finished request's result is read from, without its id.

    Returns:
        The result and the wall-clock seconds of every job that finished, keyed by label. A job the
        horde refused or faulted is absent rather than raising, because one bad entry must not cost
        the run the rest of the batch.
    """
    results: dict[str, tuple[dict[str, Any], float]] = {}
    queue = list(jobs)
    in_flight: list[Pending] = []
    deadline = time.monotonic() + MAX_WAIT_SECONDS
    while queue or in_flight:
        while queue and len(in_flight) < MAX_IN_FLIGHT:
            label, body = queue.pop(0)
            response = horde.session.post(f"{horde.api}/{endpoint}", json=body, timeout=120)
            if response.status_code >= 400:
                logger.error("%s: %s %s", label, response.status_code, response.text[:200])
                continue
            request_id = response.json()["id"]
            in_flight.append(
                Pending(
                    label,
                    request_id,
                    f"{horde.api}/{check_path}/{request_id}",
                    f"{horde.api}/{status_path}/{request_id}",
                ),
            )
            logger.info("%s: submitted (%d in flight)", label, len(in_flight))
            time.sleep(0.5)
        still: list[Pending] = []
        for pending in in_flight:
            try:
                response = horde.session.get(pending.check_url, timeout=60)
                response.raise_for_status()
                status = response.json()
            except requests.RequestException as error:
                logger.warning("%s: poll failed, retrying: %s", pending.label, error)
                still.append(pending)
                continue
            forms = status.get("forms")
            done = bool(status.get("done")) or (
                forms is not None and bool(forms) and all(form.get("state") in {"done", "faulted"} for form in forms)
            )
            if status.get("faulted"):
                logger.error("%s: faulted", pending.label)
                continue
            if done:
                final = horde.session.get(pending.status_url, timeout=60)
                final.raise_for_status()
                elapsed = time.monotonic() - pending.started
                results[pending.label] = (final.json(), elapsed)
                logger.info("%s: done in %.1f s", pending.label, elapsed)
            else:
                still.append(pending)
        in_flight = still
        if in_flight:
            if time.monotonic() > deadline:
                logger.error("giving up on %s", [pending.label for pending in in_flight])
                break
            logger.info("%d in flight, %d queued", len(in_flight), len(queue))
            time.sleep(POLL_SECONDS)
    return results


@dataclass(frozen=True)
class ImagePlan:
    """One picture a run would submit: the model it resolved to, and the request it would send."""

    entry_id: str
    model: str
    body: dict[str, Any]

    @property
    def params(self) -> dict[str, Any]:
        """Return the request's parameters, as they stand after the model's requirements."""
        params: dict[str, Any] = self.body["params"]
        return params


@dataclass
class GalleryPlan:
    """What a run would submit, which is what a dry run prints instead of submitting."""

    images: list[ImagePlan] = field(default_factory=list)
    kept_images: list[str] = field(default_factory=list)
    censored_images: list[str] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    conversations: list[tuple[str, str, int]] = field(default_factory=list)

    def lines(self) -> list[str]:
        """Return the plan as one line per entry, with a count to close it."""
        lines = [
            f"image {plan.entry_id}: {plan.model} "
            f"({', '.join(f'{name}={value}' for name, value in sorted(plan.params.items()))})"
            for plan in self.images
        ]
        lines.extend(f"text {text_id}" for text_id in self.texts)
        lines.extend(
            f"conversation {entry_id}: {text_format}, {turns} turns"
            for entry_id, text_format, turns in self.conversations
        )
        lines.append(
            f"would submit {len(self.images)} images, {len(self.texts)} texts, "
            f"{len(self.conversations)} conversations; "
            f"{len(self.kept_images)} pictures kept, {len(self.censored_images)} left censored"
        )
        return lines


def plan_images(
    *,
    horde: GalleryHorde,
    spec: dict[str, Any],
    out: Path,
    only: set[str],
    retry_censored: bool,
) -> tuple[GalleryPlan, dict[str, dict[str, Any]]]:
    """Resolve every picture to a model and a request within that model's requirements.

    Args:
        horde: The session the served-model list and the model reference are read through.
        spec: The gallery specification.
        out: The gallery root.
        only: The entry ids to consider, or an empty set for all of them.
        retry_censored: Whether an entry the safety checker refused before is offered again.

    Returns:
        The plan, and the metadata already on disk for the entries this run covers.
    """
    served = horde.served_models("image")
    reference = horde.model_reference()
    plan = GalleryPlan()
    kept: dict[str, dict[str, Any]] = {}
    for item in _selected(spec.get("images", []), only):
        entry_id = item["id"]
        folder = out / entry_id
        if (folder / META_NAME).is_file() and (folder / RESULT_IMAGE_NAME).is_file():
            kept[entry_id] = _read_json(folder / META_NAME)
            plan.kept_images.append(entry_id)
            continue
        if (folder / CENSORED_NAME).is_file() and not retry_censored:
            plan.censored_images.append(entry_id)
            continue
        model = pick_model(item["models"], served) or pick_model(spec["families"][item["family"]], served)
        if model is None:
            logger.warning("%s: no preferred model is served right now, skipped", entry_id)
            continue
        params = dict(spec["defaults"][item["family"]])
        params.update({"n": 1, "seed": str(item["seed"])})
        params = apply_requirements(params, (reference.get(model) or {}).get("requirements"))
        body = {
            "prompt": item["prompt"],
            "params": params,
            "nsfw": False,
            "censor_nsfw": True,
            "models": [model],
            "r2": True,
            "slow_workers": True,
            "shared": False,
        }
        plan.images.append(ImagePlan(entry_id=entry_id, model=model, body=body))
    return plan, kept


def run_images(
    *,
    horde: GalleryHorde,
    spec: dict[str, Any],
    out: Path,
    plan: GalleryPlan,
    kept: dict[str, dict[str, Any]],
    skip_alchemy: bool,
) -> None:
    """Render the planned pictures and apply the alchemy forms their entries ask for.

    A picture the safety checker refused leaves a ``censored.json`` behind rather than a result, so
    that a later run passes over it instead of paying for the same refusal again.
    """
    items = {item["id"]: item for item in spec.get("images", [])}
    metas: dict[str, dict[str, Any]] = dict(kept)
    bodies = {image.entry_id: image.body for image in plan.images}
    for entry_id in bodies:
        (out / entry_id).mkdir(parents=True, exist_ok=True)
    for entry_id in plan.kept_images:
        logger.info("%s: kept", entry_id)
    for entry_id in plan.censored_images:
        logger.info("%s: left censored, offer it again with --retry-censored", entry_id)

    collected = submit_and_collect(
        horde,
        list(bodies.items()),
        endpoint="generate/async",
        check_path="generate/check",
        status_path="generate/status",
    )
    for label, (result, seconds) in collected.items():
        item = items[label]
        folder = out / label
        body = bodies[label]
        generation = result["generations"][0]
        if generation.get("censored"):
            _write_json(
                folder / CENSORED_NAME,
                {
                    "id": label,
                    "request": body,
                    "seconds": round(seconds, 1),
                    "time": datetime.now(UTC).isoformat(timespec="seconds"),
                },
            )
            logger.warning("%s: censored by the safety checker, recorded and skipped", label)
            continue
        width, height = save_webp(horde.download(generation["img"]), folder / RESULT_IMAGE_NAME)
        meta = {
            "id": label,
            "prompt": item["prompt"],
            "tags": list(item.get("tags", [])),
            "model_requested": body["models"][0],
            "model": generation.get("model", body["models"][0]),
            "seed": generation.get("seed"),
            "seconds": round(seconds, 1),
            "kudos": result.get("kudos"),
            "width": width,
            "height": height,
            "request": body,
            "alchemy": {},
        }
        _write_json(folder / META_NAME, meta)
        (folder / CENSORED_NAME).unlink(missing_ok=True)
        logger.info("%s: %s in %.1f s for %s kudos", label, meta["model"], seconds, meta["kudos"])
        metas[label] = meta

    if skip_alchemy:
        return
    run_image_alchemy(horde=horde, items=items, metas=metas, out=out)


def run_image_alchemy(
    *,
    horde: GalleryHorde,
    items: dict[str, dict[str, Any]],
    metas: dict[str, dict[str, Any]],
    out: Path,
) -> None:
    """Apply each entry's outstanding alchemy forms to the picture on disk, and record what came back."""
    jobs: list[tuple[str, dict[str, Any]]] = []
    for label, meta in metas.items():
        forms = [form for form in items[label].get("alchemy", []) if form not in meta.get("alchemy", {})]
        if not forms:
            continue
        jobs.append(
            (
                label,
                {
                    "forms": [{"name": form} for form in forms],
                    "source_image": to_base64(out / label / RESULT_IMAGE_NAME),
                    "slow_workers": True,
                },
            ),
        )
    results = submit_and_collect(
        horde,
        jobs,
        endpoint="interrogate/async",
        check_path="interrogate/status",
        status_path="interrogate/status",
    )
    for label, (payload, seconds) in results.items():
        meta = metas[label]
        folder = out / label
        for entry in payload.get("forms", []):
            form = entry["form"]
            if entry.get("state") != "done":
                logger.warning("%s: form %s ended %s", label, form, entry.get("state"))
                continue
            result_value = entry.get("result", {})
            record: dict[str, Any] = {"seconds": round(seconds, 1)}
            if form == "caption":
                text = result_value.get("caption") if isinstance(result_value, dict) else str(result_value)
                (folder / "caption.txt").write_text(f"{text}\n", encoding="utf-8")
                record.update({"file": "caption.txt", "text": text})
            else:
                url = result_value.get(form) if isinstance(result_value, dict) else None
                if not isinstance(url, str) or not url.startswith("http"):
                    logger.warning("%s: form %s returned no image", label, form)
                    continue
                file_name = f"{form}.webp"
                width, height = save_webp(horde.download(url), folder / file_name)
                record.update({"file": file_name, "width": width, "height": height})
            meta.setdefault("alchemy", {})[form] = record
        _write_json(folder / META_NAME, meta)


def text_body(
    *,
    prompt: str,
    max_length: int,
    stop_sequences: list[str],
    max_context_length: int,
) -> dict[str, Any]:
    """Return the text request body, which is recorded verbatim beside whatever it answered."""
    return {
        "prompt": prompt,
        "params": {
            "max_length": max_length,
            "max_context_length": max_context_length,
            "temperature": 0.7,
            "rep_pen": 1.1,
            "n": 1,
            "stop_sequence": list(stop_sequences),
        },
        "slow_workers": True,
    }


def run_texts(*, horde: GalleryHorde, spec: dict[str, Any], out: Path, only: set[str]) -> None:
    """Answer every single-turn text entry that is not already on disk, and write each one's receipt."""
    items = {item["id"]: item for item in _selected(spec.get("texts", []), only)}
    bodies: dict[str, dict[str, Any]] = {}
    for item in items.values():
        folder = out / TEXT_DIRECTORY / item["id"]
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / META_NAME).is_file():
            logger.info("%s: kept", item["id"])
            continue
        # The text endpoint is raw completion: without a frame around the instruction most models
        # carry the instruction on rather than answer it. The frame is recorded as the entry's
        # format, because it is part of what produced the answer.
        bodies[item["id"]] = text_body(
            prompt=TEXT_FRAME.format(instruction=item["prompt"]),
            max_length=int(item.get("max_length", 80)),
            stop_sequences=INSTRUCT_STOP_SEQUENCES,
            max_context_length=TEXT_CONTEXT_LENGTH,
        )
    collected = submit_and_collect(
        horde,
        list(bodies.items()),
        endpoint="generate/text/async",
        check_path="generate/text/status",
        status_path="generate/text/status",
    )
    for label, (result, seconds) in collected.items():
        item = items[label]
        generation = result["generations"][0]
        text = str(generation.get("text", "")).strip()
        folder = out / TEXT_DIRECTORY / label
        (folder / RESULT_TEXT_NAME).write_text(f"{text}\n", encoding="utf-8")
        meta = {
            "id": label,
            "prompt": item["prompt"],
            "format": TEXT_FORMAT,
            "model": generation.get("model"),
            "text": text,
            "seconds": round(seconds, 1),
            "kudos": result.get("kudos"),
            "request": bodies[label],
        }
        _write_json(folder / META_NAME, meta)
        logger.info("%s: %s in %.1f s for %s kudos", label, meta["model"], seconds, meta["kudos"])


def conversation_prompt(*, text_format: str, history: list[dict[str, Any]], turn: str) -> str:
    """Return the prompt one turn is sent as, carrying every turn before it.

    An instruct thread is sent as the Alpaca frame repeated: each turn already answered as an
    instruction with the answer it drew, then this turn's instruction with an empty response for
    the model to fill. A raw thread carries no frame at all, because the point of it is the
    continuation a bare completion gives, so what goes out is the text as it stands so far.

    Args:
        text_format: ``instruct`` or ``raw``.
        history: The turns already answered, each with its ``prompt`` and its ``reply``.
        turn: The text of the turn being sent.

    Returns:
        The prompt to submit.
    """
    if text_format == INSTRUCT_FORMAT:
        answered = "".join(
            TEXT_FRAME.format(instruction=entry["prompt"]) + entry["reply"] + "\n\n" for entry in history
        )
        return answered + TEXT_FRAME.format(instruction=turn)
    return "".join(f"{entry['prompt']} {entry['reply']}\n\n" for entry in history) + turn


def conversation_body(*, item: dict[str, Any], history: list[dict[str, Any]], turn: str) -> dict[str, Any]:
    """Return the request body for one turn of a conversation."""
    text_format = str(item.get("format", INSTRUCT_FORMAT))
    return text_body(
        prompt=conversation_prompt(text_format=text_format, history=history, turn=turn),
        max_length=int(item.get("max_length", 80)),
        stop_sequences=INSTRUCT_STOP_SEQUENCES if text_format == INSTRUCT_FORMAT else RAW_STOP_SEQUENCES,
        max_context_length=CONVERSATION_CONTEXT_LENGTH,
    )


def run_conversations(*, horde: GalleryHorde, spec: dict[str, Any], out: Path, only: set[str]) -> None:
    """Run every conversation turn by turn, and write its transcript and its receipts.

    The turns of one conversation are strictly sequential, because each carries the answers before
    it, but the turn at the same depth in every conversation goes out together, so a set of threads
    costs about as much waiting as the longest one of them. A conversation whose turns did not all
    come back is left unwritten, so the next run starts it again rather than publishing half of it.
    """
    items = {item["id"]: item for item in _selected(spec.get("conversations", []), only)}
    histories: dict[str, list[dict[str, Any]]] = {}
    for item in items.values():
        folder = out / CONVERSATION_DIRECTORY / item["id"]
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / META_NAME).is_file():
            logger.info("%s: kept", item["id"])
            continue
        histories[item["id"]] = []

    if not histories:
        return

    for index in range(max(len(items[entry_id]["turns"]) for entry_id in histories)):
        jobs: list[tuple[str, dict[str, Any]]] = []
        turns: dict[str, str] = {}
        for entry_id, history in histories.items():
            item = items[entry_id]
            if index >= len(item["turns"]) or len(history) != index:
                continue
            turns[entry_id] = item["turns"][index]
            jobs.append((entry_id, conversation_body(item=item, history=history, turn=turns[entry_id])))
        if not jobs:
            continue
        bodies = dict(jobs)
        collected = submit_and_collect(
            horde,
            jobs,
            endpoint="generate/text/async",
            check_path="generate/text/status",
            status_path="generate/text/status",
        )
        for entry_id, (result, seconds) in collected.items():
            generation = result["generations"][0]
            record: dict[str, Any] = {
                "prompt": turns[entry_id],
                "reply": str(generation.get("text", "")).strip(),
                "model": generation.get("model"),
                "seconds": round(seconds, 1),
                "kudos": result.get("kudos"),
                "request": bodies[entry_id],
            }
            histories[entry_id].append(record)
            logger.info(
                "%s: %s in %.1f s for %s kudos",
                f"{entry_id} turn {index + 1}",
                record["model"],
                seconds,
                record["kudos"],
            )

    for entry_id, history in histories.items():
        item = items[entry_id]
        if len(history) != len(item["turns"]):
            logger.warning(
                "%s: %d of %d turns came back, left for the next run",
                entry_id,
                len(history),
                len(item["turns"]),
            )
            continue
        folder = out / CONVERSATION_DIRECTORY / entry_id
        meta = {
            "id": entry_id,
            "format": str(item.get("format", INSTRUCT_FORMAT)),
            "model": history[0]["model"],
            "turns": history,
            "seconds": round(sum(float(turn["seconds"]) for turn in history), 1),
            "kudos": _total_kudos(history),
            "file": TRANSCRIPT_NAME,
        }
        (folder / TRANSCRIPT_NAME).write_text(_transcript(history), encoding="utf-8")
        _write_json(folder / META_NAME, meta)
        logger.info("%s: %s in %.1f s for %s kudos", entry_id, meta["model"], meta["seconds"], meta["kudos"])


def build_manifest(*, spec: dict[str, Any], out: Path) -> dict[str, Any]:
    """Rebuild the manifest from everything on disk, keeping the curation the existing one carries.

    The manifest is never assembled from the entries one run touched: a run narrowed by ``--only``,
    or one that skipped the texts, would then publish a manifest missing everything it did not
    visit. What is on disk is the gallery, and this reads all of it. ``curated`` and ``note`` are
    written by the curation tool rather than by a run, so they are carried across from the manifest
    being replaced.

    Args:
        spec: The gallery specification, which supplies the reading order and an entry's tags.
        out: The gallery root.

    Returns:
        The manifest.
    """
    root_name = out.name
    existing = _read_json(out / MANIFEST_NAME) if (out / MANIFEST_NAME).is_file() else {}
    tags_by_id = {item["id"]: list(item.get("tags", [])) for item in spec.get("images", [])}

    images = [
        {
            "id": meta["id"],
            "file": f"{root_name}/{meta['id']}/{RESULT_IMAGE_NAME}",
            "prompt": meta["prompt"],
            "tags": meta.get("tags") or tags_by_id.get(meta["id"], []),
            "model": meta["model"],
            "seed": meta.get("seed"),
            "seconds": meta["seconds"],
            "kudos": meta.get("kudos"),
            "width": meta["width"],
            "height": meta["height"],
            "alchemy": {
                form: {**record, "file": f"{root_name}/{meta['id']}/{record['file']}"}
                for form, record in meta.get("alchemy", {}).items()
                if "file" in record
            },
        }
        for meta in _metas_on_disk(root=out, order=_ids(spec, "images"), section=False)
    ]

    texts = [
        {
            "id": meta["id"],
            "file": f"{root_name}/{TEXT_DIRECTORY}/{meta['id']}/{RESULT_TEXT_NAME}",
            "prompt": meta["prompt"],
            "format": meta.get("format", TEXT_FORMAT),
            "model": meta["model"],
            "text": meta["text"],
            "seconds": meta["seconds"],
            "kudos": meta.get("kudos"),
        }
        for meta in _metas_on_disk(root=out / TEXT_DIRECTORY, order=_ids(spec, "texts"), section=True)
    ]

    conversations = [
        {
            "id": meta["id"],
            "file": f"{root_name}/{CONVERSATION_DIRECTORY}/{meta['id']}/{meta.get('file', TRANSCRIPT_NAME)}",
            "format": meta["format"],
            "model": meta["model"],
            "seconds": meta.get("seconds"),
            "kudos": meta.get("kudos"),
            "turns": [
                {
                    "prompt": turn["prompt"],
                    "reply": turn["reply"],
                    "seconds": turn["seconds"],
                    "kudos": turn.get("kudos"),
                    "model": turn.get("model"),
                }
                for turn in meta.get("turns", [])
            ],
        }
        for meta in _metas_on_disk(
            root=out / CONVERSATION_DIRECTORY,
            order=_ids(spec, "conversations"),
            section=True,
        )
    ]

    return {
        "schema_version": 1,
        "generated_against": spec["api"],
        "images": _with_curation(images, existing.get("images", [])),
        "texts": _with_curation(texts, existing.get("texts", [])),
        "conversations": _with_curation(conversations, existing.get("conversations", [])),
        "censored": sorted(refusal.parent.name for refusal in out.glob(f"*/{CENSORED_NAME}")),
    }


def write_manifest(*, spec: dict[str, Any], out: Path) -> Path:
    """Write gallery.json, and return where it went.

    Every file it names is written relative to the promo tree's root, which is what publishing and
    the site both join their own base onto.
    """
    path = out / MANIFEST_NAME
    _write_json(path, build_manifest(spec=spec, out=out))
    return path


def run_gallery(
    *,
    horde: GalleryHorde,
    spec: dict[str, Any],
    out: Path,
    only: set[str] | None = None,
    skip_alchemy: bool = False,
    skip_text: bool = False,
    skip_conversations: bool = False,
    retry_censored: bool = False,
    dry_run: bool = False,
) -> Path | None:
    """Make everything the specification asks for that is not on disk yet, then rebuild the manifest.

    Args:
        horde: The session everything is submitted through.
        spec: The gallery specification.
        out: The gallery root, which is where the manifest lands.
        only: The entry ids to work on, or None for all of them.
        skip_alchemy: Whether the alchemy forms entries ask for are left alone.
        skip_text: Whether the single-turn texts are left alone.
        skip_conversations: Whether the conversations are left alone.
        retry_censored: Whether an entry the safety checker refused before is offered again.
        dry_run: Whether to print what would be submitted and stop.

    Returns:
        The manifest's path, or None for a dry run, which writes nothing.
    """
    selection = set(only or ())
    out.mkdir(parents=True, exist_ok=True)
    plan, kept = plan_images(horde=horde, spec=spec, out=out, only=selection, retry_censored=retry_censored)
    if not skip_text:
        plan.texts = [
            item["id"]
            for item in _selected(spec.get("texts", []), selection)
            if not (out / TEXT_DIRECTORY / item["id"] / META_NAME).is_file()
        ]
    if not skip_conversations:
        plan.conversations = [
            (item["id"], str(item.get("format", INSTRUCT_FORMAT)), len(item["turns"]))
            for item in _selected(spec.get("conversations", []), selection)
            if not (out / CONVERSATION_DIRECTORY / item["id"] / META_NAME).is_file()
        ]

    if dry_run:
        for line in plan.lines():
            print(line)
        return None

    run_images(horde=horde, spec=spec, out=out, plan=plan, kept=kept, skip_alchemy=skip_alchemy)
    if not skip_text:
        run_texts(horde=horde, spec=spec, out=out, only=selection)
    if not skip_conversations:
        run_conversations(horde=horde, spec=spec, out=out, only=selection)
    return write_manifest(spec=spec, out=out)


def _transcript(history: list[dict[str, Any]]) -> str:
    """Return the conversation as it reads, with what was sent marked off from what came back."""
    return "\n".join(f">>> {turn['prompt']}\n\n{turn['reply']}\n" for turn in history)


def _total_kudos(history: list[dict[str, Any]]) -> float | None:
    """Return what a whole conversation cost, or None when no turn reported a cost."""
    costs = [float(turn["kudos"]) for turn in history if turn.get("kudos") is not None]
    return round(sum(costs), 2) if costs else None


def _ids(spec: dict[str, Any], section: str) -> list[str]:
    """Return the ids a section of the specification lists, which is the order a reader expects."""
    return [item["id"] for item in spec.get(section, [])]


def _selected(items: list[dict[str, Any]], only: set[str]) -> list[dict[str, Any]]:
    """Return the entries a run covers, in the order the specification lists them."""
    return [item for item in items if not only or item["id"] in only]


def _metas_on_disk(*, root: Path, order: list[str], section: bool) -> list[dict[str, Any]]:
    """Return every ``meta.json`` under a root, in specification order with anything else after it.

    Args:
        root: The directory whose immediate subdirectories hold the entries.
        order: The ids the specification lists.
        section: Whether the root is one of the text sections; the gallery root itself holds those
            sections as subdirectories, and must not read them as pictures.

    Returns:
        The metadata of every entry found.
    """
    if not root.is_dir():
        return []
    found: dict[str, dict[str, Any]] = {}
    for meta_path in sorted(root.glob(f"*/{META_NAME}")):
        if not section and meta_path.parent.name in {TEXT_DIRECTORY, CONVERSATION_DIRECTORY}:
            continue
        meta = _read_json(meta_path)
        found[str(meta.get("id", meta_path.parent.name))] = meta
    ordered = [found.pop(entry_id) for entry_id in order if entry_id in found]
    ordered.extend(found[entry_id] for entry_id in sorted(found))
    return ordered


def _with_curation(entries: list[dict[str, Any]], existing: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Carry ``curated`` and ``note`` across from the manifest being replaced.

    Curation is a judgement about an entry rather than a fact about the render, so it is written
    into the manifest by another tool, and a rebuild that dropped it would cost every run's worth
    of that judgement.
    """
    carried = {str(entry.get("id")): entry for entry in existing if isinstance(entry, dict)}
    for entry in entries:
        previous = carried.get(entry["id"], {})
        for key in ("curated", "note"):
            if key in previous:
                entry[key] = previous[key]
    return entries


def _read_json(path: Path) -> dict[str, Any]:
    """Return a JSON object from disk."""
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON object out, formatted the way every other file in the tree is."""
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
