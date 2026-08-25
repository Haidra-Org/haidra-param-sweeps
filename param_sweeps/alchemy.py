# SPDX-FileCopyrightText: 2026 Tazlin
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Apply every alchemy form the horde knows to a shared source, and write the evidence manifest.

Alchemy is the half of the horde that takes an image in rather than a prompt: upscalers, face
restorers, background removal, and the describe family that answers in words instead of pixels.
The form vocabulary is not this tool's to invent, so it is read straight out of
``horde_sdk``'s ``KNOWN_ALCHEMY_TYPES`` and the specification is checked against it: a form the
SDK gains and the specification has not placed is a failure to load the file, not a silent gap
in the published evidence.

Every form in a use case runs on the *same* source image, which is the whole point. A reader
comparing two upscalers is comparing upscalers only if the picture underneath them is the same
picture, so the sources are published once at the root of the tree and every entry points at
one of them rather than carrying its own copy.

A form this machine cannot run is published as an entry whose ``notes`` say so and whose
``files`` carry no result. That is deliberate: the consumer draws an honest empty state from a
manifest that admits the gap, and cannot draw one at all from a manifest that omits the form.

Three backends answer the forms, because the horde runs them in three places. The post-processors
are horde-engine's, and are applied on this machine's GPU with the same stack a worker runs. The
describe family is the alchemist worker's own, has no local surface at all, and is answered by
submitting the image to the live horde and publishing what came back. The control maps are
horde-image-utilities', which the worker runs as its own service, and are answered by
[`HordeImageUtilitiesBackend`][param_sweeps.image_utilities.HordeImageUtilitiesBackend].

The ``annotation`` form is the one form that is not one comparison. It carries a control type, and
each of the detectors ``horde_sdk``'s ``KNOWN_ANNOTATION_CONTROL_TYPES`` names reads something
different out of the same picture, so the specification places the form once per detector and the
entries are told apart by their control type rather than by their form.

The upscalers additionally publish into the sweep library as an ordinary strip, under
[`SweepParameter.UPSCALER_FORM`][param_sweeps.spec.SweepParameter]. One source, one stop per
upscaler, so the enlarge page gets a scrubber over the same comparison the manifest lists.

Critical public members:

- [`AlchemySpec`][param_sweeps.alchemy.AlchemySpec]: the checked-in form placement.
- [`known_alchemy_forms`][param_sweeps.alchemy.known_alchemy_forms]: the SDK's own form list.
- [`known_annotation_control_types`][param_sweeps.alchemy.known_annotation_control_types]: the
  SDK's own detector list.
- [`AlchemyBackend`][param_sweeps.alchemy.AlchemyBackend]: the contract a backend implements.
- [`HordelibAlchemyBackend`][param_sweeps.alchemy.HordelibAlchemyBackend]: the horde-engine one.
- [`LiveHordeAlchemyBackend`][param_sweeps.alchemy.LiveHordeAlchemyBackend]: the public API one.
- [`run_alchemy`][param_sweeps.alchemy.run_alchemy]: apply the forms and write everything.
- [`build_manifest`][param_sweeps.alchemy.build_manifest]: the manifest as the frontpage reads it.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import shutil
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal, override

from horde_sdk.generation_parameters.alchemy.consts import (
    KNOWN_ALCHEMY_TYPES,
    KNOWN_ANNOTATION_CONTROL_TYPES,
    is_facefixer_form,
    is_strip_background_form,
    is_upscaler_form,
)
from PIL import Image
from pydantic import BaseModel, ConfigDict, model_validator
from strenum import StrEnum

from param_sweeps.checkpoint import CampaignCheckpoint
from param_sweeps.horde_client import FormRefusedError, HordeApiClient
from param_sweeps.outputs import WebpEncoding, image_size, published_crop, save_render, write_index
from param_sweeps.spec import (
    TOOL_ROOT,
    CropRegion,
    PublishedCrop,
    RenderStrip,
    SweepParameter,
    SweepSpec,
    value_label,
)

logger = logging.getLogger(__name__)

ALCHEMY_DIRECTORY_NAME = "alchemy"
"""Where the alchemy evidence lives, relative to the root of the sweep library."""

MANIFEST_FILENAME = "manifest.json"
SOURCES_DIRECTORY_NAME = "sources"
"""Where the shared source images are published, once each, at the root of the alchemy tree."""

RESULT_IMAGE_FILENAME = "result.webp"
RESULT_TEXT_FILENAME = "result.txt"
RESULT_JSON_FILENAME = "result.json"
RESULT_SVG_FILENAME = "result.svg"

SOURCE_CROP_FILENAME = "source_crop.webp"
"""Where an entry's source crop lands, inside the entry's own directory.

Inside the entry rather than beside the shared source, because two entries can crop the same
source at different regions and one file beside the source could only hold one of them.
"""

RESULT_CROP_FILENAME = "result_crop.webp"
"""Where an entry's result crop lands, inside the entry's own directory."""

RESULT_URL_SCHEME_PREFIX = "http"
"""What an answer that is a picture starts with: the API returns a URL rather than the bytes."""

ANSWER_LABEL_FIELDS = ("text", "hex")
"""What a listed answer calls the thing it names: a phrase CLIP matched, or a palette colour."""

ANSWER_WEIGHT_FIELDS = ("confidence", "proportion")
"""What a listed answer calls how strongly it holds: a confidence, or a share of the picture."""

UNAVAILABLE_NOTE_PREFIX = "No result was produced here:"
"""What leads a manifest note into the reason a form could not be run."""

EMPTY_FORM_MEMBER = "_NONE"
"""The placeholder member of the SDK's form enum, which names no form and is not published."""

ANNOTATION_FORM = "annotation"
"""The one form that carries a control type: which detector reads the picture is the whole answer."""

CONTROL_MAP_ELSEWHERE_REASON = (
    "the control maps are drawn by horde-image-utilities, the service an alchemist worker runs"
    " beside itself for them, which this tool reaches with --backend hiu"
)
"""Why a backend that is not the image-utilities one publishes an annotation entry as a gap."""

BETA_CATEGORIES_VARIABLE = "HORDELIB_BETA_MODEL_CATEGORIES"
BETA_API_KEY_VARIABLE = "HORDELIB_BETA_MODELS_API_KEY"
BETA_PRIMARY_API_URL_VARIABLE = "HORDE_MODEL_REFERENCE_PRIMARY_API_URL"
"""The three horde-engine reads to decide whether the pending model reference is in play."""

BETA_MODEL_CATEGORIES = ("gfpgan",)
"""Which reference categories the beta opt-in covers: the face restorers, and nothing else yet.

GFPGANv1.3 and RestoreFormer are named by the shared vocabulary but carried only by the pending
queue, so the fix-faces comparison is short two of its four entries without this. The upscalers
are a longer list of gaps and are left alone until each one is worth publishing.
"""

BETA_PRIMARY_API_URL = "https://models.aihorde.net/api"
"""The model reference service the pending records are read from."""

ANONYMOUS_API_KEY = "0000000000"
"""The anonymous key, which is reader-level access and all the pending queue is asked for."""


class AlchemyUseCase(StrEnum):
    """What a reader came to the alchemy pages to do, which is how the forms are grouped.

    The horde's own vocabulary is a flat list of processors. A reader does not arrive wanting
    ``RealESRGAN_x4plus``; they arrive wanting a bigger picture. Each form lands in exactly one
    of these, and a form that fits none of them lands in the one that lies least and says so in
    its notes.
    """

    DESCRIBE = "describe"
    """Ask the image what it is: captioning, interrogation, and the other read-only answers."""

    CONTROL_MAP = "control_map"
    """Read the structure out of a picture and draw it: edges, depth, pose, regions, tone."""

    ENLARGE = "enlarge"
    """Make the picture bigger than it was rendered."""

    FIX_FACES = "fix_faces"
    """Restore a face the sampler left soft."""

    CUT_OUT = "cut_out"
    """Take the subject off its background."""


class AlchemyFormKind(StrEnum):
    """How a form is executed, which decides what a backend does with it."""

    UPSCALE = "upscale"
    FACEFIX = "facefix"
    STRIP_BACKGROUND = "strip_background"
    UNSUPPORTED = "unsupported"
    """A form no backend this tool drives can run, which is published as an honest gap."""


FROZEN_MODEL_CONFIG = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())


def known_alchemy_forms() -> tuple[str, ...]:
    """Return every alchemy form the shared vocabulary names, in its own declaration order.

    Read from ``horde_sdk`` rather than restated here, so the specification is checked against
    what the horde actually offers instead of against a copy of it that can go stale.
    """
    return tuple(
        str(form.value) for name, form in KNOWN_ALCHEMY_TYPES.__members__.items() if name != EMPTY_FORM_MEMBER
    )


def known_annotation_control_types() -> tuple[str, ...]:
    """Return every control map the shared vocabulary names, in its own declaration order.

    Read from ``horde_sdk`` for the same reason the form list is: a detector the horde gains is
    then a failure to load the specification rather than a page that quietly never shows it.
    """
    return tuple(str(control_type.value) for control_type in KNOWN_ANNOTATION_CONTROL_TYPES)


def classify_form(form: str) -> AlchemyFormKind:
    """Return how this form is executed, using the SDK's own membership tests.

    Args:
        form: The form name, as the manifest spells it.

    Returns:
        The kind, which is ``UNSUPPORTED`` for every form whose backend is not horde-engine.
    """
    if is_upscaler_form(form):
        return AlchemyFormKind.UPSCALE
    if is_facefixer_form(form):
        return AlchemyFormKind.FACEFIX
    if is_strip_background_form(form):
        return AlchemyFormKind.STRIP_BACKGROUND
    return AlchemyFormKind.UNSUPPORTED


def form_label(form: str) -> str:
    """Return the filename-safe token a form's directory and strip stop are named by."""
    return value_label(form)


def _check_placed_once(*, placed: list[str], known: list[str], subject: str) -> None:
    """Validate that a placement covers a shared vocabulary exactly, naming what went wrong.

    Args:
        placed: What the specification placed.
        known: What the shared vocabulary names.
        subject: What is being placed, as the error message calls it.

    Raises:
        ValueError: If something is placed twice, is not placed at all, or is not known.
    """
    duplicated = sorted({name for name in placed if placed.count(name) > 1})
    if duplicated:
        raise ValueError(f"these {subject} are placed more than once: {', '.join(duplicated)}")

    missing = sorted(set(known) - set(placed))
    if missing:
        raise ValueError(f"these known {subject} are not placed in any use case: {', '.join(missing)}")

    unknown = sorted(set(placed) - set(known))
    if unknown:
        raise ValueError(f"these {subject} are placed but are not known to horde_sdk: {', '.join(unknown)}")


class AlchemyEntrySpec(BaseModel):
    """Represents one form's placement: which page it belongs on and what it runs against."""

    model_config = FROZEN_MODEL_CONFIG

    form: str
    use_case: AlchemyUseCase
    title: str
    source: str
    """The image this form is applied to, relative to the tool root."""

    control_type: str | None = None
    """Which detector runs, for the ``annotation`` form, whose answer is the detector's own.

    Every other form names its own answer, so this is unset for them.
    """

    crop: CropRegion | None = None
    """The region published at its own pixels, cut from both the source and the result.

    Set it where what the form did lives in detail a page cannot show at the size it draws the
    picture: a face a restorer redrew, or the texture an upscaler invented.
    """

    notes: str | None = None
    """What the consumer should be told about this entry beyond its result.

    This is where a use case that had to be approximated admits it, and where a source that is
    not one of the three world anchors says where it came from.
    """

    @property
    def label(self) -> str:
        """Return the filename-safe token this entry's directory is named by."""
        return form_label(self.control_type if self.control_type is not None else self.form)

    @property
    def relative_directory(self) -> str:
        """Return this entry's directory, relative to the alchemy tree, with forward slashes."""
        return f"{self.use_case}/{self.label}"


class AlchemySpec(BaseModel):
    """Represents the whole checked-in alchemy specification."""

    model_config = FROZEN_MODEL_CONFIG

    schema_version: Literal[1]
    generated_against: str
    """What produced the published results, as the manifest reports it."""

    entries: list[AlchemyEntrySpec]

    @model_validator(mode="after")
    def _check_the_shared_vocabulary_is_placed_exactly_once(self) -> AlchemySpec:
        """Validate that the specification covers the shared vocabulary and nothing else.

        A form the SDK names and this file does not place would quietly vanish from the published
        evidence, and a form this file places twice would give the consumer two answers to the
        same question, so both are read errors rather than surprises at render time. The
        ``annotation`` form is held to the same rule one level down: it is placed once per control
        type the SDK names, because each detector is a different answer about the same picture.

        Raises:
            ValueError: If a form or a control type is placed twice, is not placed at all, or is
                not one the SDK names.
        """
        annotation_entries = [entry for entry in self.entries if entry.form == ANNOTATION_FORM]
        forms_placed = [entry.form for entry in self.entries if entry.form != ANNOTATION_FORM]
        if annotation_entries:
            forms_placed.append(ANNOTATION_FORM)

        _check_placed_once(placed=forms_placed, known=list(known_alchemy_forms()), subject="alchemy forms")

        carries_control_type = {entry.form for entry in self.entries if entry.control_type is not None}
        without_control_type = {entry.form for entry in self.entries if entry.control_type is None}
        misplaced = sorted((carries_control_type - {ANNOTATION_FORM}) | (without_control_type & {ANNOTATION_FORM}))
        if misplaced:
            raise ValueError(
                f"a control type belongs to the {ANNOTATION_FORM} form and to no other,"
                f" but these entries disagree: {', '.join(misplaced)}",
            )

        _check_placed_once(
            placed=[entry.control_type for entry in self.entries if entry.control_type is not None],
            known=list(known_annotation_control_types()),
            subject="annotation control types",
        )

        return self

    def entry_for_form(self, form: str) -> AlchemyEntrySpec:
        """Return the entry placing this form.

        The ``annotation`` form is placed once per control type, so this returns the first of its
        entries; ask by control type instead when it matters which detector is meant.

        Raises:
            KeyError: If the specification places no such form.
        """
        for entry in self.entries:
            if entry.form == form:
                return entry
        raise KeyError(f"no alchemy entry for form {form!r}")


class ManifestFiles(BaseModel):
    """Represents where one entry's images and text live, relative to the alchemy tree."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    source: str
    result: str | None = None
    """The image the form produced, for the forms that answer in pixels."""

    result_text: str | None = None
    """The file holding what the form said, for the describe family, which answers in words."""

    result_json: str | None = None
    """The same answer as the API returned it, for a consumer that wants the fields rather than the lines."""


class ManifestEntry(BaseModel):
    """Represents one form as the frontpage's alchemy manifest lists it."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    use_case: AlchemyUseCase
    form: str
    title: str
    files: ManifestFiles
    control_type: str | None = None
    """Which detector drew the map, for the ``annotation`` form, and nothing for the others."""

    model: str | None = None
    """The weights the form ran on, when the backend can name them."""

    source_width: int | None = None
    """The source image's width in pixels, read from the published file."""

    source_height: int | None = None
    """The source image's height in pixels, read from the published file."""

    result_width: int | None = None
    """The result's width in pixels, or None where the form answered in words or in vectors.

    An upscaler's whole answer is this number against the source's, and a page that cannot say
    it is showing four times the pixels is showing the reader the same picture twice.
    """

    result_height: int | None = None
    """The result's height in pixels, or None where there is no raster result to measure."""

    source_crop: PublishedCrop | None = None
    """The part of the source published at its own pixels, or None where none is."""

    result_crop: PublishedCrop | None = None
    """The part of the result published at its own pixels, or None where none is."""

    notes: str | None = None


class AlchemyManifest(BaseModel):
    """Represents the whole manifest at the root of the alchemy evidence tree."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    generated_against: str
    entries: list[ManifestEntry]


class AlchemyOutcome(BaseModel):
    """Represents what a backend made of one form applied to one image.

    Exactly one of the three results is set. An unavailable form is a first-class outcome rather
    than an exception, because "this machine cannot run it" is something the published evidence
    says out loud rather than something that ends the run.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, protected_namespaces=())

    image: Image.Image | None = None
    text: str | None = None
    structured: dict[str, Any] | None = None
    """The answer as the API returned it, alongside the text a page reads, for a textual answer."""

    svg: str | None = None
    unavailable_reason: str | None = None
    model_name_used: str | None = None

    @property
    def is_available(self) -> bool:
        """Return whether the backend produced a result at all."""
        return self.unavailable_reason is None


class AlchemyBackend(ABC):
    """A backend that applies alchemy forms to images.

    Subclass Integration:
        [`prepare`][param_sweeps.alchemy.AlchemyBackend.prepare] is called once before any form
        is applied. [`apply`][param_sweeps.alchemy.AlchemyBackend.apply] is called once per
        entry, never concurrently, and reports an unavailable form through its outcome rather
        than by raising. [`close`][param_sweeps.alchemy.AlchemyBackend.close] is always called.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the backend's name, as recorded in every sidecar it produces."""

    @abstractmethod
    def prepare(self) -> None:
        """Set the backend up, fetching whatever the forms need."""

    @abstractmethod
    def apply(self, *, form: str, control_type: str | None, source_image: Image.Image) -> AlchemyOutcome:
        """Apply one form to one image, and say what came of it.

        Args:
            form: The form to apply.
            control_type: Which detector to run, for the ``annotation`` form, and None otherwise.
            source_image: The picture to apply it to.
        """

    @property
    def service_host(self) -> str | None:
        """Return the host that answered, when a service outside this process did.

        The default names none, because a backend running in this process has no host to name.
        """
        return None

    @property
    def service_version(self) -> str | None:
        """Return the version the answering service reports, when it reports one."""
        return None

    def close(self) -> None:
        """Release whatever the backend is holding.

        The default releases nothing, because a backend that holds nothing should not have to
        say so. It is not abstract for that reason.
        """
        return


class HordelibAlchemyBackend(AlchemyBackend):
    """Applies alchemy forms through horde-engine on this machine's GPU.

    This is the same inference stack an AI Horde alchemist worker runs, so what it produces is
    what the horde produces. ``hordelib`` is imported inside the methods that need it, for the
    same reason [`LocalRunner`][param_sweeps.local_runner.LocalRunner] does: importing it starts
    a ComfyUI environment check and pulls in torch.

    Only the post-processing forms have a horde-engine surface at all. The describe family, the
    NSFW detector and the newer image-utility forms are served by the alchemist's own backend,
    which is not this stack, and are reported unavailable rather than approximated with
    something else that happens to be installed.
    """

    POST_PROCESSING_MODEL_MANAGERS = ("esrgan", "gfpgan", "codeformer", "miscellaneous")
    """The managers that own the upscaler and face-restorer weights, plus the miscellaneous one.

    The miscellaneous manager is loaded for the same reason the sweep runner loads it: several
    processors resolve their weights through it and find no model folder without it.
    """

    def __init__(
        self,
        *,
        gpu_backend: str,
        device_index: int,
        beta_categories: tuple[str, ...] = (),
    ) -> None:
        """Configure a local alchemy backend.

        Args:
            gpu_backend: The compute backend to pin the process to.
            device_index: The accelerator index to pin the process to.
            beta_categories: Which reference categories to resolve from the pending queue as well
                as from the published reference. Empty means the published reference only.
        """
        self._gpu_backend = gpu_backend
        self._device_index = device_index
        self._beta_categories = beta_categories
        self._horde: Any | None = None

    @property
    def name(self) -> str:
        """Return the backend's name, as recorded in every sidecar it produces."""
        return f"alchemy:{self._gpu_backend}"

    def prepare(self) -> None:
        """Initialise horde-engine and load the post-processing model managers."""
        import hordelib
        from hordelib.utils.torch_memory import AcceleratorKind

        logger.info("Initialising hordelib: backend=%s device_index=%s", self._gpu_backend, self._device_index)
        hordelib.initialise(
            setup_logging=True,
            logging_verbosity=3,
            accelerator_kind=AcceleratorKind(self._gpu_backend),
            device_index=self._device_index,
        )

        from hordelib.horde import HordeLib
        from hordelib.shared_model_manager import SharedModelManager

        self._configure_beta_models()
        SharedModelManager.load_model_managers(list(self.POST_PROCESSING_MODEL_MANAGERS))
        self._horde = HordeLib()

    def _configure_beta_models(self) -> None:
        """Opt the requested categories into the pending model reference, before any manager loads.

        horde-engine registers the pending provider from the environment as it builds the model
        managers, and each manager reads its database once as it is constructed, so the variables
        have to be in place before ``load_model_managers``. A variable the caller already set is
        left as it is: an environment that names its own key or its own service outranks the
        defaults this tool would otherwise supply.
        """
        if not self._beta_categories:
            return

        os.environ[BETA_CATEGORIES_VARIABLE] = ",".join(self._beta_categories)
        os.environ.setdefault(BETA_API_KEY_VARIABLE, ANONYMOUS_API_KEY)
        os.environ.setdefault(BETA_PRIMARY_API_URL_VARIABLE, BETA_PRIMARY_API_URL)
        logger.info("Beta model categories enabled: %s", ", ".join(self._beta_categories))

    def apply(self, *, form: str, control_type: str | None, source_image: Image.Image) -> AlchemyOutcome:
        """Apply one form through horde-engine, reporting an unrunnable one as unavailable.

        Raises:
            RuntimeError: If the backend was not prepared first.
        """
        if self._horde is None:
            raise RuntimeError("prepare() must be called before apply()")

        if control_type is not None:
            return AlchemyOutcome(unavailable_reason=CONTROL_MAP_ELSEWHERE_REASON)

        kind = classify_form(form)
        if kind is AlchemyFormKind.UNSUPPORTED:
            return AlchemyOutcome(
                unavailable_reason=(
                    f"the {form} form is served by the alchemist backend rather than by horde-engine,"
                    " which is what this tool renders through"
                ),
            )

        try:
            return self._applied(horde=self._horde, kind=kind, form=form, source_image=source_image)
        except RuntimeError as backend_error:
            # A form the shared vocabulary names but the live model reference does not carry has
            # no weights to resolve, which is a published gap rather than a failed run.
            logger.warning("Form unavailable: form=%s reason=%s", form, backend_error)
            return AlchemyOutcome(unavailable_reason=str(backend_error))
        except ImportError as missing_dependency:
            logger.warning("Form unavailable: form=%s reason=%s", form, missing_dependency)
            return AlchemyOutcome(unavailable_reason=str(missing_dependency))

    @staticmethod
    def _applied(*, horde: Any, kind: AlchemyFormKind, form: str, source_image: Image.Image) -> AlchemyOutcome:
        """Return what horde-engine made of this form, having established it has a surface for it.

        Raises:
            RuntimeError: If the backend produced no image.
        """
        if kind is AlchemyFormKind.STRIP_BACKGROUND:
            from hordelib.utils.image_utils import ImageUtils

            return AlchemyOutcome(image=ImageUtils.strip_background(source_image), model_name_used=form)

        payload = {"model": form, "source_image": source_image}
        result = horde.image_upscale(payload) if kind is AlchemyFormKind.UPSCALE else horde.image_facefix(payload)
        if result.image is None:
            raise RuntimeError(f"horde-engine returned no image for the {form} form")

        return AlchemyOutcome(image=result.image, model_name_used=form)

    def close(self) -> None:
        """Drop the engine handle."""
        self._horde = None


class LiveHordeAlchemyBackend(AlchemyBackend):
    """Applies alchemy forms by asking the live AI Horde, so the answers are the horde's own.

    The describe family has no horde-engine surface: captioning, interrogation, the NSFW check and
    the newer read-only forms are served by the alchemist worker's own backend. The only way to
    publish what they say without inventing it is to submit the image to the public API and print
    what comes back, which is what this does: one form per request, submitted to
    ``/interrogate/async`` and polled to its answer exactly as a follower's would be.

    A form the deployed API's request schema does not carry is reported unavailable, carrying the
    API's own account of what it does accept. A transport failure is not: a timeout or a dropped
    connection says nothing about what the horde serves, and publishing it as a gap would put a
    claim about the horde's offering into the manifest that only this machine's network supports.
    """

    def __init__(self, *, client: HordeApiClient) -> None:
        """Configure a live backend.

        Args:
            client: The API client to submit through. Not closed here: whoever opened it closes it.
        """
        self._client = client

    @property
    @override
    def name(self) -> str:
        """Return the backend's name, as recorded in every sidecar it produces."""
        return f"alchemy:{self._client.api_base_url}"

    @override
    def prepare(self) -> None:
        """Set nothing up: the horde holds the weights and the client holds the session."""
        return

    @override
    def apply(self, *, form: str, control_type: str | None, source_image: Image.Image) -> AlchemyOutcome:
        """Ask the horde to run one form, and say what came of it."""
        if control_type is not None:
            return AlchemyOutcome(unavailable_reason=CONTROL_MAP_ELSEWHERE_REASON)

        try:
            completed = self._client.interrogate(form=form, source_image_base64=encoded_for_api(source_image))
        except FormRefusedError as refusal:
            logger.warning("Form unavailable: form=%s reason=%s", form, refusal)
            return AlchemyOutcome(unavailable_reason=str(refusal))

        if not completed.result:
            return AlchemyOutcome(unavailable_reason=f"the horde ran the {form} form and returned no answer")

        answer = completed.result.get(form, completed.result)
        if isinstance(answer, str) and answer.startswith(RESULT_URL_SCHEME_PREFIX):
            return AlchemyOutcome(image=self._client.download_image(answer), model_name_used=form)
        if isinstance(answer, str) and answer.lstrip().startswith(("<?xml", "<svg")):
            # The vectorizer answers with the SVG itself. It is a picture, not a
            # transcript, so it publishes as the result a browser can draw.
            return AlchemyOutcome(svg=answer)

        # The weights are the worker's business and the response does not name them, so the
        # manifest says nothing about them rather than repeating the form name as if it were one.
        return AlchemyOutcome(
            text=render_form_answer(form=form, result=completed.result),
            structured=dict(completed.result),
        )


def encoded_for_api(image: Image.Image) -> str:
    """Return this image as the base64 string the API's ``source_image`` field takes."""
    buffer = io.BytesIO()
    image.save(buffer, format="WEBP", quality=WebpEncoding.QUALITY, method=WebpEncoding.METHOD)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def render_form_answer(*, form: str, result: Mapping[str, Any]) -> str:
    """Return what a form said, as the text the evidence page shows verbatim.

    The API wraps every answer in a key named after the form, and what is inside it is a sentence
    for the captioner, a single judgement for the NSFW check and the aesthetic score, and a
    mapping for the forms that answer in data. Nothing here summarises: the numbers the horde
    returned are the numbers written down, because a page showing this as the horde's own answer
    cannot show a rounded-off version of it.

    Args:
        form: The form the answer came from.
        result: The mapping the API returned for it.

    Returns:
        The answer, as lines of text.
    """
    answer = result.get(form, result)
    if isinstance(answer, str):
        return answer
    if isinstance(answer, Mapping):
        return "\n".join(_answer_lines(answer))
    return f"{form}: {_scalar(answer)}"


def _answer_lines(answer: Mapping[str, Any]) -> Iterator[str]:
    """Yield one line per field of an answer that came back as a mapping."""
    for field_name, value in answer.items():
        if isinstance(value, list):
            yield f"{field_name}: {', '.join(_rendered_item(item) for item in value)}"
        else:
            yield f"{field_name}: {_scalar(value)}"


def _rendered_item(item: Any) -> str:
    """Return one item of a listed answer: what it names, and how strongly it holds."""
    if not isinstance(item, Mapping):
        return _scalar(item)

    label = next((str(item[field_name]) for field_name in ANSWER_LABEL_FIELDS if field_name in item), None)
    weight = next((item[field_name] for field_name in ANSWER_WEIGHT_FIELDS if field_name in item), None)
    if label is None:
        return ", ".join(f"{field_name}={_scalar(value)}" for field_name, value in item.items())

    return f"{label} ({_scalar(round(weight, 4))})" if isinstance(weight, int | float) else label


def _scalar(value: Any) -> str:
    """Return one value as the answer spells it, so a boolean reads as the JSON it arrived as."""
    return json.dumps(value) if isinstance(value, bool | None) else str(value)


class AlchemyResult(BaseModel):
    """Represents what applying one form produced."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    form: str
    use_case: AlchemyUseCase
    available: bool
    control_type: str | None = None
    """Which detector drew the map, for the annotation form, so a report can name it."""

    reused: bool = False
    """Whether the result was already on disk and the form was not applied again."""


def load_alchemy_spec(spec_path: Path) -> AlchemySpec:
    """Read and validate the alchemy specification at this path.

    Args:
        spec_path: The specification file to read.

    Returns:
        The validated specification.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the file is not valid JSON, or does not validate as a specification.
    """
    raw_text = spec_path.read_text(encoding="utf-8")
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as decode_error:
        raise ValueError(f"{spec_path} is not valid JSON: {decode_error}") from decode_error

    return AlchemySpec.model_validate(parsed)


def alchemy_root(output_root: Path) -> Path:
    """Return the alchemy evidence tree's root, given the sweep library's root."""
    return output_root / ALCHEMY_DIRECTORY_NAME


def published_source_name(entry: AlchemyEntrySpec) -> str:
    """Return where this entry's source is published, relative to the alchemy tree.

    Named after the input file rather than after the entry, which is what makes two forms
    sharing a source share one published copy of it, and makes that sharing visible to a reader
    comparing their results.
    """
    return f"{SOURCES_DIRECTORY_NAME}/{PurePosixPath(entry.source).name}"


def published_source_crop_name(entry: AlchemyEntrySpec) -> str:
    """Return where this entry's source crop is published, relative to the alchemy tree."""
    return f"{entry.relative_directory}/{SOURCE_CROP_FILENAME}"


def published_result_crop_name(entry: AlchemyEntrySpec) -> str:
    """Return where this entry's result crop is published, relative to the alchemy tree."""
    return f"{entry.relative_directory}/{RESULT_CROP_FILENAME}"


def build_manifest(*, spec: AlchemySpec, output_root: Path) -> AlchemyManifest:
    """Build the manifest from the specification and the files actually on disk.

    Every placed form is listed whether or not it produced anything. An entry with no result is
    the honest report that the form could not be run here, and its ``notes`` say why; leaving it
    out would tell the consumer the form does not exist.

    Args:
        spec: The alchemy specification.
        output_root: The root of the sweep library, which the alchemy tree sits inside.

    Returns:
        The manifest.
    """
    root = alchemy_root(output_root)
    entries: list[ManifestEntry] = []

    for entry in spec.entries:
        record = _read_entry_record(root=root, entry=entry)
        result_name = f"{entry.relative_directory}/{RESULT_IMAGE_FILENAME}"
        svg_name = f"{entry.relative_directory}/{RESULT_SVG_FILENAME}"
        if not (root / result_name).is_file() and (root / svg_name).is_file():
            result_name = svg_name
        text_name = f"{entry.relative_directory}/{RESULT_TEXT_FILENAME}"
        json_name = f"{entry.relative_directory}/{RESULT_JSON_FILENAME}"

        source_name = published_source_name(entry)
        published_result = result_name if (root / result_name).is_file() else None
        source_width, source_height = _published_size(root / source_name)
        result_width, result_height = _published_size(root / result_name)

        entries.append(
            ManifestEntry(
                use_case=entry.use_case,
                form=entry.form,
                title=entry.title,
                files=ManifestFiles(
                    source=source_name,
                    result=published_result,
                    result_text=text_name if (root / text_name).is_file() else None,
                    result_json=json_name if (root / json_name).is_file() else None,
                ),
                control_type=entry.control_type,
                model=record.model if record is not None else None,
                source_width=source_width,
                source_height=source_height,
                result_width=result_width,
                result_height=result_height,
                source_crop=_published_entry_crop(
                    region=entry.crop,
                    root=root,
                    file=published_source_crop_name(entry),
                    frame_name=source_name,
                ),
                result_crop=_published_entry_crop(
                    region=entry.crop,
                    root=root,
                    file=published_result_crop_name(entry),
                    frame_name=result_name,
                ),
                notes=_manifest_notes(entry=entry, record=record),
            ),
        )

    return AlchemyManifest(generated_against=spec.generated_against, entries=entries)


def write_manifest(*, spec: AlchemySpec, output_root: Path) -> Path:
    """Write the alchemy manifest, and return its path.

    Side Effects:
        Overwrites any existing manifest. The content depends only on the specification and the
        files present, so rewriting it never introduces a spurious change.
    """
    manifest_path = alchemy_root(output_root) / MANIFEST_FILENAME
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(spec=spec, output_root=output_root)
    manifest_path.write_text(
        json.dumps(manifest.model_dump(exclude_none=True), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def upscaler_form_strip(*, sweep_spec: SweepSpec) -> RenderStrip:
    """Return the sweep library's upscaler-form strip.

    Raises:
        ValueError: If the sweep specification does not describe exactly one, which is what makes
            one shared source answerable for the whole comparison.
    """
    matching = [strip for strip in sweep_spec.expand_strips() if strip.parameter is SweepParameter.UPSCALER_FORM]
    if len(matching) != 1:
        raise ValueError(
            f"the sweep specification describes {len(matching)} {SweepParameter.UPSCALER_FORM} strips;"
            " it has to describe one",
        )
    return matching[0]


def run_alchemy(
    *,
    spec: AlchemySpec,
    sweep_spec: SweepSpec,
    backend: AlchemyBackend,
    output_root: Path,
    forms: tuple[str, ...] = (),
    control_types: tuple[str, ...] = (),
    force: bool = False,
) -> list[AlchemyResult]:
    """Apply every selected form, publish the strip, and rewrite the manifest and the index.

    An entry whose result is already on disk is left alone unless ``force`` says otherwise, which
    is all the resumption this needs: an alchemy form is deterministic given its source, so the
    file being there is the same statement a checkpoint entry would make.

    Args:
        spec: The alchemy specification.
        sweep_spec: The sweep specification, which carries the upscaler-form strip.
        backend: The backend to apply the forms through, unprepared.
        output_root: The root of the sweep library.
        forms: Which forms to apply. Empty means every one the specification places.
        control_types: Which control maps to draw. Empty means every one; naming any restricts
            the run to the annotation entries drawing them.
        force: Apply selected forms even when their result is already on disk.

    Returns:
        One result per selected form.

    Raises:
        RuntimeError: If a selected entry's source image is not where the specification says.
        ValueError: If the sweep specification does not carry exactly one upscaler-form strip.

    Side Effects:
        Runs on the accelerator, writes the alchemy tree, the strip's images and sidecars, the
        checkpoint, the manifest and the index.
    """
    strip = upscaler_form_strip(sweep_spec=sweep_spec)
    selected = [entry for entry in spec.entries if _is_selected(entry=entry, forms=forms, control_types=control_types)]
    results: list[AlchemyResult] = []
    root = alchemy_root(output_root)

    try:
        backend.prepare()

        for entry_number, entry in enumerate(selected, start=1):
            print(f"[{entry_number}/{len(selected)}] {entry.use_case}/{entry.label}")
            results.append(
                _apply_one_entry(
                    entry=entry,
                    backend=backend,
                    root=root,
                    strip=strip,
                    output_root=output_root,
                    force=force,
                ),
            )
    finally:
        backend.close()
        write_manifest(spec=spec, output_root=output_root)
        write_index(spec=sweep_spec, output_root=output_root)

    return results


def _is_selected(*, entry: AlchemyEntrySpec, forms: tuple[str, ...], control_types: tuple[str, ...]) -> bool:
    """Return whether this entry is in the run the filters describe.

    Naming a control type narrows the run to the annotation entries drawing it, which is what
    makes ``--control-type canny`` a request for one map rather than for every form as well.
    """
    if forms and entry.form not in forms:
        return False
    return not control_types or entry.control_type in control_types


def _apply_one_entry(
    *,
    entry: AlchemyEntrySpec,
    backend: AlchemyBackend,
    root: Path,
    strip: RenderStrip,
    output_root: Path,
    force: bool,
) -> AlchemyResult:
    """Apply one form, write everything it produced, and return what happened.

    Raises:
        RuntimeError: If the entry's source image is not where the specification says it is.
    """
    _publish_source(entry=entry, root=root)

    if not force and _already_applied(root=root, entry=entry):
        logger.info("Result already on disk, leaving it alone: entry=%s", entry.relative_directory)
        return AlchemyResult(
            form=entry.form, use_case=entry.use_case, control_type=entry.control_type, available=True, reused=True
        )

    started_at = datetime.now(UTC)
    outcome = backend.apply(form=entry.form, control_type=entry.control_type, source_image=_source_image(entry))
    finished_at = datetime.now(UTC)

    _write_entry_record(root=root, entry=entry, outcome=outcome, backend=backend)
    if not outcome.is_available:
        _clear_stale_results(root=root, entry=entry)
        return AlchemyResult(
            form=entry.form, use_case=entry.use_case, control_type=entry.control_type, available=False
        )

    directory = root / entry.relative_directory
    directory.mkdir(parents=True, exist_ok=True)

    if outcome.image is not None:
        outcome.image.save(
            directory / RESULT_IMAGE_FILENAME,
            format="WEBP",
            quality=WebpEncoding.QUALITY,
            method=WebpEncoding.METHOD,
        )
        _publish_to_strip(
            entry=entry,
            image=outcome.image,
            strip=strip,
            output_root=output_root,
            backend_name=backend.name,
            started_at=started_at,
            finished_at=finished_at,
            model_name_used=outcome.model_name_used,
        )

    if outcome.text is not None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / RESULT_TEXT_FILENAME).write_text(outcome.text.strip() + "\n", encoding="utf-8")

    if outcome.structured is not None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / RESULT_JSON_FILENAME).write_text(
            json.dumps(outcome.structured, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    if outcome.svg is not None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / RESULT_SVG_FILENAME).write_text(outcome.svg.strip() + "\n", encoding="utf-8")

    return AlchemyResult(form=entry.form, use_case=entry.use_case, control_type=entry.control_type, available=True)


def _publish_to_strip(
    *,
    entry: AlchemyEntrySpec,
    image: Image.Image,
    strip: RenderStrip,
    output_root: Path,
    backend_name: str,
    started_at: datetime,
    finished_at: datetime,
    model_name_used: str | None,
) -> None:
    """Write an upscaler's result into the sweep library as its strip stop, when it has one.

    A form the strip does not declare a stop for writes nothing here, which is what keeps the
    strip a comparison of upscalers rather than of everything the manifest happens to hold.
    """
    stop_jobs = [job for job in strip.jobs if job.stop_value == entry.form]
    if not stop_jobs:
        return

    job = stop_jobs[0]
    save_render(
        images=[image],
        job=job,
        output_root=output_root,
        runner_name=backend_name,
        started_at=started_at,
        finished_at=finished_at,
        model_name_used=model_name_used,
    )
    CampaignCheckpoint(output_root=output_root).mark_complete(
        relative_image_path=job.checkpoint_key,
        runner_name=backend_name,
        seed=job.seed,
    )


class _EntryRecord(BaseModel):
    """Represents the sidecar written beside an entry, recording how its result was produced."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    form: str
    use_case: AlchemyUseCase
    source: str
    backend: str
    applied_at: str
    model: str | None = None
    unavailable_reason: str | None = None
    control_type: str | None = None
    service_host: str | None = None
    """Where the answer came from, when it came from a service rather than from this process."""

    service_version: str | None = None


def _write_entry_record(
    *,
    root: Path,
    entry: AlchemyEntrySpec,
    outcome: AlchemyOutcome,
    backend: AlchemyBackend,
) -> None:
    """Write the sidecar recording what the backend made of this form."""
    directory = root / entry.relative_directory
    directory.mkdir(parents=True, exist_ok=True)
    record = _EntryRecord(
        form=entry.form,
        use_case=entry.use_case,
        source=published_source_name(entry),
        backend=backend.name,
        applied_at=datetime.now(UTC).isoformat(),
        model=outcome.model_name_used,
        unavailable_reason=outcome.unavailable_reason,
        control_type=entry.control_type,
        service_host=backend.service_host,
        service_version=backend.service_version,
    )
    (directory / "meta.json").write_text(
        json.dumps(record.model_dump(exclude_none=True), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _read_entry_record(*, root: Path, entry: AlchemyEntrySpec) -> _EntryRecord | None:
    """Return the sidecar beside this entry, or None when the form has not been applied here."""
    record_path = root / entry.relative_directory / "meta.json"
    if not record_path.is_file():
        return None

    return _EntryRecord.model_validate_json(record_path.read_text(encoding="utf-8"))


def _published_size(path: Path) -> tuple[int | None, int | None]:
    """Return a published file's pixel size, or a pair of Nones when it is not a picture.

    A form can answer in words or in vectors, and an entry can carry no result at all, so this
    reports the absence of a size rather than treating it as a fault.
    """
    if not path.is_file():
        return None, None

    try:
        return image_size(path)
    except OSError as unreadable:
        logger.debug("No pixel size for %s: %s", path, unreadable)
        return None, None


def _published_entry_crop(
    *,
    region: CropRegion | None,
    root: Path,
    file: str,
    frame_name: str,
) -> PublishedCrop | None:
    """Return one side's crop record when it is on disk, and None when the entry declares none."""
    if region is None:
        return None

    return published_crop(file=file, root=root, region=region, frame_relative_path=frame_name)


def _manifest_notes(*, entry: AlchemyEntrySpec, record: _EntryRecord | None) -> str | None:
    """Return what the manifest tells the consumer about this entry.

    The specification's own note comes first, because it is about the form's placement and holds
    whether or not the form ran. An unavailability reason is appended to it rather than replacing
    it, so a reader gets both the caveat and the gap, and is led in by a sentence of its own so
    that the two read as two statements rather than as one run-on.
    """
    reason = record.unavailable_reason if record is not None else None
    gap = f"{UNAVAILABLE_NOTE_PREFIX} {reason}" if reason else None
    parts = [part for part in (entry.notes, gap) if part]
    return " ".join(parts) if parts else None


def _already_applied(*, root: Path, entry: AlchemyEntrySpec) -> bool:
    """Return whether this entry already carries a result on disk."""
    directory = root / entry.relative_directory
    return any(
        (directory / name).is_file() for name in (RESULT_IMAGE_FILENAME, RESULT_SVG_FILENAME, RESULT_TEXT_FILENAME)
    )


def _clear_stale_results(*, root: Path, entry: AlchemyEntrySpec) -> None:
    """Remove a result left over from a run where the form was available and now is not.

    A manifest built from the files on disk would otherwise advertise a result whose entry says
    the form could not be run, which is the one thing the honest-gap reporting exists to avoid.
    """
    directory = root / entry.relative_directory
    for filename in (RESULT_IMAGE_FILENAME, RESULT_SVG_FILENAME, RESULT_TEXT_FILENAME, RESULT_JSON_FILENAME):
        (directory / filename).unlink(missing_ok=True)


def _publish_source(*, entry: AlchemyEntrySpec, root: Path) -> None:
    """Copy this entry's source into the alchemy tree, so the published tree is self-contained.

    Raises:
        RuntimeError: If the source is not where the specification says it is.
    """
    source_path = _source_path(entry)
    destination = root / published_source_name(entry)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, destination)


def _source_path(entry: AlchemyEntrySpec) -> Path:
    """Return this entry's source image path.

    Raises:
        RuntimeError: If the source is not where the specification says it is.
    """
    source_path = TOOL_ROOT / entry.source
    if not source_path.is_file():
        raise RuntimeError(f"the source for the {entry.form} form is missing: {source_path}")
    return source_path


def _source_image(entry: AlchemyEntrySpec) -> Image.Image:
    """Return this entry's source image, decoded.

    Raises:
        RuntimeError: If the source is not where the specification says it is.
    """
    with Image.open(_source_path(entry)) as opened:
        return opened.convert("RGB")


def use_case_counts(spec: AlchemySpec) -> dict[AlchemyUseCase, int]:
    """Return how many forms each use case holds, which is what the command line reports."""
    counts: dict[AlchemyUseCase, int] = dict.fromkeys(AlchemyUseCase, 0)
    for entry in spec.entries:
        counts[entry.use_case] += 1
    return counts
