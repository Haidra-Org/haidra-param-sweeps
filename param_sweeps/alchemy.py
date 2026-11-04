# SPDX-FileCopyrightText: 2026 Tazlin
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Apply every alchemy form in the horde's vocabulary to a shared source, and write the evidence manifest.

Alchemy is the half of the horde that takes an image as input instead of a prompt: upscalers, face
restorers, background removal, and the describe family, which returns words instead of pixels.
The form vocabulary is read from ``horde_sdk``'s ``KNOWN_ALCHEMY_TYPES``, and the specification
is checked against it. If the SDK adds a form that the specification does not place, loading
the specification fails, so the published evidence cannot have a silent gap.

Every form in a use case runs on the *same* source image. A comparison of two upscalers only
compares the upscalers if both start from the same picture, so the sources are published once at
the root of the tree, and every entry points at one of them instead of holding its own copy.

A form that this machine cannot run is published as an entry whose ``notes`` give the reason and
whose ``files`` have no result. A consumer can draw an accurate empty state from a manifest that
records the gap, but cannot draw one for a form that the manifest omits.

Three backends apply the forms, because the horde runs them in three places. The post-processors
belong to horde-engine, and are applied on this machine's GPU with the same stack that a worker
runs. The describe family belongs to the alchemist worker and cannot run locally. It is run by
submitting the image to the live horde and publishing the result. The control maps belong to
horde-image-utilities, which the worker runs as a separate service, and are produced by
[`HordeImageUtilitiesBackend`][param_sweeps.image_utilities.HordeImageUtilitiesBackend].

The ``annotation`` form is the only form with more than one comparison. It has a control type,
and each detector in ``horde_sdk``'s ``KNOWN_ANNOTATION_CONTROL_TYPES`` extracts something
different from the same picture. The specification places the form once per detector, and the
entries are distinguished by control type instead of by form.

The upscalers are also published into the sweep library as an ordinary strip, under
[`SweepParameter.UPSCALER_FORM`][param_sweeps.spec.SweepParameter]. The strip has one source and
one stop per upscaler, so the enlarge page gets a scrubber over the comparison in the manifest.

Critical public members:

- [`AlchemySpec`][param_sweeps.alchemy.AlchemySpec]: the checked-in form placement.
- [`known_alchemy_forms`][param_sweeps.alchemy.known_alchemy_forms]: the SDK's form list.
- [`known_annotation_control_types`][param_sweeps.alchemy.known_annotation_control_types]: the
  SDK's detector list.
- [`AlchemyBackend`][param_sweeps.alchemy.AlchemyBackend]: the interface that a backend implements.
- [`HordelibAlchemyBackend`][param_sweeps.alchemy.HordelibAlchemyBackend]: the horde-engine backend.
- [`LiveHordeAlchemyBackend`][param_sweeps.alchemy.LiveHordeAlchemyBackend]: the public API backend.
- [`run_alchemy`][param_sweeps.alchemy.run_alchemy]: apply the forms and write everything.
- [`build_manifest`][param_sweeps.alchemy.build_manifest]: the manifest that the alchemy pages read.
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
"""Where an entry's source crop is written, inside the entry's directory.

Inside the entry rather than beside the shared source, because two entries can crop the same
source at different regions and one file beside the source could only hold one of them.
"""

RESULT_CROP_FILENAME = "result_crop.webp"
"""Where an entry's result crop is written, inside the entry's directory."""

RESULT_FIT_FILENAME = "result_fit.webp"
"""Where an entry's bounded copy of its result is written, inside the entry's directory.

An enlarger produces a result at four times the source size, and a page that draws the whole file
downloads several megabytes and discards most of them. The full result stays published beside this copy.
"""

RESULT_URL_SCHEME_PREFIX = "http"
"""The prefix of a picture result, because the API returns a URL instead of the bytes."""

ANSWER_LABEL_FIELDS = ("text", "hex")
"""The field names that a listed result uses for its item: a phrase matched by CLIP, or a palette colour."""

ANSWER_WEIGHT_FIELDS = ("confidence", "proportion")
"""The field names that a listed result uses for its weight: a confidence, or a share of the picture."""

UNAVAILABLE_NOTE_PREFIX = "No result was produced here:"
"""The prefix of a manifest note that gives the reason a form could not be run."""

EMPTY_FORM_MEMBER = "_NONE"
"""The placeholder member of the SDK's form enum, which is not a form and is not published."""

ANNOTATION_FORM = "annotation"
"""The one form that has a control type, because the detector determines the result."""

CONTROL_MAP_ELSEWHERE_REASON = (
    "the control maps are drawn by horde-image-utilities, a separate service run by an alchemist"
    " worker, which this tool uses with --backend hiu"
)
"""Why a backend that is not the image-utilities one publishes an annotation entry as a gap."""

BETA_CATEGORIES_VARIABLE = "HORDELIB_BETA_MODEL_CATEGORIES"
BETA_API_KEY_VARIABLE = "HORDELIB_BETA_MODELS_API_KEY"
BETA_PRIMARY_API_URL_VARIABLE = "HORDE_MODEL_REFERENCE_PRIMARY_API_URL"
"""The three horde-engine values that determine whether the pending model reference is used."""

BETA_MODEL_CATEGORIES = ("gfpgan",)
"""The reference categories covered by the beta opt-in. Only the face restorers are covered.

GFPGANv1.3 and RestoreFormer are in the shared vocabulary but exist only in the pending queue,
so without this the fix-faces comparison is missing two of its four entries. The upscalers have
a longer list of gaps, and are not covered until each one is worth publishing.
"""

BETA_PRIMARY_API_URL = "https://models.aihorde.net/api"
"""The model reference service that the pending records are read from."""

ANONYMOUS_API_KEY = "0000000000"
"""The anonymous key, which is reader-level access and is all that the pending queue requires."""


class AlchemyUseCase(StrEnum):
    """The reader's goal on the alchemy pages, used to group the forms.

    The horde's vocabulary is a flat list of processors. A reader does not arrive wanting
    ``RealESRGAN_x4plus``. They arrive wanting a bigger picture. Each form goes in exactly one
    of these. A form that does not fit any of them goes in the closest one, and its notes explain
    the mismatch.
    """

    DESCRIBE = "describe"
    """Describe the image: captioning, interrogation, and the other read-only forms."""

    CONTROL_MAP = "control_map"
    """Read the structure out of a picture and draw it: edges, depth, pose, regions, tone."""

    ENLARGE = "enlarge"
    """Make the picture bigger than it was rendered."""

    FIX_FACES = "fix_faces"
    """Restore a face that the sampler left soft."""

    CUT_OUT = "cut_out"
    """Take the subject off its background."""


class AlchemyFormKind(StrEnum):
    """How a form is executed, which determines what a backend does with it."""

    UPSCALE = "upscale"
    FACEFIX = "facefix"
    STRIP_BACKGROUND = "strip_background"
    UNSUPPORTED = "unsupported"
    """A form that none of this tool's backends can run, published as a gap."""


FROZEN_MODEL_CONFIG = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())


def known_alchemy_forms() -> tuple[str, ...]:
    """Return every alchemy form in the shared vocabulary, in declaration order.

    Read from ``horde_sdk`` rather than restated here, so the specification is checked against
    the horde's current vocabulary instead of a copy that can go stale.
    """
    return tuple(
        str(form.value) for name, form in KNOWN_ALCHEMY_TYPES.__members__.items() if name != EMPTY_FORM_MEMBER
    )


def known_annotation_control_types() -> tuple[str, ...]:
    """Return every control map in the shared vocabulary, in declaration order.

    Read from ``horde_sdk`` for the same reason as the form list. If the horde adds a detector,
    loading the specification fails, so the detector cannot go missing from the pages unnoticed.
    """
    return tuple(str(control_type.value) for control_type in KNOWN_ANNOTATION_CONTROL_TYPES)


def classify_form(form: str) -> AlchemyFormKind:
    """Return how this form is executed, using the SDK's membership tests.

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
    """Return the filename-safe token used to name a form's directory and strip stop."""
    return value_label(form)


def _check_placed_once(*, placed: list[str], known: list[str], subject: str) -> None:
    """Validate that a placement covers a shared vocabulary exactly, and report any mismatch.

    Args:
        placed: The names placed by the specification.
        known: The names in the shared vocabulary.
        subject: The name of what is being placed, for the error message.

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
    """Which detector runs, for the ``annotation`` form, whose result depends on the detector.

    For every other form the form name identifies the result, so this is unset for them.
    """

    crop: CropRegion | None = None
    """The region published at its own pixels, cut from both the source and the result.

    Set it when the form's effect is in detail too fine for a page to show at the picture's
    display size: a face that a restorer redrew, or the texture that an upscaler invented.
    """

    notes: str | None = None
    """Notes for the consumer about this entry, beyond its result.

    The notes record when a use case is only approximated, and the origin of a source that is not
    one of the three world anchors.
    """

    @property
    def label(self) -> str:
        """Return the filename-safe token used to name this entry's directory."""
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

        A form that is in the SDK but not placed in this file would be missing from the published
        evidence without a warning. A form placed twice would give the consumer two results for
        the same form. Both are errors when the file is read, instead of surprises at render time.
        The ``annotation`` form follows the same rule for its control types. It is placed once
        per control type in the SDK, because each detector gives a different result for the same
        picture.

        Raises:
            ValueError: If a form or a control type is placed twice, is not placed at all, or is
                not in the SDK.
        """
        annotation_entries = [entry for entry in self.entries if entry.form == ANNOTATION_FORM]
        forms_placed = [entry.form for entry in self.entries if entry.form != ANNOTATION_FORM]
        if annotation_entries:
            forms_placed.append(ANNOTATION_FORM)

        _check_placed_once(placed=forms_placed, known=list(known_alchemy_forms()), subject="alchemy forms")

        with_control_type = {entry.form for entry in self.entries if entry.control_type is not None}
        without_control_type = {entry.form for entry in self.entries if entry.control_type is None}
        misplaced = sorted((with_control_type - {ANNOTATION_FORM}) | (without_control_type & {ANNOTATION_FORM}))
        if misplaced:
            raise ValueError(
                f"only the {ANNOTATION_FORM} form has a control type, and every {ANNOTATION_FORM} entry"
                f" needs one, but these forms break that rule: {', '.join(misplaced)}",
            )

        _check_placed_once(
            placed=[entry.control_type for entry in self.entries if entry.control_type is not None],
            known=list(known_annotation_control_types()),
            subject="annotation control types",
        )

        return self

    def entry_for_form(self, form: str) -> AlchemyEntrySpec:
        """Return the entry that places this form.

        The ``annotation`` form is placed once per control type, so this returns the first of its
        entries. Look up by control type instead when it matters which detector is meant.

        Raises:
            KeyError: If the specification does not place this form.
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
    """The image the form produced, for the forms that produce pixels."""

    result_text: str | None = None
    """The file with the form's text output, for the describe family, which produces words."""

    result_json: str | None = None
    """The same result in the API's JSON shape, for a consumer that needs the fields instead of the lines."""


class ManifestEntry(BaseModel):
    """Represents one form's entry in the alchemy manifest."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    use_case: AlchemyUseCase
    form: str
    title: str
    files: ManifestFiles
    control_type: str | None = None
    """Which detector drew the map, for the ``annotation`` form. Unset for the other forms."""

    model: str | None = None
    """The weights that the form ran on, when the backend reports them."""

    source_width: int | None = None
    """The source image's width in pixels, read from the published file."""

    source_height: int | None = None
    """The source image's height in pixels, read from the published file."""

    result_width: int | None = None
    """The result's width in pixels, or None where the form produced words or vectors.

    For an upscaler, the result is this width compared with the source's. A page that cannot
    state that it shows four times the pixels appears to show the reader the same picture twice.
    """

    result_height: int | None = None
    """The result's height in pixels, or None where there is no raster result to measure."""

    source_crop: PublishedCrop | None = None
    """The part of the source published at its own pixels, or None when there is no crop."""

    result_crop: PublishedCrop | None = None
    """The part of the result published at its own pixels, or None when there is no crop."""

    result_fit: str | None = None
    """The result scaled down to the page-sized bound, or None when there is no such copy.

    A result already within the bound does not need a copy, and a form that produces words or
    vectors has no raster to scale, so this file is usually absent. Its absence is not a fault.
    """

    notes: str | None = None


class AlchemyManifest(BaseModel):
    """Represents the whole manifest at the root of the alchemy evidence tree."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    generated_against: str
    entries: list[ManifestEntry]


class AlchemyOutcome(BaseModel):
    """Represents a backend's output for one form applied to one image.

    Exactly one of the three results is set. An unavailable form is an outcome, not an exception,
    because the published evidence records that this machine cannot run it and the run continues.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, protected_namespaces=())

    image: Image.Image | None = None
    text: str | None = None
    structured: dict[str, Any] | None = None
    """The result in the API's shape, kept beside the text that a page reads, for a textual result."""

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
        """Apply one form to one image, and return the outcome.

        Args:
            form: The form to apply.
            control_type: Which detector to run, for the ``annotation`` form, and None otherwise.
            source_image: The picture to apply it to.
        """

    @property
    def service_host(self) -> str | None:
        """Return the host of the service that served the request, for a service outside this process.

        The default is None, because a backend running in this process has no host.
        """
        return None

    @property
    def service_version(self) -> str | None:
        """Return the version reported by the service, when it reports one."""
        return None

    def close(self) -> None:
        """Release whatever the backend is holding.

        The default does not release anything, so a backend that holds nothing does not have to
        override it. It is not abstract for that reason.
        """
        return


class HordelibAlchemyBackend(AlchemyBackend):
    """Applies alchemy forms through horde-engine on this machine's GPU.

    This is the same inference stack that an AI Horde alchemist worker runs, so its results match
    the horde's. ``hordelib`` is imported inside the methods that need it, for the same reason as
    in [`LocalRunner`][param_sweeps.local_runner.LocalRunner]: importing it starts a ComfyUI
    environment check and pulls in torch.

    Only the post-processing forms are available in horde-engine. The describe family, the NSFW
    detector and the newer image-utility forms are served by the alchemist's backend, which is a
    different stack. They are reported unavailable instead of approximated with some other tool
    that happens to be installed.
    """

    POST_PROCESSING_MODEL_MANAGERS = ("esrgan", "gfpgan", "codeformer", "miscellaneous")
    """The managers that own the upscaler and face-restorer weights, plus the miscellaneous one.

    The miscellaneous manager is loaded for the same reason as in the sweep runner: several
    processors resolve their weights through it and do not find a model folder without it.
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
        have to be in place before ``load_model_managers``. A variable that is already set is not
        changed, so an environment's own key or service takes precedence over this tool's
        defaults.
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
                    f"the {form} form is served by the alchemist backend, and this backend only runs"
                    " horde-engine forms"
                ),
            )

        try:
            return self._applied(horde=self._horde, kind=kind, form=form, source_image=source_image)
        except RuntimeError as backend_error:
            # A form that is in the shared vocabulary but not in the live model reference has no
            # weights to resolve. That is published as a gap, not as a failed run.
            logger.warning("Form unavailable: form=%s reason=%s", form, backend_error)
            return AlchemyOutcome(unavailable_reason=str(backend_error))
        except ImportError as missing_dependency:
            logger.warning("Form unavailable: form=%s reason=%s", form, missing_dependency)
            return AlchemyOutcome(unavailable_reason=str(missing_dependency))

    @staticmethod
    def _applied(*, horde: Any, kind: AlchemyFormKind, form: str, source_image: Image.Image) -> AlchemyOutcome:
        """Return horde-engine's result for a form of a kind that horde-engine runs.

        Raises:
            RuntimeError: If the backend did not produce an image.
        """
        if kind is AlchemyFormKind.STRIP_BACKGROUND:
            from hordelib.utils.image_utils import ImageUtils

            return AlchemyOutcome(image=ImageUtils.strip_background(source_image), model_name_used=form)

        payload = {"model": form, "source_image": source_image}
        result = horde.image_upscale(payload) if kind is AlchemyFormKind.UPSCALE else horde.image_facefix(payload)
        if result.image is None:
            raise RuntimeError(f"horde-engine did not return an image for the {form} form")

        return AlchemyOutcome(image=result.image, model_name_used=form)

    def close(self) -> None:
        """Drop the engine handle."""
        self._horde = None


class LiveHordeAlchemyBackend(AlchemyBackend):
    """Applies alchemy forms by submitting them to the live AI Horde, so the results are the horde's.

    The describe family is not available in horde-engine. Captioning, interrogation, the NSFW
    check and the newer read-only forms are served by the alchemist worker's backend. The only way
    to publish their real output is to submit the image to the public API and print the result.
    This backend submits one form per request to ``/interrogate/async``, and polls it until it has
    a result, in the same way as a reader's own request.

    A form that is not in the deployed API's request schema is reported unavailable, with the
    API's error text about what it accepts. A transport failure is not reported as unavailable. A
    timeout or a dropped connection is a problem with this machine's network, so publishing it as
    a gap would put a false claim about the horde's offering into the manifest.
    """

    def __init__(self, *, client: HordeApiClient) -> None:
        """Configure a live backend.

        Args:
            client: The API client to submit through. The caller that opened it also closes it.
        """
        self._client = client

    @property
    @override
    def name(self) -> str:
        """Return the backend's name, as recorded in every sidecar it produces."""
        return f"alchemy:{self._client.api_base_url}"

    @override
    def prepare(self) -> None:
        """Skip setup, because the horde holds the weights and the client holds the session."""
        return

    @override
    def apply(self, *, form: str, control_type: str | None, source_image: Image.Image) -> AlchemyOutcome:
        """Submit one form to the horde, and return the outcome."""
        if control_type is not None:
            return AlchemyOutcome(unavailable_reason=CONTROL_MAP_ELSEWHERE_REASON)

        try:
            completed = self._client.interrogate(form=form, source_image_base64=encoded_for_api(source_image))
        except FormRefusedError as refusal:
            logger.warning("Form unavailable: form=%s reason=%s", form, refusal)
            return AlchemyOutcome(unavailable_reason=str(refusal))

        if not completed.result:
            return AlchemyOutcome(unavailable_reason=f"the horde ran the {form} form and did not return a result")

        form_result = completed.result.get(form, completed.result)
        if isinstance(form_result, str) and form_result.startswith(RESULT_URL_SCHEME_PREFIX):
            return AlchemyOutcome(image=self._client.download_image(form_result), model_name_used=form)
        if isinstance(form_result, str) and form_result.lstrip().startswith(("<?xml", "<svg")):
            # The vectorizer returns the SVG markup. It is a picture, so it is published as a
            # result that a browser can draw, not as text.
            return AlchemyOutcome(svg=form_result)

        # The worker chooses the weights and the response does not include them, so the manifest
        # leaves the model unset instead of repeating the form name as if it were a model.
        return AlchemyOutcome(
            text=render_form_answer(form=form, result=completed.result),
            structured=dict(completed.result),
        )


def encoded_for_api(image: Image.Image) -> str:
    """Return this image as a base64 string for the API's ``source_image`` field."""
    buffer = io.BytesIO()
    image.save(buffer, format="WEBP", quality=WebpEncoding.QUALITY, method=WebpEncoding.METHOD)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def render_form_answer(*, form: str, result: Mapping[str, Any]) -> str:
    """Return a form's result as the text that the evidence page shows verbatim.

    The API wraps every result in a key named after the form. Inside it is a sentence for the
    captioner, a single value for the NSFW check and the aesthetic score, and a mapping for the
    forms that return data. This function does not summarise. The page presents the text as the
    horde's result, so it holds the numbers that the horde returned.

    Args:
        form: The form that the result came from.
        result: The mapping that the API returned for it.

    Returns:
        The result, as lines of text.
    """
    form_result = result.get(form, result)
    if isinstance(form_result, str):
        return form_result
    if isinstance(form_result, Mapping):
        return "\n".join(_result_lines(form_result))
    return f"{form}: {_scalar(form_result)}"


def _result_lines(form_result: Mapping[str, Any]) -> Iterator[str]:
    """Yield one line per field of a result returned as a mapping."""
    for field_name, value in form_result.items():
        if isinstance(value, list):
            yield f"{field_name}: {', '.join(_rendered_item(item) for item in value)}"
        else:
            yield f"{field_name}: {_scalar(value)}"


def _rendered_item(item: Any) -> str:
    """Return one item of a listed result: its label and its weight."""
    if not isinstance(item, Mapping):
        return _scalar(item)

    label = next((str(item[field_name]) for field_name in ANSWER_LABEL_FIELDS if field_name in item), None)
    weight = next((item[field_name] for field_name in ANSWER_WEIGHT_FIELDS if field_name in item), None)
    if label is None:
        return ", ".join(f"{field_name}={_scalar(value)}" for field_name, value in item.items())

    return f"{label} ({_scalar(round(weight, 4))})" if isinstance(weight, int | float) else label


def _scalar(value: Any) -> str:
    """Return one value as it appears in the result's JSON, so a boolean prints as ``true`` or ``false``."""
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

    Named after the input file instead of the entry, so two forms with the same source share one
    published copy, and a reader comparing their results can see that they share it.
    """
    return f"{SOURCES_DIRECTORY_NAME}/{PurePosixPath(entry.source).name}"


def published_source_crop_name(entry: AlchemyEntrySpec) -> str:
    """Return where this entry's source crop is published, relative to the alchemy tree."""
    return f"{entry.relative_directory}/{SOURCE_CROP_FILENAME}"


def published_result_crop_name(entry: AlchemyEntrySpec) -> str:
    """Return where this entry's result crop is published, relative to the alchemy tree."""
    return f"{entry.relative_directory}/{RESULT_CROP_FILENAME}"


def published_result_fit_name(entry: AlchemyEntrySpec) -> str:
    """Return where this entry's bounded copy of its result is published, relative to the alchemy tree."""
    return f"{entry.relative_directory}/{RESULT_FIT_FILENAME}"


def build_manifest(*, spec: AlchemySpec, output_root: Path) -> AlchemyManifest:
    """Build the manifest from the specification and the files actually on disk.

    Every placed form is listed whether or not it produced anything. An entry without a result
    records that the form could not be run here, and its ``notes`` give the reason. Leaving it
    out would suggest to the consumer that the form does not exist.

    Args:
        spec: The alchemy specification.
        output_root: The root of the sweep library, which contains the alchemy tree.

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
                result_fit=_published_result_fit(root=root, entry=entry, result_name=published_result),
                notes=_manifest_notes(entry=entry, record=record),
            ),
        )

    return AlchemyManifest(generated_against=spec.generated_against, entries=entries)


def write_manifest(*, spec: AlchemySpec, output_root: Path) -> Path:
    """Write the alchemy manifest, and return its path.

    Side Effects:
        Overwrites any existing manifest. The content depends only on the specification and the
        files present, so rewriting it does not introduce spurious changes.
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
        ValueError: If the sweep specification does not describe exactly one. A single strip
            lets one shared source serve the whole comparison.
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

    An entry whose result is already on disk is skipped unless ``force`` is set. That is enough
    to resume a run: an alchemy form is deterministic for a given source, so the file on disk
    records the same thing as a checkpoint entry.

    Args:
        spec: The alchemy specification.
        sweep_spec: The sweep specification, which contains the upscaler-form strip.
        backend: The backend to apply the forms through, unprepared.
        output_root: The root of the sweep library.
        forms: Which forms to apply. Empty means every form in the specification.
        control_types: Which control maps to draw. Empty means every control map. Any value
            restricts the run to the annotation entries that draw those maps.
        force: Apply selected forms even when their result is already on disk.

    Returns:
        One result per selected form.

    Raises:
        RuntimeError: If a selected entry's source image is not at the path in the specification.
        ValueError: If the sweep specification does not contain exactly one upscaler-form strip.

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
    """Return whether this entry is in the run that the filters describe.

    A control type restricts the run to the annotation entries that draw it, so
    ``--control-type canny`` draws one map and does not apply every other form.
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
        RuntimeError: If the entry's source image is not at the path in the specification.
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

    A form without a stop in the strip is not written here, so the strip compares only the
    upscalers and not every form in the manifest.
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
    """The host of the service that produced the result, for a service outside this process."""

    service_version: str | None = None


def _write_entry_record(
    *,
    root: Path,
    entry: AlchemyEntrySpec,
    outcome: AlchemyOutcome,
    backend: AlchemyBackend,
) -> None:
    """Write the sidecar that records the backend's outcome for this form."""
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

    A form can produce words or vectors, and an entry can have no result at all, so this
    reports a missing size instead of treating it as a fault.
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
    """Return one side's crop record when it is on disk, or None when the entry does not declare a crop."""
    if region is None:
        return None

    return published_crop(file=file, root=root, region=region, frame_relative_path=frame_name)


def _published_result_fit(*, root: Path, entry: AlchemyEntrySpec, result_name: str | None) -> str | None:
    """Return the bounded copy of the result when it is on disk, or None.

    Only a raster result has a bounded copy, because an SVG or a line of text does not need one.
    """
    raster_name = f"{entry.relative_directory}/{RESULT_IMAGE_FILENAME}"
    if result_name != raster_name:
        return None

    fit_name = published_result_fit_name(entry)
    return fit_name if (root / fit_name).is_file() else None


def _manifest_notes(*, entry: AlchemyEntrySpec, record: _EntryRecord | None) -> str | None:
    """Return the manifest notes for this entry.

    The specification's note comes first, because it is about the form's placement and applies
    whether or not the form ran. An unavailability reason is appended to it instead of replacing
    it, so a reader gets both the caveat and the gap. The reason starts as a separate sentence,
    so the two do not run together.
    """
    reason = record.unavailable_reason if record is not None else None
    gap = f"{UNAVAILABLE_NOTE_PREFIX} {reason}" if reason else None
    parts = [part for part in (entry.notes, gap) if part]
    return " ".join(parts) if parts else None


def _already_applied(*, root: Path, entry: AlchemyEntrySpec) -> bool:
    """Return whether this entry already has a result on disk."""
    directory = root / entry.relative_directory
    return any(
        (directory / name).is_file() for name in (RESULT_IMAGE_FILENAME, RESULT_SVG_FILENAME, RESULT_TEXT_FILENAME)
    )


def _clear_stale_results(*, root: Path, entry: AlchemyEntrySpec) -> None:
    """Remove a result left from an earlier run in which the form was still available.

    Otherwise a manifest built from the files on disk would list a result for an entry that
    reports that the form could not be run.
    """
    directory = root / entry.relative_directory
    for filename in (RESULT_IMAGE_FILENAME, RESULT_SVG_FILENAME, RESULT_TEXT_FILENAME, RESULT_JSON_FILENAME):
        (directory / filename).unlink(missing_ok=True)


def _publish_source(*, entry: AlchemyEntrySpec, root: Path) -> None:
    """Copy this entry's source into the alchemy tree, so the published tree is self-contained.

    Raises:
        RuntimeError: If the source is not at the path in the specification.
    """
    source_path = _source_path(entry)
    destination = root / published_source_name(entry)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, destination)


def _source_path(entry: AlchemyEntrySpec) -> Path:
    """Return this entry's source image path.

    Raises:
        RuntimeError: If the source is not at the path in the specification.
    """
    source_path = TOOL_ROOT / entry.source
    if not source_path.is_file():
        raise RuntimeError(f"the source for the {entry.form} form is missing: {source_path}")
    return source_path


def _source_image(entry: AlchemyEntrySpec) -> Image.Image:
    """Return this entry's source image, decoded.

    Raises:
        RuntimeError: If the source is not at the path in the specification.
    """
    with Image.open(_source_path(entry)) as opened:
        return opened.convert("RGB")


def use_case_counts(spec: AlchemySpec) -> dict[AlchemyUseCase, int]:
    """Return the number of forms in each use case, for the command-line report."""
    counts: dict[AlchemyUseCase, int] = dict.fromkeys(AlchemyUseCase, 0)
    for entry in spec.entries:
        counts[entry.use_case] += 1
    return counts
