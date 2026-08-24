# SPDX-FileCopyrightText: 2026 Tazlin
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Parse the sweep specification and expand it into the strips and jobs a runner renders.

The specification is the single source of truth for the whole library: the anchor parameters
every sweep holds constant, the models and scenes, and each sweep's stops with the teaching
caption that belongs to the stop. Expansion is pure and deterministic, so the same
specification always produces the same strips, in the same order, with the same filenames,
whichever runner renders them.

Critical public members:

- [`SweepSpec`][param_sweeps.spec.SweepSpec]: the whole specification.
- [`RenderStrip`][param_sweeps.spec.RenderStrip]: one model/scene/parameter strip and its jobs.
- [`RenderJob`][param_sweeps.spec.RenderJob]: one fully resolved image to render.
- [`BatchSpec`][param_sweeps.spec.BatchSpec]: a section of one-off jobs a human signs off, which
  expands to one-stop strips so it travels the same rails a teaching strip does.
- [`RulingSet`][param_sweeps.spec.RulingSet]: the questions a reviewer has to answer about what
  the batch shows, each one pointing at the images that decide it.
- [`FeatureOverrides`][param_sweeps.spec.FeatureOverrides]: the generation features beyond the
  plain sampler parameters, statable per stop, per sweep and per batch job.
- [`load_spec`][param_sweeps.spec.load_spec]: read and validate a specification file.
"""

from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal

from horde_sdk.generation_parameters.image.consts import (
    KNOWN_IMAGE_SAMPLERS,
    KNOWN_IMAGE_SCHEDULERS,
)
from pydantic import BaseModel, ConfigDict, Field, model_validator
from strenum import StrEnum

type SamplerName = KNOWN_IMAGE_SAMPLERS | str
"""A sampler, named from the shared vocabulary but accepting any string.

The specification is data a human writes, and the horde accepts sampler names this vocabulary has
not caught up with yet, so an unrecognised name is carried through to the backend to answer for
rather than rejected while reading the file. A recognised one validates into the enum, which is
what makes a typo in a familiar name visible here instead of in a render.
"""

type SchedulerName = KNOWN_IMAGE_SCHEDULERS | str
"""A sigma schedule, named from the shared vocabulary but accepting any string, as above."""

type SolverTypeName = str
"""A sampler's `solver_type`, whose vocabulary differs per sampler family.

Left as a plain string on purpose: `dpmpp_2m_sde` takes `midpoint` or `heun` while `seeds_2` and
the `exp_heun_2_x0` pair take `phi_1` or `phi_2`, so the value is only meaningful against the
sampler beside it, which the constraints table checks and this file cannot.
"""

# `protected_namespaces` is cleared because these models describe image-generation *models*, and
# pydantic otherwise warns about every `model_`-prefixed field name as a namespace collision.
# `populate_by_name` is set because a few fields carry the horde payload's own short name as an
# alias (``n``, ``chosen``) while reading as themselves in Python.
FROZEN_MODEL_CONFIG = ConfigDict(frozen=True, extra="forbid", protected_namespaces=(), populate_by_name=True)

SPEC_SCHEMA_VERSION = 1
"""The specification layout this module can read."""

MAX_BATCH_SIZE = 20
"""The most candidates one curation batch may render, which is the horde's own ceiling."""

TOOL_ROOT = Path(__file__).resolve().parents[1]
"""The tool's own directory: what a sweep's ``source_image`` path is relative to."""

NON_LABEL_CHARACTERS = re.compile(r"[^a-z0-9_]+")


class SweepParameter(StrEnum):
    """The generation parameter a strip teaches."""

    STEPS = "steps"
    STEPS_CONVERGING = "steps_converging"
    CFG_SCALE = "cfg_scale"
    SAMPLER = "sampler"
    SAMPLER_GALLERY = "sampler_gallery"
    """Every sampler the horde publishes, one stop each, on one seed and scene. The sampler strip
    teaches the four a reader meets first; this one lets the rest be seen beside them."""
    CLIP_SKIP = "clip_skip"
    HIRES_FIX = "hires_fix"
    HIRES_FIX_DENOISING_STRENGTH = "hires_fix_denoising_strength"
    """How much of the enlarged frame the second pass redraws.

    Not the strength the first pass runs at: this one is the enlargement's own, and the backend
    derives the second pass's step count from it as well, so the strip is a reading of the pass
    rather than of one number in isolation.
    """

    TILING = "tiling"
    STEPS_CFG_GRID = "steps_cfg_grid"
    SAMPLER_STEPS_GRID = "sampler_steps_grid"
    HIRES_GRID = "hires_grid"
    KARRAS_GRID = "karras_grid"
    RESOLUTION = "resolution"
    DENOISING_STRENGTH = "denoising_strength"
    MASKED_DENOISING_STRENGTH = "masked_denoising_strength"
    """The same strength again with a mask over part of the source, where it still means what it
    means everywhere else: the backend keeps what was under the mask and redraws it by this much."""

    MASKING_GRID = "masking_grid"
    """A masked image-to-image column beside an inpainting one, read down a range of strengths.

    The two are different pipelines rather than two names for one, and this is where a reader can
    see which of them keeps what was already there.
    """

    OUTPAINT_EXTENT = "outpaint_extent"
    """How far a canvas was pushed out past the picture it started from.

    Not a field any payload carries: the caller pads the source and masks the new region itself,
    so what varies between these stops is the picture and the mask that were sent.
    """

    SOURCE_PROCESSING = "source_processing"
    """Not a parameter a strip teaches: the axis a grid lays its columns out along."""

    DENOISING_STRENGTH_CONVERGING = "denoising_strength_converging"
    """The denoising strip again on a sampler that settles, so a reader can see how much of what a
    strength does at each stop is the strength and how much is the ancestral sampler re-rolling."""
    NEGATIVE_PROMPT = "negative_prompt"
    PROMPT_EMPHASIS = "prompt_emphasis"
    SEED = "seed"
    KARRAS = "karras"
    SCHEDULER = "scheduler"
    SAMPLER_ETA = "sampler_eta"
    SAMPLER_S_NOISE = "sampler_s_noise"
    SAMPLER_S_CHURN = "sampler_s_churn"
    SAMPLER_SOLVER_TYPE = "sampler_solver_type"
    SAMPLER_ORDER = "sampler_order"
    FLOW_SHIFT = "flow_shift"
    CONTROL_STRENGTH = "control_strength"
    """How hard a ControlNet holds the render to its control image, which on the QR workflow is
    the dial between a picture that scans and a picture worth looking at."""

    CONTROLNET_GALLERY = "controlnet_gallery"
    """Every ControlNet guidance mode the horde publishes, one stop each, on one source and seed.

    The sampler gallery's counterpart for the other long vocabulary a reader has to choose from.
    A stop is three pictures rather than one: the source, the map the detector drew from it, and
    the render that map guided, because the map is what a reader is actually choosing between.
    """

    QR_STYLE = "qr_style"
    """Not a parameter a strip teaches: finished QR artworks at one strength, a treatment each."""

    SCHEDULER_STEPS_GRID = "scheduler_steps_grid"
    STEP_PREVIEW = "step_preview"
    """Not a parameter either: the canvas of one render, stop by stop, captured as it sampled."""

    STEP_PREVIEW_GUESS = "step_preview_guess"
    """The same render's own guess at the finished picture, stop by stop."""

    UPSCALER_FORM = "upscaler_form"
    """Not a parameter either: one image, put through each of the horde's upscalers in turn.

    Nothing is generated for these stops. They are alchemy applied to an image that already
    exists, so ``alchemy`` produces them and the campaign never selects them.
    """

    BATCH = "batch"
    """Not a parameter a strip teaches: what a batch job's images are filed under."""

    @property
    def directory_name(self) -> str:
        """Return the on-disk directory name for this parameter."""
        return str(self.value).replace("_", "-")


GRID_AXIS_STOP_FIELDS: dict[SweepParameter, tuple[str, ...]] = {
    SweepParameter.STEPS: ("steps",),
    SweepParameter.CFG_SCALE: ("cfg_scale",),
    SweepParameter.SAMPLER: ("sampler_name",),
    SweepParameter.CLIP_SKIP: ("clip_skip",),
    SweepParameter.HIRES_FIX: ("hires_fix",),
    SweepParameter.KARRAS: ("karras",),
    SweepParameter.SCHEDULER: ("scheduler",),
    SweepParameter.SAMPLER_ETA: ("sampler_eta",),
    SweepParameter.SAMPLER_S_CHURN: ("sampler_s_churn",),
    SweepParameter.RESOLUTION: ("width", "height"),
    SweepParameter.DENOISING_STRENGTH: ("denoising_strength",),
    SweepParameter.SOURCE_PROCESSING: ("source_processing",),
}
"""Which stop fields each parameter a grid can be laid out along writes.

``RESOLUTION`` writes two, because one square size is what a reader is choosing between when
they pick a resolution, and a grid axis holds one value per row.
"""

GRID_AXIS_LABEL_PREFIXES: dict[SweepParameter, str] = {
    SweepParameter.STEPS: "s",
    SweepParameter.CFG_SCALE: "c",
    SweepParameter.SAMPLER: "",
    SweepParameter.CLIP_SKIP: "cs",
    SweepParameter.HIRES_FIX: "",
    SweepParameter.KARRAS: "k",
    SweepParameter.SCHEDULER: "",
    SweepParameter.SAMPLER_ETA: "eta",
    SweepParameter.SAMPLER_S_CHURN: "churn",
    SweepParameter.RESOLUTION: "r",
    SweepParameter.DENOISING_STRENGTH: "d",
    SweepParameter.SOURCE_PROCESSING: "",
}
"""The filename token each axis contributes, so a cell's name says which axes it sits on."""


class StripPresentation(StrEnum):
    """How a web app should lay a strip out.

    ``TILE_REPEAT_3X3`` asks the consumer to draw each image repeated nine times in a square
    alongside the single image itself, which is the only way the difference a seamless-tiling
    flag makes is visible at all: the odd grid puts whole joins in the interior rather than only
    at the crossing point, and the isolated tile says what was actually rendered.

    ``TRIPTYCH`` asks for three pictures per stop: the strip's source, the stop's own control map,
    and the render. A reader choosing a detector is choosing what the map keeps, so the map has to
    be on the page beside the render it produced rather than described in words.
    """

    STRIP = "strip"
    GRID = "grid"
    TILE_REPEAT_3X3 = "tile_repeat_3x3"
    TRIPTYCH = "triptych"


class AnchorParameters(BaseModel):
    """Represents the values every sweep holds constant unless it is the one being swept."""

    model_config = FROZEN_MODEL_CONFIG

    sampler_name: SamplerName
    cfg_scale: float
    steps: int
    karras: bool
    denoising_strength: float
    hires_fix: bool
    tiling: bool

    scheduler: SchedulerName | None = None
    """The sigma schedule every sweep runs on. Unset leaves the schedule to the karras flag.

    Kept optional so a specification written before schedules were nameable still reads, and means
    what it always meant: karras when the flag is on, and the backend's plain schedule otherwise.
    """


class ModelSpec(BaseModel):
    """Represents one image model the library renders with."""

    model_config = FROZEN_MODEL_CONFIG

    slug: str
    horde_name: str
    baseline: str
    native_width: int
    native_height: int
    anchor_clip_skip: int
    approximate_download_size_gb: float
    note: str


class SceneSpec(BaseModel):
    """Represents one prompt the sweeps are rendered against."""

    model_config = FROZEN_MODEL_CONFIG

    scene_id: str
    label: str
    prompt: str
    derived_from: str


class SourceProcessing(StrEnum):
    """What a job does with the image it starts from."""

    IMG2IMG = "img2img"
    INPAINTING = "inpainting"
    OUTPAINTING = "outpainting"
    REMIX = "remix"


class WorkflowName(StrEnum):
    """A named pipeline the backend runs instead of the plain sampler graph."""

    QR_CODE = "qr_code"


QR_TEXT_REFERENCES = frozenset({"qr_code", "qr_text"})
"""The ``extra_texts`` references the QR workflow reads its payload from.

Both spellings are accepted because the two backends this tool drives have not always agreed:
the frontpage's own verified QR example and the web app both write ``qr_code``, while the
``hordelib`` recipe this batch was written from uses ``qr_text``. Accepting either here means a
specification is validated on what it is trying to say rather than on which backend wrote it.
"""


class ExtraText(BaseModel):
    """Represents one entry of ``extra_texts``: a string a workflow reads by its reference."""

    model_config = FROZEN_MODEL_CONFIG

    text: str
    reference: str


class ExtraSourceImage(BaseModel):
    """Represents one additional input image, as remix and multi-reference workflows take them."""

    model_config = FROZEN_MODEL_CONFIG

    image: str
    """The image's path, relative to the tool root."""

    strength: float = 1.0

    @property
    def absolute_path(self) -> Path:
        """Return the image's absolute path."""
        return TOOL_ROOT / self.image


class LoraReference(BaseModel):
    """Represents one LoRA a job applies, in the shape the horde payload carries."""

    model_config = FROZEN_MODEL_CONFIG

    name: str
    model: float = 1.0
    clip: float = 1.0
    inject_trigger: str | None = None
    is_version: bool = False


class TextualInversionReference(BaseModel):
    """Represents one textual inversion a job applies, in the shape the horde payload carries."""

    model_config = FROZEN_MODEL_CONFIG

    name: str
    inject_ti: str | None = None
    strength: float = 1.0


class FeatureOverrides(BaseModel):
    """Represents the generation features a job can turn on beyond the plain sampler parameters.

    These are the fields that select a different pipeline rather than a different value on the
    same one: a ControlNet, a named workflow, extra inputs, a mask, an alpha channel, adapters,
    post-processing, and rendering more than one candidate to curate from. Every field is
    optional and resolves the same way the parameter overrides do: the stop's, then the sweep's,
    then the documented default.
    """

    model_config = FROZEN_MODEL_CONFIG

    control_type: str | None = None
    control_strength: float | None = None
    """The ControlNet to condition on, such as ``canny`` or ``depth``."""

    image_is_control: bool | None = None
    """Whether the source image is already a control map rather than an image to derive one from."""

    return_control_map: bool | None = None
    """Whether the backend should return the derived control map instead of the render."""

    source_processing: SourceProcessing | None = None
    """What to do with the source image. Unset means plain image-to-image when there is one."""

    workflow: WorkflowName | None = None
    extra_texts: list[ExtraText] | None = None
    extra_source_images: list[ExtraSourceImage] | None = None
    source_image: str | None = None
    """The picture to start from, relative to the tool root, in place of the sweep's own.

    A stop states this only where the picture is what varies. Outpainting is the case that needs
    it: the backend never pads anything, so each extent is a canvas that was padded beforehand,
    and the stops of one strip do not share an input.
    """

    source_mask: str | None = None
    """The mask's path, relative to the tool root, for the processings that take one."""

    transparent: bool | None = None
    """Whether to render onto a real alpha channel."""

    loras: list[LoraReference] | None = None
    tis: list[TextualInversionReference] | None = None
    post_processing: list[str] | None = None
    facefixer_strength: float | None = Field(default=None, ge=0.0, le=1.0)
    """How strongly a face-fixing post-processor is blended back over the render."""

    batch_size: int | None = Field(default=None, ge=1, le=MAX_BATCH_SIZE, alias="n")
    """How many candidates to render for this job, when it is one a human curates.

    Written ``n`` in the specification, which is what the horde payload calls it.
    """

    chosen_candidate: int | None = Field(default=None, ge=1, alias="chosen")
    """Which candidate a human picked after review, one-based. Unset means the first one."""


class ParameterOverrides(FeatureOverrides):
    """Represents parameter values stated in place of the ones that would be inherited.

    Every field is optional. An unset field inherits the sweep's own overrides, then the anchor
    (or, for ``clip_skip`` and the dimensions, the model), so a stop states only what it changes
    and a sweep states only what it holds differently from the rest of the library.
    """

    model_config = FROZEN_MODEL_CONFIG

    steps: int | None = None
    cfg_scale: float | None = None
    sampler_name: SamplerName | None = None
    scheduler: SchedulerName | None = None
    clip_skip: int | None = None
    hires_fix: bool | None = None
    tiling: bool | None = None
    karras: bool | None = None
    denoising_strength: float | None = None
    seed: int | None = None

    hires_fix_denoising_strength: float | None = Field(default=None, ge=0.01, le=1.0)
    """What the second pass redraws by, when a job asks for the hires pass at all.

    Unset rather than anchored, like the solver knobs: an unset value is never sent, and the
    backend then runs the pass exactly as it does for every other strip in the library.
    """

    # Solver tuning. Each is unset unless a stop asks for it, and an unset knob is never sent, so a
    # strip that varies one of them differs from the rest of the library only in that one knob.
    sampler_eta: float | None = None
    sampler_s_noise: float | None = None
    sampler_s_churn: float | None = None
    sampler_solver_type: SolverTypeName | None = None
    sampler_order: int | None = None
    flow_shift: float | None = None

    use_negative_prompt: bool | None = None
    """Whether the sweep's negative prompt applies here. Unset means it does."""

    prompt_fragment: str | None = None
    """Text appended to the scene's prompt, which is how a strip can vary the prompt itself."""

    width: int | None = None
    height: int | None = None


class SweepStop(ParameterOverrides):
    """Represents one stop of a strip: the value that changes, and what it teaches."""

    value: bool | int | float | str
    caption: str
    label: str | None = None

    control_group: str | None = None
    """Which family this stop's control type belongs to, on the strips that walk a vocabulary."""

    @property
    def filename_label(self) -> str:
        """Return the filename-safe token identifying this stop."""
        if self.label is not None:
            return self.label
        return value_label(self.value)


class SweepBase(BaseModel):
    """Represents the fields shared by every kind of sweep."""

    model_config = FROZEN_MODEL_CONFIG

    parameter: SweepParameter
    rationale: str
    caption: str
    model_slugs: list[str]
    scene_ids: list[str]
    source_image: str | None = None
    """An image to start from, relative to the tool root. Unset renders from noise."""

    negative_prompt: str = ""
    """The negative prompt this sweep's stops can turn on. Empty means none of them do."""

    holds: ParameterOverrides = Field(default_factory=ParameterOverrides)
    """What this sweep holds differently from the anchor, for every stop that does not say."""

    directory_override: str | None = None
    """Where this sweep's strips are filed, in place of the parameter and model directories.

    The default layout gives one directory per parameter, model and scene, which two sweeps of
    the same parameter on the same model and scene would share. Stating a directory is how a set
    of such sweeps stays distinct: the ControlNet strength strips are one strength sweep per
    guidance mode, all on one model and one scene, and the mode is what tells them apart.
    """


class StripSweep(SweepBase):
    """Represents a one-dimensional sweep: a list of stops along a single parameter."""

    kind: Literal["strip"] = "strip"
    presentation: StripPresentation = StripPresentation.STRIP
    resolution_multiplier: float = 1.0
    stops: list[SweepStop]


class GridAxis(BaseModel):
    """Represents one axis of a two-dimensional sweep: the parameter it varies and its values."""

    model_config = FROZEN_MODEL_CONFIG

    parameter: SweepParameter
    values: list[bool | int | float | str]
    caption: str

    @model_validator(mode="after")
    def _check_axis_parameter_is_layable(self) -> GridAxis:
        """Validate that the axis varies a parameter a stop can state."""
        if self.parameter not in GRID_AXIS_STOP_FIELDS:
            layable = ", ".join(sorted(str(parameter) for parameter in GRID_AXIS_STOP_FIELDS))
            raise ValueError(f"a grid cannot be laid out along {self.parameter}; it can along: {layable}")
        return self

    def stop_fields(self) -> tuple[str, ...]:
        """Return the stop fields this axis writes."""
        return GRID_AXIS_STOP_FIELDS[self.parameter]

    def label_for(self, value: bool | int | float | str) -> str:
        """Return the filename token this axis contributes for one of its values."""
        prefix = GRID_AXIS_LABEL_PREFIXES[self.parameter]
        # Integers are padded so that a directory listing sorts the way the axis is declared.
        token = f"{value:02d}" if isinstance(value, int) and not isinstance(value, bool) else value_label(value)
        return f"{prefix}{token}"


class GridSweep(SweepBase):
    """Represents a two-dimensional sweep: one parameter down the rows, another across.

    The cells are generated rather than written out, so the axes cannot drift apart from the
    stops they describe. Rows are the outer loop, so the stops are in row-major order.
    """

    kind: Literal["grid"]
    rows: GridAxis
    columns: GridAxis
    cell_caption: str
    """A caption template for each cell, given ``{row}`` and ``{column}``."""


Sweep = Annotated[StripSweep | GridSweep, Field(discriminator="kind")]


class BatchSection(StrEnum):
    """The part of a review batch a job belongs to, which is how a contact sheet groups it."""

    WORLD_SCENES = "a-world-scenes"
    DIALECT_COMPARISON = "b-dialects"
    FEATURE_PROBES = "c-feature-probes"
    SUSPECT_CLAIMS = "d-suspect-claims"
    """Renders that exist to be ruled on: each one answers a claim nobody has confirmed by eye."""

    MINTED_GOLDENS = "e-minted-goldens"
    """The reference images this wave minted, re-rendered from their own parameters for review."""

    RAW_BACKEND = "f-raw-backend"
    """Images produced by driving the backend directly, for claims the rendering path itself alters."""


class BatchJob(ParameterOverrides):
    """Represents one image of a review batch: its own model, prompt and parameters.

    A batch is not a sweep. Nothing is held constant across it on purpose and no two jobs teach
    the same parameter, so each job states its model and its whole prompt rather than inheriting
    a scene, which is what lets a world's descriptor block stay byte-identical across the scenes
    it appears in: the text is written once per job and diffing two jobs shows what moved.
    """

    job_id: str
    model_slug: str
    prompt: str
    caption: str
    negative_prompt: str = ""
    source_image: str | None = None
    """The image this job starts from, relative to the tool root; unset renders from noise."""

    requires_source_from: str | None = None
    """The ``job_id`` whose chosen render is staged into ``source_image`` before this job runs."""


class BatchSpec(BaseModel):
    """Represents one section of a review batch: a list of jobs a human signs off together."""

    model_config = FROZEN_MODEL_CONFIG

    section: BatchSection
    title: str
    rationale: str
    holds: ParameterOverrides = Field(default_factory=ParameterOverrides)
    """What every job in this section inherits unless it states otherwise."""

    jobs: list[BatchJob]

    @model_validator(mode="after")
    def _check_job_ids_are_unique(self) -> BatchSpec:
        """Validate that no two jobs in this section would write to the same path."""
        job_ids = [job.job_id for job in self.jobs]
        if len(set(job_ids)) != len(job_ids):
            raise ValueError(f"job ids must be unique within section {self.section}")
        return self


class RulingControl(StrEnum):
    """How a ruling is answered, which is the form control the contact sheet renders for it."""

    RADIO = "radio"
    """One answer out of a set: the shape of a claim that is either affirmed or rejected."""

    CHECKBOX_SET = "checkbox-set"
    """Any number of answers out of a set, for a ruling that approves a list item by item."""

    ROW_SELECT = "row-select"
    """One answer per named row, all rows sharing the same options."""

    TEXT = "text"
    """No option set at all: the answer is whatever the reviewer writes."""


RULING_OTHER_OPTION_ID = "other-see-comments"
"""The escape every ruling that picks from a set carries, so no reviewer is forced into a bad fit.

It is appended by [`RulingItem.presented_options`][param_sweeps.spec.RulingItem.presented_options]
rather than written into the data, which is what makes it impossible for a ruling to be authored
without one. A specification may not declare an option under this identifier.
"""

RULING_OTHER_OPTION_LABEL = "Other / see comments"
"""What the escape option is called wherever it is rendered."""

RULING_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
"""What a ruling, option or row identifier may be spelled with.

Identifiers become element ids, form-control names and export keys, all of which are built by
concatenation, so they are held to one lowercase alphabet: an identifier that is safe in one of
those places is then safe in all of them. Underscores are allowed because a sampler or schedule
name is the honest identifier for a ruling about that sampler or schedule.
"""


class RulingChoice(BaseModel):
    """Represents one option a ruling offers."""

    model_config = FROZEN_MODEL_CONFIG

    choice_id: str
    label: str
    evidence_job_id: str | None = None
    """The batch job whose image decides this one option, linked beside it when set."""


class RulingRow(BaseModel):
    """Represents one line of a per-row ruling: a thing that needs its own answer."""

    model_config = FROZEN_MODEL_CONFIG

    row_id: str
    label: str
    evidence_job_id: str | None = None
    """The batch job whose image decides this row, linked beside it when set."""

    options: list[RulingChoice] = Field(default_factory=list)
    """This row's own options. Left empty, the row is answered from the ruling's shared set.

    Rows usually share one set, which is what makes a per-row ruling readable as a single
    question asked repeatedly. A row states its own only when the ruling is a short list of
    decisions that belong together administratively but are not the same question.
    """


class RulingItem(BaseModel):
    """Represents one question a reviewer has to answer before the wave it belongs to ships.

    A ruling states the question in the words it will be answered in, and points at the images
    that decide it, because the failure mode this exists to prevent is a claim shipping because
    nobody noticed it was never put to anyone. Every ruling carries a comment box and, unless it
    is answered in free text alone, an ``Other`` option: an answer that does not fit the offered
    set is a real answer and has to be recordable as one.
    """

    model_config = FROZEN_MODEL_CONFIG

    ruling_id: str
    title: str
    question: str
    """The precise thing being asked, in one or two sentences."""

    note: str = ""
    """Context the reviewer needs to answer, stated on the panel rather than remembered."""

    control: RulingControl
    evidence_sections: list[BatchSection] = Field(default_factory=list)
    evidence_job_ids: list[str] = Field(default_factory=list)
    options: list[RulingChoice] = Field(default_factory=list)
    rows: list[RulingRow] = Field(default_factory=list)
    """The things a ``row-select`` ruling answers one by one."""

    master_toggle_label: str | None = None
    """A control that ticks every box of a ``checkbox-set`` at once, when the set is long enough."""

    @model_validator(mode="after")
    def _check_the_control_has_what_it_needs(self) -> RulingItem:
        """Validate identifiers, and that the control is given exactly the parts it renders from.

        Raises:
            ValueError: If an identifier is malformed, duplicated or reserved, if a control is
                missing its options or rows, or if it was handed parts it cannot render.
        """
        self._check_identifier(self.ruling_id, "ruling id")
        self._check_option_ids(self.options)
        row_ids = [row.row_id for row in self.rows]
        if len(set(row_ids)) != len(row_ids):
            raise ValueError(f"ruling {self.ruling_id} reuses a row id")
        for row in self.rows:
            self._check_identifier(row.row_id, "row id")
            self._check_option_ids(row.options)

        if self.control is RulingControl.TEXT:
            if self.options or self.rows:
                raise ValueError(f"ruling {self.ruling_id} is answered in free text and can offer nothing to pick")
        elif self.control is RulingControl.ROW_SELECT:
            if not self.rows:
                raise ValueError(f"ruling {self.ruling_id} is a {RulingControl.ROW_SELECT} with no rows to answer for")
            for row in self.rows:
                if not row.options and not self.options:
                    raise ValueError(f"ruling {self.ruling_id} leaves row {row.row_id} with nothing to pick from")
        else:
            if not self.options:
                raise ValueError(f"ruling {self.ruling_id} is a {self.control} with nothing to pick from")
            if self.rows:
                raise ValueError(f"ruling {self.ruling_id} states rows, which only a {RulingControl.ROW_SELECT} has")

        if self.master_toggle_label is not None and self.control is not RulingControl.CHECKBOX_SET:
            raise ValueError(f"ruling {self.ruling_id} states a master toggle, which only a checkbox set has")

        return self

    def _check_identifier(self, identifier: str, what: str) -> None:
        """Validate one identifier's spelling.

        Raises:
            ValueError: If it is not a lowercase identifier the sheet can build element ids from.
        """
        if not RULING_IDENTIFIER.match(identifier):
            raise ValueError(f"{what} {identifier!r} is not a lowercase identifier")

    def _check_option_ids(self, options: list[RulingChoice]) -> None:
        """Validate one option set's identifiers.

        Raises:
            ValueError: If an option id repeats, is malformed, or takes the reserved escape's id.
        """
        option_ids = [option.choice_id for option in options]
        if len(set(option_ids)) != len(option_ids):
            raise ValueError(f"ruling {self.ruling_id} reuses an option id")
        for option_id in option_ids:
            self._check_identifier(option_id, f"ruling {self.ruling_id} option id")
            if option_id == RULING_OTHER_OPTION_ID:
                raise ValueError(f"ruling {self.ruling_id} declares {RULING_OTHER_OPTION_ID!r}, which is reserved")

    def presented_options(self, *, row: RulingRow | None = None) -> list[RulingChoice]:
        """Return the options as the sheet offers them: the stated set, then the ``Other`` escape.

        A free-text ruling offers nothing, because its comment box already is the escape. A row is
        offered its own options when it states any, and the ruling's shared set otherwise.
        """
        if self.control is RulingControl.TEXT:
            return []
        stated = row.options if row is not None and row.options else self.options
        return [*stated, RulingChoice(choice_id=RULING_OTHER_OPTION_ID, label=RULING_OTHER_OPTION_LABEL)]


class RulingSet(BaseModel):
    """Represents every ruling one review wave needs, and the identity its answers are keyed by."""

    model_config = FROZEN_MODEL_CONFIG

    ruling_set_id: str
    """What the saved answers are keyed by, so two waves reviewed on one browser cannot collide."""

    title: str
    intro: str = ""
    items: list[RulingItem]

    @model_validator(mode="after")
    def _check_ruling_ids_are_unique(self) -> RulingSet:
        """Validate the set's identity and that no two rulings would share an answer.

        Raises:
            ValueError: If the set is empty, its identifier is malformed, or a ruling id repeats.
        """
        if not RULING_IDENTIFIER.match(self.ruling_set_id):
            raise ValueError(f"ruling set id {self.ruling_set_id!r} is not a lowercase hyphenated identifier")
        if not self.items:
            raise ValueError("a ruling set with no rulings in it asks nothing")

        ruling_ids = [item.ruling_id for item in self.items]
        if len(set(ruling_ids)) != len(ruling_ids):
            raise ValueError("ruling ids must be unique within a set")

        return self


class GenerationParameters(BaseModel):
    """Represents the resolved generation parameters of a single image.

    This is the shape that reaches disk, in both the sidecars and the index, and the shape both
    runners translate from into their backend's own payload.
    """

    model_config = FROZEN_MODEL_CONFIG

    width: int
    height: int
    steps: int
    cfg_scale: float
    sampler_name: SamplerName
    clip_skip: int
    hires_fix: bool
    tiling: bool
    karras: bool
    denoising_strength: float
    seed: int
    negative_prompt: str = ""
    prompt: str = ""

    hires_fix_denoising_strength: float | None = None
    scheduler: SchedulerName | None = None
    sampler_eta: float | None = None
    sampler_s_noise: float | None = None
    sampler_s_churn: float | None = None
    sampler_solver_type: SolverTypeName | None = None
    sampler_order: int | None = None
    flow_shift: float | None = None

    def differing_field_names(self, other: GenerationParameters) -> set[str]:
        """Return the names of the fields whose values differ between these two."""
        own_values = self.model_dump()
        other_values = other.model_dump()
        return {field_name for field_name, own_value in own_values.items() if own_value != other_values[field_name]}


class JobFeatures(BaseModel):
    """Represents the resolved feature set of a single image.

    This is the companion to [`GenerationParameters`][param_sweeps.spec.GenerationParameters] for
    everything that selects a pipeline rather than a value on one, and it is the shape that
    reaches the sidecars and the index.
    """

    model_config = FROZEN_MODEL_CONFIG

    control_type: str | None = None
    control_strength: float | None = None
    image_is_control: bool = False
    return_control_map: bool = False
    source_processing: SourceProcessing | None = None
    workflow: WorkflowName | None = None
    extra_texts: list[ExtraText] = Field(default_factory=list)
    extra_source_images: list[ExtraSourceImage] = Field(default_factory=list)
    source_mask: str | None = None
    transparent: bool = False
    loras: list[LoraReference] = Field(default_factory=list)
    tis: list[TextualInversionReference] = Field(default_factory=list)
    post_processing: list[str] = Field(default_factory=list)
    facefixer_strength: float | None = None
    batch_size: int = 1
    chosen_candidate: int | None = None

    @property
    def is_plain(self) -> bool:
        """Return whether this job turned no feature on at all."""
        return self == JobFeatures()


class RenderJob(BaseModel):
    """Represents one fully resolved image: every parameter a runner needs, and where it lands."""

    model_config = FROZEN_MODEL_CONFIG

    parameter: SweepParameter
    model_slug: str
    horde_model_name: str
    scene_id: str
    prompt: str
    stop_index: int
    stop_label: str
    stop_value: bool | int | float | str
    caption: str
    width: int
    height: int
    steps: int
    cfg_scale: float
    sampler_name: SamplerName
    clip_skip: int
    hires_fix: bool
    tiling: bool
    karras: bool
    denoising_strength: float
    seed: int
    hires_fix_denoising_strength: float | None = None
    """What the hires pass redraws the enlarged frame by. Unset leaves the backend's own value."""

    scheduler: SchedulerName | None = None
    """The sigma schedule to run on. Unset leaves the schedule to the karras flag, as before."""

    sampler_eta: float | None = None
    sampler_s_noise: float | None = None
    sampler_s_churn: float | None = None
    sampler_solver_type: SolverTypeName | None = None
    sampler_order: int | None = None
    flow_shift: float | None = None
    negative_prompt: str = ""
    prompt_fragment: str | None = None
    """What this stop added to the scene's prompt, when the strip is one that varies the prompt."""

    source_image: str | None = None
    """The image this job starts from, relative to the tool root; unset renders from noise."""

    control_type: str | None = None
    control_strength: float | None = None
    image_is_control: bool = False
    return_control_map: bool = False
    source_processing: SourceProcessing | None = None
    workflow: WorkflowName | None = None
    extra_texts: list[ExtraText] = Field(default_factory=list)
    extra_source_images: list[ExtraSourceImage] = Field(default_factory=list)
    source_mask: str | None = None
    transparent: bool = False
    loras: list[LoraReference] = Field(default_factory=list)
    tis: list[TextualInversionReference] = Field(default_factory=list)
    post_processing: list[str] = Field(default_factory=list)
    facefixer_strength: float | None = None
    batch_size: int = Field(default=1, ge=1, le=MAX_BATCH_SIZE)
    """How many candidates this job renders. Above one, a human picks the published candidate."""

    chosen_candidate: int | None = Field(default=None, ge=1)
    """Which candidate was picked, one-based. Unset means the first one."""

    directory_override: str | None = None
    """Where this job's files go instead of the parameter/model/scene layout, for batch jobs."""

    requires_source_from: str | None = None
    """The scene id of the job whose chosen render is staged into this one's ``source_image``."""

    control_group: str | None = None
    """Which family of detectors this job's control type belongs to, for a consumer to group by.

    Stated in the specification rather than derived, because the grouping is an editorial reading
    of what each detector is for and the backend has no opinion about it.
    """

    @model_validator(mode="after")
    def _check_feature_combinations(self) -> RenderJob:
        """Validate that the features this job turns on are ones the backends can act on."""
        if self.workflow is WorkflowName.QR_CODE and not self._has_qr_payload_text():
            references = ", ".join(sorted(QR_TEXT_REFERENCES))
            raise ValueError(f"the qr_code workflow needs an extra_texts entry referenced as one of: {references}")

        if (self.image_is_control or self.return_control_map) and self.control_type is None:
            raise ValueError("image_is_control and return_control_map mean nothing without a control_type")

        if self.control_type is not None and self.source_image is None:
            raise ValueError(f"control_type {self.control_type!r} needs a source image to condition on")

        if self.source_processing is not None and self.source_image is None:
            raise ValueError(f"source_processing {self.source_processing} needs a source image")

        if self.source_mask is not None and self.source_image is None:
            raise ValueError("a source mask needs a source image to mask")

        if self.chosen_candidate is not None and self.chosen_candidate > self.batch_size:
            raise ValueError(f"chosen candidate {self.chosen_candidate} is outside a batch of {self.batch_size}")

        if self.requires_source_from is not None and self.source_image is None:
            raise ValueError("a job that depends on another job's render must name where it is staged")

        return self

    def _has_qr_payload_text(self) -> bool:
        """Return whether the extra texts carry the string the QR workflow encodes."""
        return any(extra_text.reference in QR_TEXT_REFERENCES for extra_text in self.extra_texts)

    @property
    def source_image_path(self) -> Path | None:
        """Return the absolute path of the image this job starts from, when it has one."""
        if self.source_image is None:
            return None
        return TOOL_ROOT / self.source_image

    @property
    def source_mask_path(self) -> Path | None:
        """Return the absolute path of this job's mask, when it has one."""
        if self.source_mask is None:
            return None
        return TOOL_ROOT / self.source_mask

    @property
    def resolved_source_processing(self) -> SourceProcessing | None:
        """Return what the backend should do with the source image, or None when there is none.

        A job that names a source image and says nothing else is doing plain image-to-image,
        which is what both runners assumed before the processings became stateable.
        """
        if self.source_image is None:
            return None
        return self.source_processing if self.source_processing is not None else SourceProcessing.IMG2IMG

    @property
    def relative_directory(self) -> str:
        """Return the strip's directory, relative to the output root, with forward slashes."""
        if self.directory_override is not None:
            return self.directory_override
        return f"{self.parameter.directory_name}/{self.model_slug}/{self.scene_id}"

    @property
    def candidate_numbers(self) -> list[int]:
        """Return every candidate this job renders, one-based and in render order."""
        return list(range(1, self.batch_size + 1))

    @property
    def chosen_candidate_number(self) -> int:
        """Return the candidate that gets published: the picked one, or the first."""
        return self.chosen_candidate if self.chosen_candidate is not None else 1

    def candidate_stem(self, candidate_number: int) -> str:
        """Return the filename stem of one candidate.

        A job that renders a single image keeps the plain stem it always had, so turning a
        curation batch back into a single render does not rename the file a consumer already has.
        """
        stem = f"{self.stop_index:03d}_{self.stop_label}"
        return stem if self.batch_size == 1 else f"{stem}_{candidate_number:02d}"

    def relative_candidate_image_path(self, candidate_number: int) -> str:
        """Return one candidate's image path, relative to the output root."""
        return f"{self.relative_directory}/{self.candidate_stem(candidate_number)}.webp"

    def relative_candidate_sidecar_path(self, candidate_number: int) -> str:
        """Return one candidate's sidecar path, relative to the output root."""
        return f"{self.relative_directory}/{self.candidate_stem(candidate_number)}.json"

    @property
    def relative_image_path(self) -> str:
        """Return the published image's path relative to the output root, with forward slashes."""
        return self.relative_candidate_image_path(self.chosen_candidate_number)

    @property
    def relative_sidecar_path(self) -> str:
        """Return the published sidecar's path relative to the output root, with forward slashes."""
        return self.relative_candidate_sidecar_path(self.chosen_candidate_number)

    @property
    def relative_control_map_path(self) -> str | None:
        """Return where this job's control map is published, or None when it conditions on none.

        The map lives beside the render it guided rather than in a tree of its own: the pair is
        only meaningful together, and a consumer that has the render's path has the map's.
        """
        if self.control_type is None:
            return None
        return f"{self.relative_directory}/{self.candidate_stem(1)}_map.webp"

    @property
    def checkpoint_key(self) -> str:
        """Return the path resumption is keyed by, which never moves when a candidate is picked.

        Keying on the published image would make marking a choice look like an unrendered job,
        and re-render a batch that has already been paid for.
        """
        return self.relative_candidate_image_path(1)

    def features(self) -> JobFeatures:
        """Return the resolved feature set for this job."""
        return JobFeatures(
            control_type=self.control_type,
            control_strength=self.control_strength,
            image_is_control=self.image_is_control,
            return_control_map=self.return_control_map,
            source_processing=self.resolved_source_processing,
            workflow=self.workflow,
            extra_texts=list(self.extra_texts),
            extra_source_images=list(self.extra_source_images),
            source_mask=self.source_mask,
            transparent=self.transparent,
            loras=list(self.loras),
            tis=list(self.tis),
            post_processing=list(self.post_processing),
            facefixer_strength=self.facefixer_strength,
            batch_size=self.batch_size,
            chosen_candidate=self.chosen_candidate,
        )

    def generation_parameters(self) -> GenerationParameters:
        """Return the resolved generation parameters for this job."""
        return GenerationParameters(
            width=self.width,
            height=self.height,
            steps=self.steps,
            cfg_scale=self.cfg_scale,
            sampler_name=self.sampler_name,
            clip_skip=self.clip_skip,
            hires_fix=self.hires_fix,
            tiling=self.tiling,
            karras=self.karras,
            denoising_strength=self.denoising_strength,
            seed=self.seed,
            negative_prompt=self.negative_prompt,
            prompt=self.prompt,
            hires_fix_denoising_strength=self.hires_fix_denoising_strength,
            **self.solver_parameters(),
        )

    def solver_parameters(self) -> dict[str, Any]:
        """Return the schedule and solver knobs this job states, and only those it states."""
        return {
            "scheduler": self.scheduler,
            "sampler_eta": self.sampler_eta,
            "sampler_s_noise": self.sampler_s_noise,
            "sampler_s_churn": self.sampler_s_churn,
            "sampler_solver_type": self.sampler_solver_type,
            "sampler_order": self.sampler_order,
            "flow_shift": self.flow_shift,
        }


class RenderStrip(BaseModel):
    """Represents one strip: a parameter swept over one model and one scene, with its jobs."""

    model_config = FROZEN_MODEL_CONFIG

    parameter: SweepParameter
    presentation: StripPresentation
    rationale: str
    caption: str
    image_model: ModelSpec
    scene: SceneSpec
    jobs: list[RenderJob]
    source_image: str | None = None
    grid_rows: GridAxis | None = None
    grid_columns: GridAxis | None = None
    batch_section: BatchSection | None = None
    """Which review-batch section this strip's single job belongs to, when it came from a batch."""

    directory_override: str | None = None
    """Where this strip's files go instead of the parameter/model/scene layout."""

    @property
    def strip_id(self) -> str:
        """Return the stable identifier a consumer can key this strip by."""
        if self.directory_override is not None:
            return f"{self.directory_override}/{self.scene.scene_id}"
        return f"{self.parameter.directory_name}/{self.image_model.slug}/{self.scene.scene_id}"

    @property
    def published_source_image(self) -> str | None:
        """Return where this strip's source image is published, relative to the output root.

        A copy per strip rather than per parameter: two sweeps of the same parameter can start
        from different images, and a strip that carries its own source inside its own directory
        cannot be pointed at the wrong one.
        """
        if self.source_image is None:
            return None

        suffix = PurePosixPath(self.source_image).suffix
        if self.directory_override is not None:
            return f"{self.directory_override}/{self.scene.scene_id}-source{suffix}"
        return f"{self.strip_id}/source{suffix}"


class SweepSpec(BaseModel):
    """Represents the whole checked-in sweep specification."""

    model_config = FROZEN_MODEL_CONFIG

    schema_version: Literal[1]
    seed: int
    anchor: AnchorParameters
    models: list[ModelSpec]
    scenes: list[SceneSpec] = Field(default_factory=list)
    sweeps: list[Sweep] = Field(default_factory=list)
    batches: list[BatchSpec] = Field(default_factory=list)
    """Review batches: sections of one-off jobs a human signs off, rather than teaching strips."""

    rulings: list[RulingSet] = Field(default_factory=list)
    """The questions this specification's batches exist to answer, rendered onto the contact sheet.

    More than one set, because a wave that has been ruled on is often followed by a shorter one
    asking only what the first left open. A set is what answers are keyed by, so a new set opens
    empty rather than reopening the decisions that were already made.
    """

    @model_validator(mode="after")
    def _check_references_resolve(self) -> SweepSpec:
        """Validate that identifiers are unique and every sweep or batch reference names something."""
        known_model_slugs = {image_model.slug for image_model in self.models}
        if len(known_model_slugs) != len(self.models):
            raise ValueError("model slugs must be unique")

        known_scene_ids = {scene.scene_id for scene in self.scenes}
        if len(known_scene_ids) != len(self.scenes):
            raise ValueError("scene ids must be unique")

        if not self.sweeps and not self.batches:
            raise ValueError("a specification must describe at least one sweep or one batch")

        for sweep in self.sweeps:
            unknown_models = set(sweep.model_slugs) - known_model_slugs
            if unknown_models:
                raise ValueError(f"sweep {sweep.parameter} names unknown models: {sorted(unknown_models)}")

            unknown_scenes = set(sweep.scene_ids) - known_scene_ids
            if unknown_scenes:
                raise ValueError(f"sweep {sweep.parameter} names unknown scenes: {sorted(unknown_scenes)}")

        self._check_batch_references_resolve(known_model_slugs)
        self._check_ruling_evidence_resolves()
        self._check_hires_fix_geometry()
        return self

    def _check_ruling_evidence_resolves(self) -> None:
        """Validate that every ruling points at something this specification actually renders.

        A ruling whose evidence link goes nowhere is worse than one with no link, because it
        reads as if the reviewer has been shown the images and they have not.

        Raises:
            ValueError: If two sets share an identifier, or a ruling names a batch section or a
                job the batches do not contain.
        """
        set_ids = [ruling_set.ruling_set_id for ruling_set in self.rulings]
        if len(set(set_ids)) != len(set_ids):
            raise ValueError("ruling set ids must be unique, or one set's answers would overwrite another's")

        known_sections = {batch.section for batch in self.batches}
        known_job_ids = {job.job_id for batch in self.batches for job in batch.jobs}

        for item in (item for ruling_set in self.rulings for item in ruling_set.items):
            unknown_sections = set(item.evidence_sections) - known_sections
            if unknown_sections:
                raise ValueError(
                    f"ruling {item.ruling_id} cites sections no batch renders: {sorted(unknown_sections)}",
                )

            cited: list[str | None] = [option.evidence_job_id for option in item.options]
            for row in item.rows:
                cited.append(row.evidence_job_id)
                cited.extend(option.evidence_job_id for option in row.options)
            cited_job_ids = set(item.evidence_job_ids) | {job_id for job_id in cited if job_id is not None}

            unknown_job_ids = cited_job_ids - known_job_ids
            if unknown_job_ids:
                raise ValueError(f"ruling {item.ruling_id} cites jobs no batch renders: {sorted(unknown_job_ids)}")

    def _check_hires_fix_geometry(self) -> None:
        """Validate that every job asking for the hires pass renders where the backends run it.

        Three ways a job can ask for the pass and not get it. Either requested dimension at or
        below the baseline's training size (512 for SD1.5, 1024 for SDXL) turns the flag off, and
        so does inpainting or outpainting, whatever the size; such a job renders as if the flag
        were never set and publishes a pair of identical images as a comparison. Stating what the
        pass redraws by without asking for the pass is the third: nothing reads the value.

        Raises:
            ValueError: If a job asks for the hires pass where the backends would not run it, or
                states its strength without asking for it.
        """
        disable_threshold_by_baseline = {"stable_diffusion_1": 512, "stable_diffusion_xl": 1024}
        painting = (SourceProcessing.INPAINTING, SourceProcessing.OUTPAINTING)

        for job in self.expand_jobs():
            where = f"job {job.scene_id!r} ({job.parameter}/{job.model_slug})"

            if not job.hires_fix:
                if job.hires_fix_denoising_strength is not None:
                    raise ValueError(
                        f"{where} states a hires denoising strength of"
                        f" {job.hires_fix_denoising_strength} without asking for the hires pass,"
                        f" and nothing reads it",
                    )
                continue

            threshold = disable_threshold_by_baseline.get(self.model_for_slug(job.model_slug).baseline)
            if threshold is not None and (job.width <= threshold or job.height <= threshold):
                raise ValueError(
                    f"{where} asks for hires fix at {job.width}x{job.height}, but the backends"
                    f" silently disable it unless both dimensions exceed {threshold}",
                )

            if job.resolved_source_processing in painting:
                raise ValueError(
                    f"{where} asks for hires fix while {job.resolved_source_processing}, which the"
                    f" backends silently disable",
                )

    def _check_batch_references_resolve(self, known_model_slugs: set[str]) -> None:
        """Validate every batch job's model and dependency, and that job ids are unique overall.

        Raises:
            ValueError: If a job names a model or a producing job that the specification does
                not describe, or if two sections reuse a job id.
        """
        all_job_ids: list[str] = [job.job_id for batch in self.batches for job in batch.jobs]
        if len(set(all_job_ids)) != len(all_job_ids):
            raise ValueError("batch job ids must be unique across every section")

        for batch in self.batches:
            for job in batch.jobs:
                if job.model_slug not in known_model_slugs:
                    raise ValueError(f"batch job {job.job_id} names unknown model {job.model_slug!r}")

                if job.requires_source_from is not None and job.requires_source_from not in all_job_ids:
                    raise ValueError(
                        f"batch job {job.job_id} depends on {job.requires_source_from!r}, which no job produces",
                    )

    def model_for_slug(self, slug: str) -> ModelSpec:
        """Return the model with this slug.

        Raises:
            KeyError: If no model in the specification carries the slug.
        """
        for image_model in self.models:
            if image_model.slug == slug:
                return image_model
        raise KeyError(f"no model with slug {slug!r}")

    def scene_for_id(self, scene_id: str) -> SceneSpec:
        """Return the scene with this identifier.

        Raises:
            KeyError: If no scene in the specification carries the identifier.
        """
        for scene in self.scenes:
            if scene.scene_id == scene_id:
                return scene
        raise KeyError(f"no scene with id {scene_id!r}")

    def ruling_set_for_id(self, ruling_set_id: str) -> RulingSet:
        """Return the ruling set with this identifier.

        Raises:
            KeyError: If no set in the specification carries the identifier.
        """
        for ruling_set in self.rulings:
            if ruling_set.ruling_set_id == ruling_set_id:
                return ruling_set
        raise KeyError(f"no ruling set with id {ruling_set_id!r}")

    def expand_strips(self) -> list[RenderStrip]:
        """Return every strip the specification describes, in declaration order.

        A batch job expands to a strip of one stop, so a batch travels through the campaign
        driver, the checkpoint and the index on exactly the same rails a teaching strip does.
        """
        strips: list[RenderStrip] = []
        for sweep in self.sweeps:
            for model_slug in sweep.model_slugs:
                for scene_id in sweep.scene_ids:
                    strips.append(self._expand_one_strip(sweep, model_slug, scene_id))

        for batch in self.batches:
            for job_ordinal, batch_job in enumerate(batch.jobs, start=1):
                strips.append(self._expand_batch_job(batch, batch_job, job_ordinal))

        return strips

    def _expand_batch_job(self, batch: BatchSpec, batch_job: BatchJob, job_ordinal: int) -> RenderStrip:
        """Return the one-stop strip a batch job expands to."""
        image_model = self.model_for_slug(batch_job.model_slug)
        held = batch.holds
        directory = f"{SweepParameter.BATCH.directory_name}/{batch.section}/{image_model.slug}"
        scene = SceneSpec(
            scene_id=batch_job.job_id,
            label=batch_job.caption,
            prompt=batch_job.prompt,
            derived_from=batch.title,
        )

        job = RenderJob(
            parameter=SweepParameter.BATCH,
            model_slug=image_model.slug,
            horde_model_name=image_model.horde_name,
            scene_id=batch_job.job_id,
            prompt=batch_job.prompt,
            stop_index=job_ordinal,
            stop_label=batch_job.job_id,
            stop_value=batch_job.job_id,
            caption=batch_job.caption,
            width=_resolved(batch_job.width, held.width, image_model.native_width),
            height=_resolved(batch_job.height, held.height, image_model.native_height),
            steps=_resolved(batch_job.steps, held.steps, self.anchor.steps),
            cfg_scale=_resolved(batch_job.cfg_scale, held.cfg_scale, self.anchor.cfg_scale),
            sampler_name=_resolved(batch_job.sampler_name, held.sampler_name, self.anchor.sampler_name),
            clip_skip=_resolved(batch_job.clip_skip, held.clip_skip, image_model.anchor_clip_skip),
            hires_fix=_resolved(batch_job.hires_fix, held.hires_fix, self.anchor.hires_fix),
            tiling=_resolved(batch_job.tiling, held.tiling, self.anchor.tiling),
            karras=_resolved(batch_job.karras, held.karras, self.anchor.karras),
            denoising_strength=_resolved(
                batch_job.denoising_strength,
                held.denoising_strength,
                self.anchor.denoising_strength,
            ),
            hires_fix_denoising_strength=_resolved(
                batch_job.hires_fix_denoising_strength,
                held.hires_fix_denoising_strength,
                None,
            ),
            seed=_resolved(batch_job.seed, held.seed, self.seed),
            scheduler=_resolved(batch_job.scheduler, held.scheduler, self.anchor.scheduler),
            negative_prompt=batch_job.negative_prompt,
            **resolved_solver_parameters(stop=batch_job, held=held),
            source_image=batch_job.source_image,
            requires_source_from=batch_job.requires_source_from,
            directory_override=directory,
            **resolved_features(stop=batch_job, held=held),
        )

        return RenderStrip(
            parameter=SweepParameter.BATCH,
            presentation=StripPresentation.STRIP,
            rationale=batch.rationale,
            caption=batch_job.caption,
            image_model=image_model,
            scene=scene,
            jobs=[job],
            source_image=batch_job.source_image,
            batch_section=batch.section,
            directory_override=directory,
        )

    def expand_jobs(self) -> list[RenderJob]:
        """Return every render job the specification describes, in strip order."""
        return [job for strip in self.expand_strips() for job in strip.jobs]

    def _expand_one_strip(self, sweep: StripSweep | GridSweep, model_slug: str, scene_id: str) -> RenderStrip:
        image_model = self.model_for_slug(model_slug)
        scene = self.scene_for_id(scene_id)

        if isinstance(sweep, GridSweep):
            stops = _grid_stops(sweep)
            presentation = StripPresentation.GRID
            resolution_multiplier = 1.0
        else:
            stops = sweep.stops
            presentation = sweep.presentation
            resolution_multiplier = sweep.resolution_multiplier

        jobs = [
            self._build_job(
                sweep=sweep,
                image_model=image_model,
                scene=scene,
                stop=stop,
                stop_index=stop_index,
                resolution_multiplier=resolution_multiplier,
            )
            for stop_index, stop in enumerate(stops, start=1)
        ]

        return RenderStrip(
            parameter=sweep.parameter,
            presentation=presentation,
            rationale=sweep.rationale,
            caption=sweep.caption,
            image_model=image_model,
            scene=scene,
            jobs=jobs,
            source_image=sweep.source_image,
            grid_rows=sweep.rows if isinstance(sweep, GridSweep) else None,
            grid_columns=sweep.columns if isinstance(sweep, GridSweep) else None,
            directory_override=sweep.directory_override,
        )

    def _build_job(
        self,
        *,
        sweep: StripSweep | GridSweep,
        image_model: ModelSpec,
        scene: SceneSpec,
        stop: SweepStop,
        stop_index: int,
        resolution_multiplier: float,
    ) -> RenderJob:
        held = sweep.holds
        width = _resolved(stop.width, held.width, _scaled(image_model.native_width, resolution_multiplier))
        height = _resolved(stop.height, held.height, _scaled(image_model.native_height, resolution_multiplier))
        uses_negative_prompt = _resolved(stop.use_negative_prompt, held.use_negative_prompt, True)
        prompt_fragment = _resolved(stop.prompt_fragment, held.prompt_fragment, None)

        return RenderJob(
            parameter=sweep.parameter,
            model_slug=image_model.slug,
            horde_model_name=image_model.horde_name,
            scene_id=scene.scene_id,
            prompt=scene.prompt if prompt_fragment is None else f"{scene.prompt}, {prompt_fragment}",
            prompt_fragment=prompt_fragment,
            stop_index=stop_index,
            stop_label=stop.filename_label,
            stop_value=stop.value,
            caption=stop.caption,
            control_group=stop.control_group,
            width=width,
            height=height,
            steps=_resolved(stop.steps, held.steps, self.anchor.steps),
            cfg_scale=_resolved(stop.cfg_scale, held.cfg_scale, self.anchor.cfg_scale),
            sampler_name=_resolved(stop.sampler_name, held.sampler_name, self.anchor.sampler_name),
            clip_skip=_resolved(stop.clip_skip, held.clip_skip, image_model.anchor_clip_skip),
            hires_fix=_resolved(stop.hires_fix, held.hires_fix, self.anchor.hires_fix),
            tiling=_resolved(stop.tiling, held.tiling, self.anchor.tiling),
            karras=_resolved(stop.karras, held.karras, self.anchor.karras),
            denoising_strength=_resolved(
                stop.denoising_strength,
                held.denoising_strength,
                self.anchor.denoising_strength,
            ),
            hires_fix_denoising_strength=_resolved(
                stop.hires_fix_denoising_strength,
                held.hires_fix_denoising_strength,
                None,
            ),
            seed=_resolved(stop.seed, held.seed, self.seed),
            scheduler=_resolved(stop.scheduler, held.scheduler, self.anchor.scheduler),
            negative_prompt=sweep.negative_prompt if uses_negative_prompt else "",
            source_image=_resolved(stop.source_image, held.source_image, sweep.source_image),
            directory_override=sweep.directory_override,
            **resolved_solver_parameters(stop=stop, held=held),
            **resolved_features(stop=stop, held=held),
        )


def resolved_solver_parameters(*, stop: ParameterOverrides, held: ParameterOverrides) -> dict[str, Any]:
    """Return the schedule and solver knobs, resolved from the stop then the sweep.

    They default to unset rather than to an anchor value, because unset is not a value: it is the
    backend building the sampler exactly as it did before any of these knobs existed.

    Args:
        stop: The stop's own overrides.
        held: What the sweep holds for every stop that does not state its own.

    Returns:
        The keyword arguments to build a [`RenderJob`][param_sweeps.spec.RenderJob] with.
    """
    return {
        "sampler_eta": _resolved(stop.sampler_eta, held.sampler_eta, None),
        "sampler_s_noise": _resolved(stop.sampler_s_noise, held.sampler_s_noise, None),
        "sampler_s_churn": _resolved(stop.sampler_s_churn, held.sampler_s_churn, None),
        "sampler_solver_type": _resolved(stop.sampler_solver_type, held.sampler_solver_type, None),
        "sampler_order": _resolved(stop.sampler_order, held.sampler_order, None),
        "flow_shift": _resolved(stop.flow_shift, held.flow_shift, None),
    }


def resolved_features(*, stop: FeatureOverrides, held: FeatureOverrides) -> dict[str, Any]:
    """Return the feature fields a render job takes, resolved from the stop then the sweep.

    Every feature is off unless something turns it on, which is why the defaults here are the
    inert ones rather than the sweep's: a feature that leaked from a sweep into a stop that did
    not ask for it would change what an image teaches without saying so anywhere.

    Args:
        stop: The stop's own overrides.
        held: What the sweep holds for every stop that does not state its own.

    Returns:
        The keyword arguments to build a [`RenderJob`][param_sweeps.spec.RenderJob] with.
    """
    return {
        "control_type": _resolved(stop.control_type, held.control_type, None),
        "control_strength": _resolved(stop.control_strength, held.control_strength, None),
        "image_is_control": _resolved(stop.image_is_control, held.image_is_control, False),
        "return_control_map": _resolved(stop.return_control_map, held.return_control_map, False),
        "source_processing": _resolved(stop.source_processing, held.source_processing, None),
        "workflow": _resolved(stop.workflow, held.workflow, None),
        "extra_texts": _resolved(stop.extra_texts, held.extra_texts, []),
        "extra_source_images": _resolved(stop.extra_source_images, held.extra_source_images, []),
        "source_mask": _resolved(stop.source_mask, held.source_mask, None),
        "transparent": _resolved(stop.transparent, held.transparent, False),
        "loras": _resolved(stop.loras, held.loras, []),
        "tis": _resolved(stop.tis, held.tis, []),
        "post_processing": _resolved(stop.post_processing, held.post_processing, []),
        "facefixer_strength": _resolved(stop.facefixer_strength, held.facefixer_strength, None),
        "batch_size": _resolved(stop.batch_size, held.batch_size, 1),
        "chosen_candidate": _resolved(stop.chosen_candidate, held.chosen_candidate, None),
    }


def _resolved[ValueT](stop_value: ValueT | None, held_value: ValueT | None, inherited: ValueT) -> ValueT:
    """Return the first value that was actually stated: the stop's, then the sweep's, then the inherited one."""
    if stop_value is not None:
        return stop_value
    if held_value is not None:
        return held_value
    return inherited


def value_label(value: bool | int | float | str) -> str:
    """Return a filename-safe token for a stop value.

    Booleans become ``on``/``off`` because ``true``/``false`` reads poorly in a filename, and a
    decimal point is replaced rather than kept so that the only dot in a filename is its
    extension separator.
    """
    if isinstance(value, bool):
        return "on" if value else "off"

    text = str(value).lower().replace(".", "_")
    return NON_LABEL_CHARACTERS.sub("-", text).strip("-_") or "value"


def _scaled(dimension: int, multiplier: float) -> int:
    """Return the dimension scaled and snapped down to a multiple of 64."""
    scaled = int(dimension * multiplier)
    return max(64, scaled - (scaled % 64))


def _grid_stops(sweep: GridSweep) -> list[SweepStop]:
    """Return the grid's cells in row-major order: the row axis outer, the column axis inner."""
    stops: list[SweepStop] = []
    for row_value in sweep.rows.values:
        for column_value in sweep.columns.values:
            row_text = _grid_value_text(row_value)
            column_text = _grid_value_text(column_value)
            cell: dict[str, Any] = {
                "value": f"{row_text}x{column_text}",
                "label": f"{sweep.rows.label_for(row_value)}_{sweep.columns.label_for(column_value)}",
                "caption": sweep.cell_caption.format(row=row_text, column=column_text),
            }
            for field_name in sweep.rows.stop_fields():
                cell[field_name] = row_value
            for field_name in sweep.columns.stop_fields():
                cell[field_name] = column_value

            stops.append(SweepStop.model_validate(cell))
    return stops


def _grid_value_text(value: bool | int | float | str) -> str:
    """Return an axis value as a caption reads it: no trailing zero, and a flag as off or on."""
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def load_spec(spec_path: Path) -> SweepSpec:
    """Read and validate the specification at this path.

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

    return SweepSpec.model_validate(parsed)
