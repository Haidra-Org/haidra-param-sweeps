# SPDX-FileCopyrightText: 2026 Tazlin
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compose share cards and posters from the renders that are already in the library.

A link to a reference page unfurls in a chat or a feed as its ``og:image`` image. For a page
whose evidence is a strip of renders, the strip is a better picture than a logo. The same frames
make a square poster for a forum, a tall card for a phone feed, or a two-frame comparison, so one
composer serves every one of those from a small declarative specification rather than a file per
occasion.

A card is one of a few templates drawn on one of a few canvases in one of two palettes. The
template sets the frame layout, the format sets the canvas, the theme sets the colours, and the
stop selection picks the strip's frames. A job names one of each. A set names a template and
format and produces one card per parameter or per tutorial, so the whole library gets preview
images without a job per page.

Cards are promotional material cut from the evidence rather than evidence, so they are written
to a separate tree beside the library, with a manifest at its root, and are published under a
separate prefix. A page finds its card by naming convention rather than by lookup:
``sweeps/<parameter>.jpg`` for a setting, ``tutorials/<flow>.jpg`` for a worked example.

Critical public members:

- [`PromoSpec`][param_sweeps.promo.PromoSpec]: the specification, with its jobs and sets.
- [`expand_jobs`][param_sweeps.promo.expand_jobs]: the explicit jobs plus the jobs that the sets expand to.
- [`render_card`][param_sweeps.promo.render_card]: one job drawn as an image.
- [`write_cards`][param_sweeps.promo.write_cards]: every job written, and the manifest beside them.
"""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw, ImageFilter, ImageFont
from pydantic import BaseModel, ConfigDict, Field
from strenum import StrEnum

from param_sweeps.flows import MANIFEST_FILENAME as TUTORIAL_MANIFEST_FILENAME
from param_sweeps.flows import TUTORIALS_DIRECTORY_NAME, ManifestFlow, TutorialManifest
from param_sweeps.outputs import INDEX_FILENAME, IndexStop, IndexStrip, SweepIndex

logger = logging.getLogger(__name__)

PROMO_DIRECTORY_NAME = "promo"
"""The tree for the cards, beside the library and not inside it."""

CARDS_MANIFEST_FILENAME = "cards.json"
"""The manifest at the root of the promo tree, which lists every card by its key."""

SWEEP_CARDS_DIRECTORY_NAME = "sweeps"
"""The subtree for a per-parameter set, one card per parameter."""

TUTORIAL_CARDS_DIRECTORY_NAME = "tutorials"
"""The subtree for a per-tutorial set, one card per flow."""

CARDS_MANIFEST_SCHEMA_VERSION = 1

OVERLAY_OUTLINE = (6, 9, 18)
"""The outline and shadow colour for text drawn over a picture."""

OVERLAY_INK = "#ffffff"
OVERLAY_MUTED = "#f3f4f6"

SELECTION_PATTERN = re.compile(r"^(preview|all|spread:(\d+)|first:(\d+))$")


GALLERY_DIRECTORY_NAME = "gallery"
"""The directory of the promo gallery's entries under the promo tree, one directory per picture."""

GALLERY_RESULT_FILENAME = "result.webp"
"""The picture inside a gallery entry's directory."""


class CardTemplate(StrEnum):
    """How a card arranges its frames."""

    STRIP = "strip"
    """The chosen stops in a row (or a few rows), each with its value under it."""

    PAIR = "pair"
    """Two frames side by side with a divider between them, like the explorer's wipe view."""

    GRID = "grid"
    """Every chosen stop in as square a grid as the canvas allows."""

    HERO = "hero"
    """One frame filling the picture area, with the title over its bottom edge."""


class CardFormat(StrEnum):
    """The canvas that a card is drawn on, named after where it is shown."""

    OG = "og"
    """The link-preview canvas that every unfurler crops to."""

    SQUARE = "square"
    WIDE = "wide"
    STORY = "story"
    """A phone-shaped canvas for short-video covers and vertical feeds."""

    POSTER = "poster"
    """A large square for a forum post that will be opened at full size."""


CANVAS_SIZES: dict[CardFormat, tuple[int, int]] = {
    CardFormat.OG: (1200, 630),
    CardFormat.SQUARE: (1080, 1080),
    CardFormat.WIDE: (1920, 1080),
    CardFormat.STORY: (1080, 1920),
    CardFormat.POSTER: (2048, 2048),
}


class CardTheme(StrEnum):
    """Which palette a card uses."""

    DARK = "dark"
    LIGHT = "light"


class Palette(BaseModel):
    """Represents one theme's colours."""

    model_config = ConfigDict(frozen=True)

    background: str
    ink: str
    muted: str
    accent: str
    pill_background: str
    pill_ink: str
    divider: str


DEFAULT_PALETTES: dict[CardTheme, Palette] = {
    CardTheme.DARK: Palette(
        background="#0f172a",
        ink="#f8fafc",
        muted="#94a3b8",
        accent="#60a5fa",
        pill_background="#1e293b",
        pill_ink="#e2e8f0",
        divider="#f8fafc",
    ),
    CardTheme.LIGHT: Palette(
        background="#f8fafc",
        ink="#111827",
        muted="#4b5563",
        accent="#1d4ed8",
        pill_background="#e5e7eb",
        pill_ink="#111827",
        divider="#111827",
    ),
}


class BrandSpec(BaseModel):
    """Represents the brand details shown on every card."""

    model_config = ConfigDict(frozen=True)

    name: str
    url: str
    tagline: str = "Same prompt. Same seed."
    logo: str | None = None
    """An image drawn at the foot of the card, relative to the specification file."""


class FontSpec(BaseModel):
    """Represents a card's three typefaces, as paths relative to the specification file."""

    model_config = ConfigDict(frozen=True)

    regular: str
    bold: str
    mono: str


class CardJob(BaseModel):
    """Represents one card to draw.

    A job names its frames in one of four ways: a ``strip`` by the library's strip id, a
    ``parameter`` (resolved to that parameter's preferred strip), a ``tutorial`` by flow id, or a
    ``gallery`` picture by its entry id. A gallery picture does not have its own text, so a job
    that names one must state its title and subtitle.
    """

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    job_id: str = Field(alias="id")
    """The card's key in the manifest, and the file name unless ``output`` sets one."""

    template: CardTemplate
    format: CardFormat = CardFormat.OG
    theme: CardTheme = CardTheme.DARK
    strip: str | None = None
    parameter: str | None = None
    tutorial: str | None = None
    gallery: str | None = None
    """A promo gallery entry, drawn from ``gallery/<id>/result.webp`` under the promo tree."""

    stops: str | list[int] = "spread:6"
    """Which frames take part: ``preview`` (the strip's own pair), ``all``, ``spread:N`` (N frames
    evenly across the strip, first and last included), ``first:N``, or explicit stop indices."""

    title: str | None = None
    subtitle: str | None = None
    link: str | None = None
    """The address printed under the brand, when the card links to a more specific page than the site."""

    tagline: str | None = None
    """The line above the address, when the brand's tagline does not fit the card's message."""

    file_type: Literal["jpg", "png"] = "jpg"
    output: str | None = None
    """The file's path under the promo tree. Defaults to ``<id>.<file_type>``."""

    def output_path(self) -> str:
        """Return the card's path relative to the promo tree's root."""
        return self.output if self.output is not None else f"{self.job_id}.{self.file_type}"


class CardSet(BaseModel):
    """Represents one card per parameter or per tutorial, all drawn the same way."""

    model_config = ConfigDict(frozen=True)

    every: Literal["parameter", "tutorial"]
    template: CardTemplate
    format: CardFormat = CardFormat.OG
    theme: CardTheme = CardTheme.DARK
    stops: str | list[int] = "spread:6"
    file_type: Literal["jpg", "png"] = "jpg"
    directory: str | None = None
    """The set's subtree. Defaults to ``sweeps`` or ``tutorials`` by kind."""

    key_prefix: str | None = None
    """The prefix of the manifest keys. Defaults to ``sweep:`` or ``tutorial:`` by kind."""

    def resolved_directory(self) -> str:
        """Return this set's subtree."""
        if self.directory is not None:
            return self.directory
        return SWEEP_CARDS_DIRECTORY_NAME if self.every == "parameter" else TUTORIAL_CARDS_DIRECTORY_NAME

    def resolved_key_prefix(self) -> str:
        """Return the prefix of this set's manifest keys."""
        if self.key_prefix is not None:
            return self.key_prefix
        return "sweep:" if self.every == "parameter" else "tutorial:"


class PromoSpec(BaseModel):
    """Represents the whole specification: brand, typefaces, palettes, the text for each parameter, and the jobs."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    brand: BrandSpec
    fonts: FontSpec | None = None
    """The brand's typefaces. None draws every card in Pillow's bundled face."""

    palettes: dict[CardTheme, Palette] = Field(default_factory=lambda: dict(DEFAULT_PALETTES))
    titles: dict[str, str] = Field(default_factory=dict)
    """A parameter's reader-facing name (``cfg_scale`` is "Guidance"). Unlisted parameters are
    humanised from their field name."""

    subtitles: dict[str, str] = Field(default_factory=dict)
    """The text under a parameter's title on its card: one plain sentence on what the setting does.
    An unlisted parameter falls back to the strip's model, scene and seed, which describe where the
    frames came from but not what they show."""

    preferred_strips: dict[str, str] = Field(default_factory=dict)
    """The strip that a per-parameter set draws for a parameter with several strips. Unlisted
    parameters use the first complete strip in index order."""

    skip_parameters: list[str] = Field(default_factory=list)
    """Parameters that a per-parameter set leaves out, such as the internal timeline strips."""

    pair_labels: tuple[str, str] = ("Before", "After")
    """The labels of a tutorial pair's two frames."""

    jobs: list[CardJob] = Field(default_factory=list)
    sets: list[CardSet] = Field(default_factory=list)


class CardOutcome(BaseModel):
    """Represents the outcome of one job."""

    model_config = ConfigDict(frozen=True)

    key: str
    relative_path: str
    """The card's path relative to the promo tree's root."""

    width: int
    height: int
    written: bool
    size_bytes: int


class CardManifestEntry(BaseModel):
    """Represents one card's entry in the manifest."""

    model_config = ConfigDict(frozen=True)

    file: str
    """The card's path relative to the promo tree's root."""

    width: int
    height: int
    template: CardTemplate
    source: str
    """The strip id or flow id of the card's source."""


class CardsManifest(BaseModel):
    """Represents the manifest at the root of the promo tree: every card, by key."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = CARDS_MANIFEST_SCHEMA_VERSION
    cards: dict[str, CardManifestEntry]


class Fonts:
    """The three typefaces, loaded at the sizes that a card requests.

    Without a specification, every face is Pillow's bundled face, so the composer can run and be
    tested without a brand typeface.
    """

    def __init__(self, spec: FontSpec | None, *, base: Path) -> None:
        """Store where the faces are. No file is read until a size is requested."""
        self._paths: dict[str, Path] = {}
        if spec is not None:
            self._paths = {
                "regular": base / spec.regular,
                "bold": base / spec.bold,
                "mono": base / spec.mono,
            }
        self._cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}

    def get(self, face: str, size: int) -> ImageFont.FreeTypeFont:
        """Return the face at the size, reading the file the first time it is requested.

        Raises:
            RuntimeError: If no face was specified and Pillow's bundled one is not a scalable face
                here, which happens only where Pillow was built without FreeType.
        """
        key = (face, size)
        if key not in self._cache:
            if face in self._paths:
                self._cache[key] = ImageFont.truetype(str(self._paths[face]), size)
            else:
                default = ImageFont.load_default(size=size)
                if not isinstance(default, ImageFont.FreeTypeFont):
                    raise RuntimeError("no font was specified and Pillow has no scalable default face here")
                self._cache[key] = default
        return self._cache[key]


def load_promo_spec(spec_path: Path) -> PromoSpec:
    """Read and validate the specification.

    Raises:
        FileNotFoundError: If the file does not exist.
        pydantic.ValidationError: If the file is not a valid specification.
    """
    return PromoSpec.model_validate_json(spec_path.read_text(encoding="utf-8"))


def humanise_parameter(parameter: str) -> str:
    """Return a field name as a title: ``cfg_scale`` becomes ``Cfg scale``."""
    return parameter.replace("_", " ").capitalize()


def parameter_title(spec: PromoSpec, parameter: str) -> str:
    """Return the specification's title for a parameter, or the humanised field name."""
    return spec.titles.get(parameter, humanise_parameter(parameter))


def preferred_strip(index: SweepIndex, *, parameter: str, override: str | None = None) -> IndexStrip:
    """Return the strip that a per-parameter card draws.

    Args:
        index: The library's index.
        parameter: The parameter to find a strip for.
        override: The specification's preferred strip id, checked to belong to the parameter.

    Raises:
        ValueError: If the parameter does not have a complete published strip, or the override
            names a strip of another parameter.
    """
    candidates = [strip for strip in index.strips if str(strip.parameter) == parameter]
    if override is not None:
        for strip in candidates:
            if strip.strip_id == override:
                return strip
        raise ValueError(f"preferred strip {override!r} is not a published strip of {parameter!r}")
    complete = [strip for strip in candidates if strip.complete and strip.stops]
    if not complete:
        raise ValueError(f"no complete strip is published for {parameter!r}")
    return complete[0]


def find_strip(index: SweepIndex, strip_id: str) -> IndexStrip:
    """Return the strip with the id.

    Raises:
        ValueError: If no strip has the id.
    """
    for strip in index.strips:
        if strip.strip_id == strip_id:
            return strip
    raise ValueError(f"no strip {strip_id!r} in the index")


def find_flow(manifest: TutorialManifest, flow_id: str) -> ManifestFlow:
    """Return the tutorial flow with the id.

    Raises:
        ValueError: If no flow has the id, or the flow was skipped and has no result to show.
    """
    for flow in manifest.flows:
        if flow.flow_id == flow_id:
            if flow.result is None:
                raise ValueError(f"tutorial {flow_id!r} has no result to draw")
            return flow
    raise ValueError(f"no tutorial {flow_id!r} in the manifest")


def select_stops(strip: IndexStrip, selection: str | list[int]) -> list[IndexStop]:
    """Return the stops that a selection names, in strip order.

    Raises:
        ValueError: If the selection is not one of the recognised forms or names a stop the
            strip does not have.
    """
    count = len(strip.stops)
    if isinstance(selection, list):
        for stop_index in selection:
            if not 0 <= stop_index < count:
                raise ValueError(f"stop {stop_index} is outside {strip.strip_id!r} ({count} stops)")
        return [strip.stops[stop_index] for stop_index in selection]

    match = SELECTION_PATTERN.match(selection)
    if match is None:
        raise ValueError(f"unrecognised stop selection {selection!r}")
    if selection == "all":
        return list(strip.stops)
    if selection == "preview":
        # A strip whose held value is its last stop resolves its default preview to the same
        # frame twice. A pair of one frame shows nothing, so the ends are used instead.
        if strip.preview is None or strip.preview.left_index == strip.preview.right_index:
            return [strip.stops[0], strip.stops[-1]]
        return [strip.stops[strip.preview.left_index], strip.stops[strip.preview.right_index]]
    wanted = int(match.group(2) or match.group(3))
    if wanted <= 0:
        raise ValueError(f"a selection must include at least one stop, not {selection!r}")
    if selection.startswith("first:"):
        return list(strip.stops[:wanted])
    return [strip.stops[position] for position in spread_positions(count, wanted)]


def spread_positions(count: int, wanted: int) -> list[int]:
    """Return ``wanted`` indices evenly across ``count`` items, first and last included."""
    if wanted >= count:
        return list(range(count))
    if wanted == 1:
        return [0]
    positions = [round(step * (count - 1) / (wanted - 1)) for step in range(wanted)]
    return sorted(set(positions))


def expand_jobs(spec: PromoSpec, *, index: SweepIndex, manifest: TutorialManifest | None) -> list[CardJob]:
    """Return the explicit jobs followed by the jobs that each set unfolds to.

    A per-parameter set yields one job for every parameter that the index publishes a complete
    strip for, keyed ``sweep:<parameter>``. A per-tutorial set yields one for every flow with a
    result, keyed ``tutorial:<flow>``. A parameter that the specification skips, or one without a
    complete strip, is skipped with a log line instead of failing the run.

    Raises:
        ValueError: If a tutorial set is requested and there is no manifest.
    """
    jobs = list(spec.jobs)
    for card_set in spec.sets:
        if card_set.every == "parameter":
            jobs.extend(_parameter_set_jobs(spec, card_set, index=index))
        else:
            if manifest is None:
                raise ValueError("a tutorial set needs the tutorial manifest, and none was found")
            jobs.extend(_tutorial_set_jobs(card_set, manifest=manifest))
    return jobs


def _parameter_set_jobs(spec: PromoSpec, card_set: CardSet, *, index: SweepIndex) -> list[CardJob]:
    seen: list[str] = []
    for strip in index.strips:
        parameter = str(strip.parameter)
        if parameter not in seen:
            seen.append(parameter)
    jobs: list[CardJob] = []
    for parameter in seen:
        if parameter in spec.skip_parameters:
            continue
        try:
            strip = preferred_strip(index, parameter=parameter, override=spec.preferred_strips.get(parameter))
        except ValueError as reason:
            logger.info("skipping %s: %s", parameter, reason)
            continue
        jobs.append(
            CardJob(
                job_id=f"{card_set.resolved_key_prefix()}{parameter}",
                template=card_set.template,
                format=card_set.format,
                theme=card_set.theme,
                strip=strip.strip_id,
                stops=card_set.stops,
                file_type=card_set.file_type,
                output=f"{card_set.resolved_directory()}/{parameter}.{card_set.file_type}",
            ),
        )
    return jobs


def _tutorial_set_jobs(card_set: CardSet, *, manifest: TutorialManifest) -> list[CardJob]:
    jobs: list[CardJob] = []
    for flow in manifest.flows:
        if flow.result is None:
            continue
        jobs.append(
            CardJob(
                job_id=f"{card_set.resolved_key_prefix()}{flow.flow_id}",
                template=card_set.template,
                format=card_set.format,
                theme=card_set.theme,
                tutorial=flow.flow_id,
                stops=card_set.stops,
                file_type=card_set.file_type,
                output=f"{card_set.resolved_directory()}/{flow.flow_id}.{card_set.file_type}",
            ),
        )
    return jobs


class Frame(BaseModel):
    """Represents one picture on a card, with the label under it."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    path: Path
    label: str


class CardContent(BaseModel):
    """Represents a job's content after its strip or tutorial is looked up."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    frames: list[Frame]
    title: str
    subtitle: str
    link: str
    tagline: str | None = None
    source: str


def resolve_content(
    job: CardJob,
    *,
    spec: PromoSpec,
    index: SweepIndex,
    manifest: TutorialManifest | None,
    library_root: Path,
    promo_root: Path | None = None,
) -> CardContent:
    """Return the frames and text that a job draws.

    Args:
        job: The card to resolve.
        spec: The specification, for the text and the brand.
        index: The library's index.
        manifest: The tutorial manifest, or None when the library does not have one.
        library_root: The root of the library on disk.
        promo_root: The root of the promo tree, from which gallery pictures are read. None when
            no job names a gallery picture.

    Raises:
        ValueError: If the job does not name anything to draw, or names it incorrectly.
    """
    if job.gallery is not None:
        if promo_root is None:
            raise ValueError(f"job {job.job_id!r} names a gallery picture, and there is no promo tree to read it from")
        if job.title is None or job.subtitle is None:
            raise ValueError(f"job {job.job_id!r} names a gallery picture, so it needs its own title and subtitle")
        picture = promo_root / GALLERY_DIRECTORY_NAME / job.gallery / GALLERY_RESULT_FILENAME
        if not picture.is_file():
            raise ValueError(
                f"job {job.job_id!r} names gallery picture {job.gallery!r}, and there is none at {picture}"
            )
        return CardContent(
            frames=[Frame(path=picture, label=job.gallery)],
            title=job.title,
            subtitle=job.subtitle,
            link=job.link or spec.brand.url,
            tagline=job.tagline,
            source=f"gallery/{job.gallery}",
        )

    if job.tutorial is not None:
        if manifest is None:
            raise ValueError(f"job {job.job_id!r} names a tutorial, and there is no manifest")
        flow = find_flow(manifest, job.tutorial)
        flow_root = library_root / TUTORIALS_DIRECTORY_NAME / flow.directory
        result = flow_root / str(flow.result)
        before = flow.before or flow.source
        frames = [Frame(path=result, label=flow.title)]
        if job.template is CardTemplate.PAIR and before is not None:
            frames = [
                Frame(path=flow_root / before, label=spec.pair_labels[0]),
                Frame(path=result, label=spec.pair_labels[1]),
            ]
        return CardContent(
            frames=frames,
            title=job.title or flow.title,
            subtitle=job.subtitle or flow.summary,
            link=job.link or spec.brand.url,
            tagline=job.tagline,
            source=flow.flow_id,
        )

    if job.strip is not None:
        strip = find_strip(index, job.strip)
    elif job.parameter is not None:
        strip = preferred_strip(index, parameter=job.parameter, override=spec.preferred_strips.get(job.parameter))
    else:
        raise ValueError(f"job {job.job_id!r} names neither a strip, a parameter nor a tutorial")

    stops = select_stops(
        strip, "preview" if job.template is CardTemplate.PAIR and job.stops == "spread:6" else job.stops
    )
    if job.template is CardTemplate.PAIR and len(stops) != 2:
        raise ValueError(f"a pair needs exactly two stops, and {job.job_id!r} selected {len(stops)}")
    if job.template is CardTemplate.HERO:
        stops = stops[-1:]
    frames = [Frame(path=library_root / stop.file, label=stop.label) for stop in stops]
    parameter = str(strip.parameter)
    return CardContent(
        frames=frames,
        title=job.title or parameter_title(spec, parameter),
        subtitle=job.subtitle
        or spec.subtitles.get(parameter)
        or f"{strip.image_model.name} · {strip.scene.label.lower()} · seed {strip.held.seed}",
        link=job.link or spec.brand.url,
        tagline=job.tagline,
        source=strip.strip_id,
    )


def render_card(
    job: CardJob,
    *,
    spec: PromoSpec,
    index: SweepIndex,
    manifest: TutorialManifest | None,
    library_root: Path,
    fonts: Fonts,
    spec_root: Path,
    promo_root: Path | None = None,
) -> Image.Image:
    """Draw one job and return the image.

    Args:
        job: The card to draw.
        spec: The specification, for the brand, palette and text.
        index: The library's index.
        manifest: The tutorial manifest, or None when the library does not have one.
        library_root: The root of the library on disk.
        fonts: The loaded typefaces.
        spec_root: The specification's directory, which its logo path is relative to.
        promo_root: The root of the promo tree, for a job that draws a gallery picture.

    Raises:
        ValueError: If the job does not name anything drawable.
        FileNotFoundError: If a frame named in the index is not on disk.
    """
    content = resolve_content(
        job, spec=spec, index=index, manifest=manifest, library_root=library_root, promo_root=promo_root
    )
    width, height = CANVAS_SIZES[job.format]
    palette = spec.palettes[job.theme]
    scale = width / 1200 if width >= height else width / 1080
    canvas = Image.new("RGB", (width, height), palette.background)
    draw = ImageDraw.Draw(canvas)
    margin = round(56 * scale)

    overlay_text = _OverlayText(canvas, scale=scale) if job.template is CardTemplate.HERO else None
    if overlay_text is not None:
        _draw_hero(canvas, content, fonts=fonts, scale=scale, margin=margin, text=overlay_text)
    else:
        header_bottom = _draw_header(
            draw, content, palette=palette, fonts=fonts, scale=scale, margin=margin, width=width
        )
        footer_top = height - margin - round(44 * scale)
        area = (margin, header_bottom + round(28 * scale), width - margin, footer_top - round(24 * scale))
        # A tutorial without a source picture, such as text to image, has one frame, and shows
        # it alone.
        if job.template is CardTemplate.PAIR and len(content.frames) == 2:
            _draw_pair(canvas, content.frames, area=area, palette=palette, fonts=fonts, scale=scale)
        else:
            _draw_frames(canvas, content.frames, area=area, palette=palette, fonts=fonts, scale=scale)

    _draw_footer(
        canvas,
        spec.brand,
        link=content.link,
        tagline=content.tagline,
        palette=palette,
        fonts=fonts,
        scale=scale,
        margin=margin,
        spec_root=spec_root,
        overlay_text=overlay_text,
    )
    return canvas


class _OverlayText:
    """White text drawn over a picture, with a dark outline and shadow around each glyph.

    The picture behind the text can be any colour, so the text has its own contrast: a soft glow
    in the shape of the letters, then an outline, then the fill. Runs are queued and drawn in those
    three passes, so one run's outline does not cover a neighbouring run's fill.
    """

    OUTLINE_RATIO = 0.075
    """Outline width as a share of the face size."""

    GLOW_RATIO = 0.16
    """Blur radius of the glow as a share of the largest queued face size."""

    def __init__(self, canvas: Image.Image, *, scale: float) -> None:
        """Store the canvas. Nothing is drawn until ``draw``."""
        self._canvas = canvas
        # Small faces use this minimum outline width, because a thinner one is too faint at preview size.
        self._least_outline = round(3 * scale)
        self._runs: list[tuple[tuple[float, float], str, ImageFont.FreeTypeFont, str]] = []

    def add(self, position: tuple[float, float], words: str, *, font: ImageFont.FreeTypeFont, fill: str) -> None:
        """Queue one run of text."""
        self._runs.append((position, words, font, fill))

    def _outline_width(self, font: ImageFont.FreeTypeFont) -> int:
        return max(self._least_outline, round(font.size * self.OUTLINE_RATIO))

    def draw(self) -> None:
        """Draw every queued run onto the canvas and empty the queue."""
        if not self._runs:
            return
        glow = Image.new("L", self._canvas.size, 0)
        glow_draw = ImageDraw.Draw(glow)
        for position, words, font, _ in self._runs:
            width = self._outline_width(font)
            glow_draw.text(position, words, font=font, fill=255, stroke_width=width, stroke_fill=255)
        radius = max(font.size for _, _, font, _ in self._runs) * self.GLOW_RATIO
        glow = glow.filter(ImageFilter.GaussianBlur(radius)).point(lambda value: min(255, int(value * 1.6)))
        self._canvas.paste(Image.new("RGB", self._canvas.size, OVERLAY_OUTLINE), (0, 0), glow)

        draw = ImageDraw.Draw(self._canvas)
        for position, words, font, _ in self._runs:
            width = self._outline_width(font)
            draw.text(
                position, words, font=font, fill=OVERLAY_OUTLINE, stroke_width=width, stroke_fill=OVERLAY_OUTLINE
            )
        for position, words, font, fill in self._runs:
            draw.text(position, words, font=font, fill=fill)
        self._runs.clear()


def _draw_header(
    draw: ImageDraw.ImageDraw,
    content: CardContent,
    *,
    palette: Palette,
    fonts: Fonts,
    scale: float,
    margin: int,
    width: int,
) -> int:
    """Draw the title and subtitle at the top, and return the y coordinate below which the picture area starts."""
    title_font = fonts.get("bold", round(54 * scale))
    subtitle_font = fonts.get("regular", round(24 * scale))
    y = margin
    draw.text((margin, y), content.title, font=title_font, fill=palette.ink)
    y += _text_height(title_font) + round(10 * scale)
    subtitle = _ellipsise(content.subtitle, font=subtitle_font, max_width=width - 2 * margin)
    draw.text((margin, y), subtitle, font=subtitle_font, fill=palette.muted)
    return y + _text_height(subtitle_font)


def _draw_footer(
    canvas: Image.Image,
    brand: BrandSpec,
    *,
    link: str,
    tagline: str | None,
    palette: Palette,
    fonts: Fonts,
    scale: float,
    margin: int,
    spec_root: Path,
    overlay_text: _OverlayText | None,
) -> None:
    """Draw the brand line along the bottom: logo and name at the left, tagline and address at the right.

    Over a picture, the text goes through ``overlay_text``, which is drawn here once the line is
    queued. The logo is pasted afterwards, so the glow does not dim it.
    """
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    name_font = fonts.get("bold", round(28 * scale))
    small_font = fonts.get("mono", round(20 * scale))
    ink = OVERLAY_INK if overlay_text is not None else palette.ink
    muted = OVERLAY_MUTED if overlay_text is not None else palette.muted

    def put(position: tuple[float, float], words: str, *, font: ImageFont.FreeTypeFont, fill: str) -> None:
        if overlay_text is not None:
            overlay_text.add(position, words, font=font, fill=fill)
        else:
            draw.text(position, words, font=font, fill=fill)

    line_height = round(44 * scale)
    y = height - margin - line_height
    x = margin
    logo_at: tuple[Image.Image, tuple[int, int]] | None = None
    if brand.logo is not None:
        logo_path = spec_root / brand.logo
        if logo_path.is_file():
            with Image.open(logo_path) as logo:
                logo_height = line_height
                logo_width = max(1, round(logo.width * logo_height / logo.height))
                resized = logo.convert("RGBA").resize((logo_width, logo_height), Image.Resampling.LANCZOS)
                logo_at = (resized, (x, y))
                x += logo_width + round(14 * scale)
    put((x, y + (line_height - _text_height(name_font)) // 2), brand.name, font=name_font, fill=ink)

    address = link
    tagline = tagline if tagline is not None else brand.tagline
    address_width = draw.textlength(address, font=small_font)
    tagline_width = draw.textlength(tagline, font=small_font)
    right = width - margin
    put((right - address_width, y + line_height - _text_height(small_font)), address, font=small_font, fill=muted)
    put((right - tagline_width, y), tagline, font=small_font, fill=muted)
    if overlay_text is not None:
        overlay_text.draw()
    if logo_at is not None:
        canvas.paste(logo_at[0], logo_at[1], logo_at[0])


def _draw_frames(
    canvas: Image.Image,
    frames: list[Frame],
    *,
    area: tuple[int, int, int, int],
    palette: Palette,
    fonts: Fonts,
    scale: float,
) -> None:
    """Lay the frames out in the rows and columns that give them the most room, with a value under each."""
    left, top, right, bottom = area
    gap = round(18 * scale)
    label_font = fonts.get("mono", round(22 * scale))
    label_height = _text_height(label_font) + round(22 * scale)
    aspect = _aspect(frames[0].path)
    columns, cell_width, cell_height = _best_layout(
        len(frames),
        area_width=right - left,
        area_height=bottom - top,
        gap=gap,
        label_height=label_height,
        aspect=aspect,
    )
    rows = math.ceil(len(frames) / columns)
    block_width = columns * cell_width + (columns - 1) * gap
    block_height = rows * (cell_height + label_height) + (rows - 1) * gap
    origin_x = left + (right - left - block_width) // 2
    origin_y = top + (bottom - top - block_height) // 2
    for position, frame in enumerate(frames):
        row, column = divmod(position, columns)
        x = origin_x + column * (cell_width + gap)
        y = origin_y + row * (cell_height + label_height + gap)
        _paste_fitted(canvas, frame.path, box=(x, y, x + cell_width, y + cell_height))
        _draw_pill(
            canvas,
            frame.label,
            centre_x=x + cell_width // 2,
            top=y + cell_height + round(10 * scale),
            font=label_font,
            palette=palette,
            scale=scale,
            max_width=cell_width,
        )


def _best_layout(
    count: int,
    *,
    area_width: int,
    area_height: int,
    gap: int,
    label_height: int,
    aspect: float,
) -> tuple[int, int, int]:
    """Return the column count and cell size that leave the frames largest inside the area."""
    best: tuple[int, int, int] = (1, 1, 1)
    best_area = 0
    for columns in range(1, count + 1):
        rows = math.ceil(count / columns)
        cell_width = (area_width - gap * (columns - 1)) // columns
        cell_height = (area_height - (gap + label_height) * rows + gap) // rows
        if cell_width <= 0 or cell_height <= 0:
            continue
        fitted_width = min(cell_width, round(cell_height * aspect))
        fitted_height = min(cell_height, round(cell_width / aspect))
        if fitted_width * fitted_height > best_area:
            best_area = fitted_width * fitted_height
            best = (columns, fitted_width, fitted_height)
    return best


def _draw_pair(
    canvas: Image.Image,
    frames: list[Frame],
    *,
    area: tuple[int, int, int, int],
    palette: Palette,
    fonts: Fonts,
    scale: float,
) -> None:
    """Draw two frames edge to edge with a divider and handle between them, with a value at the bottom of each."""
    left, top, right, bottom = area
    label_font = fonts.get("mono", round(24 * scale))
    aspect = _aspect(frames[0].path)
    pair_width = right - left
    pair_height = bottom - top
    half_width = min(pair_width // 2, round(pair_height * aspect))
    half_height = min(pair_height, round(half_width / aspect))
    origin_x = left + (pair_width - 2 * half_width) // 2
    origin_y = top + (pair_height - half_height) // 2
    boxes = [
        (origin_x, origin_y, origin_x + half_width, origin_y + half_height),
        (origin_x + half_width, origin_y, origin_x + 2 * half_width, origin_y + half_height),
    ]
    for frame, box in zip(frames, boxes, strict=True):
        _paste_covered(canvas, frame.path, box=box)
    draw = ImageDraw.Draw(canvas)
    divider_x = origin_x + half_width
    line_width = max(2, round(4 * scale))
    draw.line([(divider_x, origin_y), (divider_x, origin_y + half_height)], fill=palette.divider, width=line_width)
    radius = round(22 * scale)
    centre_y = origin_y + half_height // 2
    draw.ellipse(
        (divider_x - radius, centre_y - radius, divider_x + radius, centre_y + radius),
        fill=palette.divider,
        outline=palette.background,
        width=line_width,
    )
    # Two small triangles in the handle, drawn rather than set in type so no face has to include them.
    arrow = round(radius * 0.36)
    gap_from_centre = round(radius * 0.18)
    draw.polygon(
        [
            (divider_x - gap_from_centre, centre_y - arrow),
            (divider_x - gap_from_centre - arrow, centre_y),
            (divider_x - gap_from_centre, centre_y + arrow),
        ],
        fill=palette.background,
    )
    draw.polygon(
        [
            (divider_x + gap_from_centre, centre_y - arrow),
            (divider_x + gap_from_centre + arrow, centre_y),
            (divider_x + gap_from_centre, centre_y + arrow),
        ],
        fill=palette.background,
    )
    inset = round(18 * scale)
    for frame, box, align_right in zip(frames, boxes, (False, True), strict=True):
        pill_width = round(draw.textlength(frame.label, font=label_font)) + round(28 * scale)
        centre_x = box[2] - inset - pill_width // 2 if align_right else box[0] + inset + pill_width // 2
        _draw_pill(
            canvas,
            frame.label,
            centre_x=centre_x,
            top=box[3] - inset - _text_height(label_font) - round(16 * scale),
            font=label_font,
            palette=palette,
            scale=scale,
            max_width=half_width - 2 * inset,
        )


def _draw_hero(
    canvas: Image.Image,
    content: CardContent,
    *,
    fonts: Fonts,
    scale: float,
    margin: int,
    text: _OverlayText,
) -> None:
    """Fill the canvas with the frame, darken its lower part, and queue the title and subtitle on ``text``."""
    width, height = canvas.size
    _paste_covered(canvas, content.frames[0].path, box=(0, 0, width, height))
    shade_height = round(height * 0.55)
    shade = Image.new("RGBA", (width, shade_height), (0, 0, 0, 0))
    shade_draw = ImageDraw.Draw(shade)
    for row in range(shade_height):
        alpha = round(200 * (row / shade_height) ** 1.4)
        shade_draw.line([(0, row), (width, row)], fill=(8, 12, 24, alpha))
    canvas.paste(shade, (0, height - shade_height), shade)

    draw = ImageDraw.Draw(canvas)
    title_font = fonts.get("bold", round(64 * scale))
    # A link preview is shown at about a third of the canvas, so the subtitle is set large enough to
    # be read there and is allowed a second line before it is cut.
    subtitle_font = fonts.get("regular", round(34 * scale))
    text_width = width - 2 * margin
    lines = _wrap(content.title, font=title_font, max_width=text_width, draw=draw)
    subtitle_lines = _wrap(content.subtitle, font=subtitle_font, max_width=text_width, draw=draw)
    if len(subtitle_lines) > 2:
        rest = " ".join(subtitle_lines[1:])
        subtitle_lines = [subtitle_lines[0], _ellipsise(rest, font=subtitle_font, max_width=text_width)]
    footer_height = round(44 * scale) + margin
    subtitle_line_height = _text_height(subtitle_font) + round(2 * scale)
    subtitle_top = height - footer_height - round(24 * scale) - subtitle_line_height * len(subtitle_lines)
    for position, line in enumerate(subtitle_lines):
        text.add(
            (margin, subtitle_top + position * subtitle_line_height), line, font=subtitle_font, fill=OVERLAY_MUTED
        )
    line_height = _text_height(title_font) + round(6 * scale)
    y = subtitle_top - round(14 * scale) - line_height * len(lines)
    for line in lines:
        text.add((margin, y), line, font=title_font, fill=OVERLAY_INK)
        y += line_height


def _draw_pill(
    canvas: Image.Image,
    text: str,
    *,
    centre_x: int,
    top: int,
    font: ImageFont.FreeTypeFont,
    palette: Palette,
    scale: float,
    max_width: int,
) -> None:
    """Draw a value in a rounded capsule centred on ``centre_x``."""
    draw = ImageDraw.Draw(canvas)
    # In a nine-column poster, a full sampler name is more useful than its first four letters, so
    # the font shrinks to about two thirds of its size before the text is cut.
    room = max(1, max_width - round(28 * scale))
    smallest = max(8, round(font.size * 0.66))
    while font.getlength(text) > room and font.size > smallest:
        font = font.font_variant(size=font.size - 1)
    label = _ellipsise(text, font=font, max_width=room)
    text_width = draw.textlength(label, font=font)
    pad_x = round(14 * scale)
    pad_y = round(8 * scale)
    pill_width = round(text_width) + 2 * pad_x
    pill_height = _text_height(font) + 2 * pad_y
    left = centre_x - pill_width // 2
    draw.rounded_rectangle(
        (left, top, left + pill_width, top + pill_height),
        radius=pill_height // 2,
        fill=palette.pill_background,
    )
    draw.text((left + pad_x, top + pad_y), label, font=font, fill=palette.pill_ink)


def _paste_fitted(canvas: Image.Image, path: Path, *, box: tuple[int, int, int, int]) -> None:
    """Paste the picture scaled to fit inside the box, centred."""
    left, top, right, bottom = box
    with Image.open(path) as source:
        picture = source.convert("RGB")
    picture.thumbnail((right - left, bottom - top), Image.Resampling.LANCZOS)
    x = left + (right - left - picture.width) // 2
    y = top + (bottom - top - picture.height) // 2
    canvas.paste(picture, (x, y))


def _paste_covered(canvas: Image.Image, path: Path, *, box: tuple[int, int, int, int]) -> None:
    """Paste the picture scaled to cover the box, cropped about its centre."""
    left, top, right, bottom = box
    box_width = right - left
    box_height = bottom - top
    with Image.open(path) as source:
        picture = source.convert("RGB")
    factor = max(box_width / picture.width, box_height / picture.height)
    scaled = picture.resize(
        (max(1, math.ceil(picture.width * factor)), max(1, math.ceil(picture.height * factor))),
        Image.Resampling.LANCZOS,
    )
    crop_left = (scaled.width - box_width) // 2
    crop_top = (scaled.height - box_height) // 2
    canvas.paste(scaled.crop((crop_left, crop_top, crop_left + box_width, crop_top + box_height)), (left, top))


def _aspect(path: Path) -> float:
    with Image.open(path) as picture:
        return picture.width / picture.height


def _text_height(font: ImageFont.FreeTypeFont) -> int:
    ascent, descent = font.getmetrics()
    return ascent + descent


def _ellipsise(text: str, *, font: ImageFont.FreeTypeFont, max_width: int) -> str:
    """Return the text, cut with an ellipsis where it would run past the width."""
    if font.getlength(text) <= max_width:
        return text
    ellipsis = "…"
    kept = text
    while kept and font.getlength(kept + ellipsis) > max_width:
        kept = kept[:-1]
    return kept.rstrip() + ellipsis


def _wrap(text: str, *, font: ImageFont.FreeTypeFont, max_width: int, draw: ImageDraw.ImageDraw) -> list[str]:
    """Break the text into lines no wider than ``max_width``, at spaces."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and draw.textlength(candidate, font=font) > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def read_tutorial_manifest(library_root: Path) -> TutorialManifest | None:
    """Return the tutorial manifest, or None when the tutorial campaign has not run."""
    manifest_path = library_root / TUTORIALS_DIRECTORY_NAME / TUTORIAL_MANIFEST_FILENAME
    if not manifest_path.is_file():
        return None
    return TutorialManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))


def write_cards(
    spec: PromoSpec,
    *,
    spec_root: Path,
    library_root: Path,
    promo_root: Path,
    only: list[str] | None = None,
    force: bool = False,
) -> list[CardOutcome]:
    """Draw every job into the promo tree and write the manifest at its root.

    A card already on disk is kept unless ``force`` is set, so re-running after adding
    one job draws one card. The manifest is rewritten from every job each time, whether or not its
    card was redrawn or the run was restricted to some jobs, so it always describes the whole set.

    Args:
        spec: The specification.
        spec_root: The specification's directory, which its font and logo paths are relative to.
        library_root: The root of the library on disk, where the frames are read from.
        promo_root: The root of the promo tree, where the cards and their manifest are written.
        only: Job keys to restrict the run to, or None for all of them.
        force: Whether to redraw cards that are already on disk.

    Returns:
        One outcome per job drawn or kept.

    Raises:
        FileNotFoundError: If there is no index to read.
        ValueError: If a job does not name anything drawable, or ``only`` names a job that does not exist.
    """
    index_path = library_root / INDEX_FILENAME
    if not index_path.is_file():
        raise FileNotFoundError(f"no {INDEX_FILENAME} at {library_root}: render the library first")
    index = SweepIndex.model_validate_json(index_path.read_text(encoding="utf-8"))
    manifest = read_tutorial_manifest(library_root)
    jobs = expand_jobs(spec, index=index, manifest=manifest)
    contents = {
        job.job_id: resolve_content(
            job, spec=spec, index=index, manifest=manifest, library_root=library_root, promo_root=promo_root
        )
        for job in jobs
    }
    if only:
        missing = sorted(set(only) - set(contents))
        if missing:
            raise ValueError(f"no such job: {', '.join(missing)}")

    fonts = Fonts(spec.fonts, base=spec_root)
    root = promo_root
    outcomes: list[CardOutcome] = []
    for job in jobs:
        if only and job.job_id not in only:
            continue
        relative = job.output_path()
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        width, height = CANVAS_SIZES[job.format]
        written = False
        if force or not destination.is_file():
            card = render_card(
                job,
                spec=spec,
                index=index,
                manifest=manifest,
                library_root=library_root,
                fonts=fonts,
                spec_root=spec_root,
                promo_root=promo_root,
            )
            if job.file_type == "jpg":
                card.save(destination, format="JPEG", quality=90, optimize=True, progressive=True)
            else:
                card.save(destination, format="PNG", optimize=True)
            written = True
            logger.info("wrote %s", destination)
        outcomes.append(
            CardOutcome(
                key=job.job_id,
                relative_path=relative,
                width=width,
                height=height,
                written=written,
                size_bytes=destination.stat().st_size,
            ),
        )

    entries = {
        job.job_id: CardManifestEntry(
            file=job.output_path(),
            width=CANVAS_SIZES[job.format][0],
            height=CANVAS_SIZES[job.format][1],
            template=job.template,
            source=contents[job.job_id].source,
        )
        for job in jobs
    }
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / CARDS_MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(CardsManifest(cards=entries).model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    return outcomes
