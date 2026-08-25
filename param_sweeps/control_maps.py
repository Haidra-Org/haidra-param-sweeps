# SPDX-FileCopyrightText: 2026 Tazlin
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Publish the control map beside every render that was guided by one.

A strip that walks the ControlNet vocabulary is asking the reader to choose between detectors,
and a detector is only knowable by what it draws. The render alone cannot answer that: two
detectors that read a scene completely differently can land on renders that look alike, and a
detector that found nothing at all in a scene looks, from the render, like one that worked. So
the map is published as evidence in its own right, in the same directory as the render it
guided and under the same filename stem.

The maps are drawn on a horde-image-utilities service, which is the same detector code an
alchemist worker runs. The backend the sweeps render through derives its own map internally and
never hands it back, so asking the service is the only way to publish the real thing rather than
an approximation drawn by whatever else happens to be installed.

A detector the service will not run is recorded as a gap carrying the service's own sentence,
and the render it belongs to is left alone: the render still exists, and saying that no map came
back is more useful than leaving a hole a reader cannot interpret.

Critical public members:

- [`ControlMapOutcome`][param_sweeps.control_maps.ControlMapOutcome]: what became of one map.
- [`draw_control_maps`][param_sweeps.control_maps.draw_control_maps]: draw and publish the maps
  every job in a specification needs.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image
from pydantic import BaseModel, ConfigDict

from param_sweeps.alchemy import ANNOTATION_FORM, AlchemyBackend
from param_sweeps.outputs import WebpEncoding
from param_sweeps.spec import TOOL_ROOT, RenderJob, SweepParameter, SweepSpec

logger = logging.getLogger(__name__)

FROZEN_MODEL_CONFIG = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())


class ControlMapSidecar(BaseModel):
    """Represents the JSON record written beside every published control map."""

    model_config = FROZEN_MODEL_CONFIG

    form: str = ANNOTATION_FORM
    control_type: str
    control_group: str | None
    source: str
    render: str | None = None
    """The render the map guided, and None on a map published once for a whole strip."""

    backend: str
    service_host: str | None
    service_version: str | None
    drawn_at: str
    image: str


class ControlMapOutcome(BaseModel):
    """Represents what became of one job's control map."""

    model_config = FROZEN_MODEL_CONFIG

    control_type: str
    strip_id: str
    relative_path: str
    drawn: bool
    skipped: bool = False
    """Whether the map was already on disk and left alone."""

    unavailable_reason: str | None = None
    """What the service said about a map it would not draw, when it said anything."""


def jobs_needing_control_maps(
    *,
    spec: SweepSpec,
    parameters: Sequence[SweepParameter] | None = None,
) -> list[RenderJob]:
    """Return every job that conditions on a ControlNet and can therefore carry a map.

    Args:
        spec: The specification to read.
        parameters: Restrict to these parameters, or None for every one of them.

    Returns:
        The jobs, in specification order, one per published map.
    """
    wanted = set(parameters) if parameters else None
    return [
        job
        for job in spec.expand_jobs()
        if job.control_type is not None
        and job.source_image is not None
        and (wanted is None or job.parameter in wanted)
    ]


def draw_control_maps(
    *,
    spec: SweepSpec,
    output_root: Path,
    backend: AlchemyBackend,
    parameters: Sequence[SweepParameter] | None = None,
    force: bool = False,
) -> list[ControlMapOutcome]:
    """Draw and publish the control map every strip and every job that conditions on one needs.

    A strip held to a single detector publishes one map beside its own source, because the dial
    such a strip turns is only readable against the map it is holding the render to and the map is
    the same at every stop. A strip that changes detector publishes one map per stop instead,
    beside the render each guided.

    Args:
        spec: The specification to read the jobs from.
        output_root: The root of the output tree, which the maps land inside.
        backend: The prepared backend to ask for each map.
        parameters: Restrict to these parameters, or None for every one of them.
        force: Redraw maps that are already on disk.

    Returns:
        One outcome per published map, in specification order, the strip's own map first.

    Side Effects:
        Writes a WebP and a JSON sidecar beside each source or render whose map the backend drew.
    """
    outcomes: list[ControlMapOutcome] = []
    sources: dict[Path, Image.Image] = {}

    for strip in spec.expand_strips():
        if parameters and strip.parameter not in parameters:
            continue

        strip_map_path = strip.relative_control_map_path
        held_control_type = strip.jobs[0].control_type if strip.jobs else None
        if strip_map_path is not None and strip.source_image is not None and held_control_type is not None:
            outcomes.append(
                _publish_control_map(
                    backend=backend,
                    sources=sources,
                    output_root=output_root,
                    relative_path=strip_map_path,
                    strip_id=strip.strip_id,
                    control_type=held_control_type,
                    control_group=strip.jobs[0].control_group,
                    source_image=strip.source_image,
                    render=None,
                    force=force,
                ),
            )
            continue

        for job in strip.jobs:
            job_map_path = job.relative_control_map_path
            if job.source_image is None or job.control_type is None or job_map_path is None:
                continue
            if parameters and job.parameter not in parameters:
                continue

            outcomes.append(
                _publish_control_map(
                    backend=backend,
                    sources=sources,
                    output_root=output_root,
                    relative_path=job_map_path,
                    strip_id=strip.strip_id,
                    control_type=job.control_type,
                    control_group=job.control_group,
                    source_image=job.source_image,
                    render=job.relative_image_path,
                    force=force,
                ),
            )

    return outcomes


def _publish_control_map(
    *,
    backend: AlchemyBackend,
    sources: dict[Path, Image.Image],
    output_root: Path,
    relative_path: str,
    strip_id: str,
    control_type: str,
    control_group: str | None,
    source_image: str,
    render: str | None,
    force: bool,
) -> ControlMapOutcome:
    """Draw one map on the backend and write it with its sidecar, and return what became of it.

    Args:
        backend: The prepared backend to ask for the map.
        sources: Images already opened, keyed by resolved path, added to as sources are opened.
        output_root: The root of the output tree.
        relative_path: Where the map is published, relative to the output root.
        strip_id: The strip the map belongs to, for the reader of an outcome.
        control_type: The detector to draw with.
        control_group: The detector's family, for the sidecar.
        source_image: The image to draw from, relative to the tool root.
        render: The render the map guided, or None for a map published once for a strip.
        force: Redraw a map that is already on disk.

    Returns:
        The outcome, which says whether the map was drawn, reused, or refused by the service.

    Side Effects:
        Writes a WebP and a JSON sidecar under the output root.
    """
    destination = output_root / relative_path
    if destination.is_file() and not force:
        return ControlMapOutcome(
            control_type=control_type,
            strip_id=strip_id,
            relative_path=relative_path,
            drawn=True,
            skipped=True,
        )

    source_path = (TOOL_ROOT / source_image).resolve()
    if source_path not in sources:
        with Image.open(source_path) as opened:
            sources[source_path] = opened.convert("RGB")

    outcome = backend.apply(
        form=ANNOTATION_FORM,
        control_type=control_type,
        source_image=sources[source_path],
    )

    if outcome.image is None:
        logger.warning(
            "No control map published: control_type=%s strip=%s reason=%s",
            control_type,
            strip_id,
            outcome.unavailable_reason,
        )
        return ControlMapOutcome(
            control_type=control_type,
            strip_id=strip_id,
            relative_path=relative_path,
            drawn=False,
            unavailable_reason=outcome.unavailable_reason,
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    outcome.image.save(
        destination,
        format="WEBP",
        quality=WebpEncoding.QUALITY,
        method=WebpEncoding.METHOD,
    )

    sidecar = ControlMapSidecar(
        control_type=control_type,
        control_group=control_group,
        source=source_image,
        render=render,
        backend=backend.name,
        service_host=backend.service_host,
        service_version=backend.service_version,
        drawn_at=datetime.now(UTC).isoformat(),
        image=destination.name,
    )
    destination.with_suffix(".json").write_text(sidecar.model_dump_json(indent=2) + "\n", encoding="utf-8")

    return ControlMapOutcome(
        control_type=control_type,
        strip_id=strip_id,
        relative_path=relative_path,
        drawn=True,
    )
