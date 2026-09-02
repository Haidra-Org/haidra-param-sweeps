# SPDX-FileCopyrightText: 2026 Tazlin
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Check that the cards a specification asks for are the ones drawn, from the frames it names.

The cards exist to be unfurled and posted, so the checks are about what lands on the canvas: that
every chosen frame is on it, that the canvas is the size the format promises, that a set yields
one card per parameter at the path a page can predict, and that the manifest lists it all.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from param_sweeps.outputs import write_index
from param_sweeps.promo import (
    CANVAS_SIZES,
    CARDS_MANIFEST_FILENAME,
    CardFormat,
    CardJob,
    CardSet,
    CardTemplate,
    PromoSpec,
    expand_jobs,
    resolve_content,
    select_stops,
    spread_positions,
    write_cards,
)
from param_sweeps.publish import PublishTree, published_relative_paths, read_index
from param_sweeps.spec import SweepSpec

STOP_COLOURS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
"""One flat colour per stop, so a frame on the canvas is a colour on the canvas."""

BRAND: dict[str, Any] = {"name": "Test Horde", "url": "example.test/docs"}


def three_stop_spec() -> SweepSpec:
    """Return a specification holding one three-stop steps strip and one two-stop cfg strip."""
    return SweepSpec.model_validate(
        {
            "schema_version": 1,
            "seed": 1234567890,
            "anchor": {
                "sampler_name": "k_euler_a",
                "cfg_scale": 7.5,
                "steps": 30,
                "karras": True,
                "denoising_strength": 1.0,
                "hires_fix": False,
                "tiling": False,
            },
            "models": [
                {
                    "slug": "small",
                    "horde_name": "stable_diffusion",
                    "baseline": "stable_diffusion_1",
                    "native_width": 512,
                    "native_height": 512,
                    "anchor_clip_skip": 1,
                    "approximate_download_size_gb": 4.3,
                    "note": "A model to hang a rule on.",
                },
            ],
            "scenes": [{"scene_id": "thing", "label": "A thing", "prompt": "a thing", "derived_from": "nothing"}],
            "sweeps": [
                {
                    "kind": "strip",
                    "parameter": "steps",
                    "rationale": "Frames to draw.",
                    "caption": "Three step counts.",
                    "model_slugs": ["small"],
                    "scene_ids": ["thing"],
                    "presentation": "strip",
                    "stops": [
                        {"value": 4, "steps": 4, "caption": "Few."},
                        {"value": 15, "steps": 15, "caption": "Some."},
                        {"value": 30, "steps": 30, "caption": "Enough."},
                    ],
                },
                {
                    "kind": "strip",
                    "parameter": "cfg_scale",
                    "rationale": "A second parameter.",
                    "caption": "Two guidance values.",
                    "model_slugs": ["small"],
                    "scene_ids": ["thing"],
                    "presentation": "strip",
                    "stops": [
                        {"value": 3, "cfg_scale": 3, "caption": "Loose."},
                        {"value": 12, "cfg_scale": 12, "caption": "Tight."},
                    ],
                },
            ],
        },
    )


@pytest.fixture
def library(tmp_path: Path) -> Path:
    """Render flat-colour frames for every job and index them, returning the library root."""
    spec = three_stop_spec()
    for job in spec.expand_jobs():
        image_path = tmp_path / job.relative_image_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        colour = STOP_COLOURS[job.stop_index % len(STOP_COLOURS)]
        Image.new("RGB", (64, 64), colour).save(image_path, format="WEBP", lossless=True)
        (tmp_path / job.relative_sidecar_path).write_text("{}\n", encoding="utf-8")
    write_index(spec=spec, output_root=tmp_path)
    return tmp_path


def promo_spec(**overrides: Any) -> PromoSpec:
    return PromoSpec.model_validate({"brand": BRAND, **overrides})


def promo_root(library: Path) -> Path:
    return library.parent / "promo"


def test_spread_keeps_both_ends_and_never_repeats() -> None:
    assert spread_positions(12, 6) == [0, 2, 4, 7, 9, 11]
    assert spread_positions(3, 6) == [0, 1, 2]
    assert spread_positions(5, 1) == [0]


def test_select_stops_answers_every_form(library: Path) -> None:
    strip = read_index(output_root=library).strips[0]

    assert [stop.label for stop in select_stops(strip, "all")] == ["4", "15", "30"]
    assert [stop.label for stop in select_stops(strip, "preview")] == ["4", "30"]
    assert [stop.label for stop in select_stops(strip, "spread:2")] == ["4", "30"]
    assert [stop.label for stop in select_stops(strip, "first:2")] == ["4", "15"]
    assert [stop.label for stop in select_stops(strip, [2, 0])] == ["30", "4"]
    with pytest.raises(ValueError, match="unrecognised"):
        select_stops(strip, "middle")
    with pytest.raises(ValueError, match="outside"):
        select_stops(strip, [3])


def test_a_parameter_set_yields_one_card_per_parameter_at_a_predictable_path(library: Path) -> None:
    spec = promo_spec(
        sets=[{"every": "parameter", "template": "strip"}],
        skip_parameters=["cfg_scale"],
    )

    jobs = expand_jobs(spec, index=read_index(output_root=library), manifest=None)

    assert [(job.job_id, job.output_path(), job.strip) for job in jobs] == [
        ("sweep:steps", "sweeps/steps.jpg", "steps/small/thing"),
    ]


def test_a_card_holds_every_chosen_frame_on_the_canvas_its_format_names(library: Path) -> None:
    spec = promo_spec(
        jobs=[
            {
                "id": "steps-og",
                "template": "strip",
                "format": "og",
                "stops": "all",
                "file_type": "png",
                "strip": "steps/small/thing",
            },
        ],
    )

    outcomes = write_cards(spec, spec_root=library, library_root=library, promo_root=promo_root(library))

    assert [(outcome.key, outcome.relative_path, outcome.written) for outcome in outcomes] == [
        ("steps-og", "steps-og.png", True),
    ]
    with Image.open(promo_root(library) / "steps-og.png") as card:
        assert card.size == CANVAS_SIZES[CardFormat.OG]
        colours = {colour for _, colour in card.convert("RGB").getcolors(maxcolors=1_000_000) or []}
    for colour in STOP_COLOURS:
        assert colour in colours


def test_a_promo_tree_without_a_manifest_cannot_be_published(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="draw the cards first"):
        published_relative_paths(output_root=tmp_path, tree=PublishTree.PROMO)


def test_the_manifest_lists_every_card_by_key_and_the_promo_tree_publishes_them_apart(library: Path) -> None:
    spec = promo_spec(sets=[{"every": "parameter", "template": "pair"}])

    write_cards(spec, spec_root=library, library_root=library, promo_root=promo_root(library))

    manifest = json.loads((promo_root(library) / CARDS_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["cards"] == {
        "sweep:steps": {
            "file": "sweeps/steps.jpg",
            "width": 1200,
            "height": 630,
            "template": "pair",
            "source": "steps/small/thing",
        },
        "sweep:cfg_scale": {
            "file": "sweeps/cfg_scale.jpg",
            "width": 1200,
            "height": 630,
            "template": "pair",
            "source": "cfg-scale/small/thing",
        },
    }
    published = published_relative_paths(output_root=promo_root(library), tree=PublishTree.PROMO)
    assert published == ["cards.json", "sweeps/steps.jpg", "sweeps/cfg_scale.jpg"]
    library_paths = published_relative_paths(output_root=library)
    assert not any(path.startswith("sweeps/") or path == "cards.json" for path in library_paths)


def test_a_rerun_keeps_cards_unless_forced_and_still_rewrites_the_manifest(library: Path) -> None:
    spec = promo_spec(sets=[{"every": "parameter", "template": "strip"}])
    write_cards(spec, spec_root=library, library_root=library, promo_root=promo_root(library))
    manifest_path = promo_root(library) / CARDS_MANIFEST_FILENAME
    manifest_path.unlink()

    kept = write_cards(spec, spec_root=library, library_root=library, promo_root=promo_root(library))
    forced = write_cards(spec, spec_root=library, library_root=library, promo_root=promo_root(library), force=True)

    assert [outcome.written for outcome in kept] == [False, False]
    assert [outcome.written for outcome in forced] == [True, True]
    assert manifest_path.is_file()


def test_restricting_to_one_job_draws_one_card_and_lists_all(library: Path) -> None:
    spec = promo_spec(sets=[{"every": "parameter", "template": "strip"}])

    outcomes = write_cards(
        spec, spec_root=library, library_root=library, promo_root=promo_root(library), only=["sweep:cfg_scale"]
    )

    assert [outcome.key for outcome in outcomes] == ["sweep:cfg_scale"]
    manifest = json.loads((promo_root(library) / CARDS_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert sorted(manifest["cards"]) == ["sweep:cfg_scale", "sweep:steps"]
    with pytest.raises(ValueError, match="no such job: sweep:seed"):
        write_cards(spec, spec_root=library, library_root=library, promo_root=promo_root(library), only=["sweep:seed"])


def test_a_pair_takes_the_strips_own_preview_and_a_hero_takes_the_last_frame(library: Path) -> None:
    spec = promo_spec(titles={"steps": "Steps"})
    index = read_index(output_root=library)
    pair = CardJob(job_id="p", template=CardTemplate.PAIR, strip="steps/small/thing")
    hero = CardJob(job_id="h", template=CardTemplate.HERO, strip="steps/small/thing", title="Look, then decide")

    pair_content = resolve_content(pair, spec=spec, index=index, manifest=None, library_root=library)
    hero_content = resolve_content(hero, spec=spec, index=index, manifest=None, library_root=library)

    assert [frame.label for frame in pair_content.frames] == ["4", "30"]
    assert pair_content.title == "Steps"
    assert [frame.label for frame in hero_content.frames] == ["30"]
    assert hero_content.title == "Look, then decide"


def test_a_job_naming_nothing_or_a_missing_strip_is_refused(library: Path) -> None:
    spec = promo_spec()
    index = read_index(output_root=library)

    with pytest.raises(ValueError, match="neither a strip"):
        resolve_content(
            CardJob(job_id="x", template=CardTemplate.STRIP),
            spec=spec,
            index=index,
            manifest=None,
            library_root=library,
        )
    with pytest.raises(ValueError, match="no strip"):
        resolve_content(
            CardJob(job_id="x", template=CardTemplate.STRIP, strip="steps/big/thing"),
            spec=spec,
            index=index,
            manifest=None,
            library_root=library,
        )
    with pytest.raises(ValueError, match="needs the tutorial manifest"):
        expand_jobs(
            promo_spec(sets=[CardSet(every="tutorial", template=CardTemplate.HERO)]),
            index=index,
            manifest=None,
        )


def test_a_gallery_manifest_beside_the_cards_is_published_with_them(library: Path) -> None:
    spec = promo_spec(sets=[{"every": "parameter", "template": "strip"}])
    write_cards(spec, spec_root=library, library_root=library, promo_root=promo_root(library))
    gallery = promo_root(library) / "gallery"
    gallery.mkdir()
    (gallery / "gallery.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "images": [
                    {
                        "id": "fox",
                        "file": "gallery/fox/result.webp",
                        "alchemy": {"caption": {"file": "gallery/fox/caption.txt"}, "nsfw": {}},
                    },
                ],
                "texts": [{"id": "poem", "file": "gallery/text/poem/result.txt"}],
            },
        ),
        encoding="utf-8",
    )

    published = published_relative_paths(output_root=promo_root(library), tree=PublishTree.PROMO)

    assert published == [
        "cards.json",
        "sweeps/steps.jpg",
        "sweeps/cfg_scale.jpg",
        "gallery/gallery.json",
        "gallery/fox/result.webp",
        "gallery/fox/caption.txt",
        "gallery/text/poem/result.txt",
    ]
