# SPDX-FileCopyrightText: 2026 Tazlin
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Check how the command line resolves its output roots: the flag, then the environment, then the default."""

from __future__ import annotations

from pathlib import Path

import pytest

from param_sweeps.cli import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PROMO_OUTPUT_ROOT,
    OUTPUT_ROOT_VARIABLE,
    PROMO_OUTPUT_ROOT_VARIABLE,
    _publish_tree,
    build_parser,
)
from param_sweeps.publish import PublishTree
from param_sweeps.spec import TOOL_ROOT

LIBRARY_COMMANDS: tuple[tuple[str, ...], ...] = (
    ("plan",),
    ("local",),
    ("horde",),
    ("sampler-cost",),
    ("step-preview",),
    ("alchemy",),
    ("review",),
    ("choose", "choices.json"),
    ("control-maps",),
    ("derivatives",),
    ("crops",),
    ("index",),
    ("promo", "--promo-spec", "promo.json"),
)
"""Every subcommand with an ``--output-root`` flag that defaults to the library's root."""


@pytest.fixture
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Return a monkeypatch with both output root variables removed."""
    monkeypatch.delenv(OUTPUT_ROOT_VARIABLE, raising=False)
    monkeypatch.delenv(PROMO_OUTPUT_ROOT_VARIABLE, raising=False)
    return monkeypatch


def test_the_default_roots_are_inside_the_repository() -> None:
    assert DEFAULT_OUTPUT_ROOT == TOOL_ROOT / "output" / "param-sweeps"
    assert DEFAULT_PROMO_OUTPUT_ROOT == TOOL_ROOT / "output" / "promo"


@pytest.mark.parametrize("command", LIBRARY_COMMANDS)
def test_the_library_root_defaults_to_the_repository_output_directory(
    command: tuple[str, ...],
    clean_environment: pytest.MonkeyPatch,
) -> None:
    assert build_parser().parse_args(list(command)).output_root == DEFAULT_OUTPUT_ROOT


@pytest.mark.parametrize("command", LIBRARY_COMMANDS)
def test_the_environment_variable_replaces_the_default_library_root(
    command: tuple[str, ...],
    clean_environment: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    clean_environment.setenv(OUTPUT_ROOT_VARIABLE, str(tmp_path / "library"))

    assert build_parser().parse_args(list(command)).output_root == tmp_path / "library"


@pytest.mark.parametrize("command", LIBRARY_COMMANDS)
def test_the_flag_wins_over_the_environment_variable(
    command: tuple[str, ...],
    clean_environment: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    clean_environment.setenv(OUTPUT_ROOT_VARIABLE, str(tmp_path / "from-environment"))
    arguments = build_parser().parse_args([*command, "--output-root", str(tmp_path / "from-flag")])

    assert arguments.output_root == tmp_path / "from-flag"


def test_a_blank_environment_variable_leaves_the_default(clean_environment: pytest.MonkeyPatch) -> None:
    clean_environment.setenv(OUTPUT_ROOT_VARIABLE, "   ")
    clean_environment.setenv(PROMO_OUTPUT_ROOT_VARIABLE, "")

    assert build_parser().parse_args(["index"]).output_root == DEFAULT_OUTPUT_ROOT
    assert build_parser().parse_args(["promo", "--promo-spec", "promo.json"]).promo_root == DEFAULT_PROMO_OUTPUT_ROOT


def test_the_flows_tree_follows_the_library_root(clean_environment: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert build_parser().parse_args(["flows"]).output_root == DEFAULT_OUTPUT_ROOT / "tutorials"

    clean_environment.setenv(OUTPUT_ROOT_VARIABLE, str(tmp_path))
    assert build_parser().parse_args(["flows"]).output_root == tmp_path / "tutorials"

    flagged = build_parser().parse_args(["flows", "--output-root", str(tmp_path / "elsewhere")])
    assert flagged.output_root == tmp_path / "elsewhere"


def test_the_promo_root_resolves_flag_then_environment_then_default(
    clean_environment: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    promo_command = ["promo", "--promo-spec", "promo.json"]
    assert build_parser().parse_args(promo_command).promo_root == DEFAULT_PROMO_OUTPUT_ROOT

    clean_environment.setenv(PROMO_OUTPUT_ROOT_VARIABLE, str(tmp_path / "from-environment"))
    assert build_parser().parse_args(promo_command).promo_root == tmp_path / "from-environment"

    flagged = build_parser().parse_args([*promo_command, "--promo-root", str(tmp_path / "from-flag")])
    assert flagged.promo_root == tmp_path / "from-flag"


def test_the_gallery_follows_the_promo_root(clean_environment: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert build_parser().parse_args(["gallery"]).out == DEFAULT_PROMO_OUTPUT_ROOT / "gallery"

    clean_environment.setenv(PROMO_OUTPUT_ROOT_VARIABLE, str(tmp_path))
    assert build_parser().parse_args(["gallery"]).out == tmp_path / "gallery"

    flagged = build_parser().parse_args(["gallery", "--out", str(tmp_path / "elsewhere")])
    assert flagged.out == tmp_path / "elsewhere"


@pytest.mark.parametrize("command", ["manifest", "upload", "verify", "qr-check"])
def test_a_publishing_command_resolves_the_root_of_the_tree_it_works_on(
    command: str,
    clean_environment: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    assert _publish_tree(build_parser().parse_args([command])) == (PublishTree.SWEEPS, DEFAULT_OUTPUT_ROOT)
    assert _publish_tree(build_parser().parse_args([command, "--tree", "promo"])) == (
        PublishTree.PROMO,
        DEFAULT_PROMO_OUTPUT_ROOT,
    )

    clean_environment.setenv(OUTPUT_ROOT_VARIABLE, str(tmp_path / "library"))
    clean_environment.setenv(PROMO_OUTPUT_ROOT_VARIABLE, str(tmp_path / "promo"))
    assert _publish_tree(build_parser().parse_args([command]))[1] == tmp_path / "library"
    assert _publish_tree(build_parser().parse_args([command, "--tree", "promo"]))[1] == tmp_path / "promo"

    flagged = build_parser().parse_args([command, "--tree", "promo", "--output-root", str(tmp_path / "flag")])
    assert _publish_tree(flagged)[1] == tmp_path / "flag"
