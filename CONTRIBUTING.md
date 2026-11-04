# Contributing to haidra-param-sweeps

In brief:

- Install [uv](https://docs.astral.sh/uv/) and the zbar system library, then run `uv sync`.
- Before opening a pull request, `ruff check`, `ruff format --check`, `mypy` and `pytest` must pass. CI runs the same
  commands.
- Code follows the [Haidra Python style guide](https://github.com/Haidra-Org/haidra-assets/blob/main/docs/meta/python.md).
  Pull request descriptions follow the
  [Haidra pull request standard](https://github.com/Haidra-Org/haidra-assets/blob/main/docs/meta/pull_requests.md).

## Development environment

### System requirements

- Python 3.12 or newer. uv installs a managed interpreter when one is missing. `.python-version` selects 3.13.
- The zbar shared library. `pyzbar` is a ctypes wrapper and loads it at runtime to decode QR codes, and the tests in
  `tests/test_qr_decode.py` fail without it.
    - Debian and Ubuntu: `sudo apt-get install libzbar0`
    - Fedora: `sudo dnf install zbar-libs`
    - macOS: `brew install zbar`
    - Windows: the `pyzbar` wheel includes the zbar DLLs. If the import fails with a missing DLL error, install the
      [Visual C++ 2013 redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).

### Installing

```sh
uv sync
```

This installs the tool, the `publish` extra and the `dev` dependency group into `.venv/`. The `publish` extra is part
of the `dev` group so the publishing tests in `tests/test_publish.py` run instead of skipping.

The `local` extra and the torch extras (`cu126`, `cu130`, `cu132`, `cpu`) install a GPU stack and are not needed for
development or for the test suite. See the README for the local runner.

To build the documentation site as well:

```sh
uv sync --group docs
uv run zensical build --strict
```

The site is written to `site/`.

## Checks

Run these from the repository root. CI runs the same commands on Python 3.12 and 3.13.

```sh
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

`ruff format` is the formatter. The ruff rule selection and the strict mypy configuration are in `pyproject.toml`.

### pre-commit

```sh
uv run pre-commit install
uv run pre-commit run --all-files
```

The hooks run ruff, mypy and file hygiene checks (trailing whitespace, final newlines, LF line endings, valid JSON,
TOML and YAML).

## Dependencies

Dependencies are locked in `uv.lock`. After changing `pyproject.toml`, run `uv lock` and commit both files. CI installs
with `uv sync --locked` and fails when the lock file is out of date.

## Licensing

The project is licensed under the GNU Affero General Public License v3.0 or later. Source files start with SPDX
headers:

```python
# SPDX-FileCopyrightText: 2026 Tazlin
#
# SPDX-License-Identifier: AGPL-3.0-or-later
```

By contributing, you agree that your contribution is licensed under the same terms.
