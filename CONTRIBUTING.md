# Contributing to haidra-param-sweeps

In brief:

- Install [uv](https://docs.astral.sh/uv/) and the zbar system library, then run `uv sync`.
- Before opening a pull request, run `ruff check`, `ruff format --check`, `mypy` and `pytest`. All four must pass, and
  CI runs the same commands.
- Code follows the [Haidra Python style guide](https://github.com/Haidra-Org/haidra-assets/blob/main/docs/meta/python.md).
  Pull request descriptions follow the
  [Haidra pull request standard](https://github.com/Haidra-Org/haidra-assets/blob/main/docs/meta/pull_requests.md).

## Development environment

### System requirements

- Python 3.12 or newer. uv installs a managed interpreter when one is missing. `.python-version` selects 3.13.
- The zbar shared library. `pyzbar` is a ctypes wrapper that loads zbar at runtime to decode QR codes. The tests in
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
of the `dev` group, so the publishing tests in `tests/test_publish.py` run instead of skipping.

The `local` extra and the torch extras (`cu126`, `cu130`, `cu132`, `cpu`) install a GPU stack. Development and the test
suite do not need them. To set up the local runner, see
[Set up the runners](docs/how-to/set-up-runners.md).

To build the documentation site as well:

```sh
uv sync --group docs
uv run zensical build --clean --strict
```

The site is written to `site/`, which git ignores.

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

The hooks run ruff, mypy and file hygiene checks: trailing whitespace, final newlines, LF line endings, and valid JSON,
TOML and YAML.

## What the tests cover

The tests do not generate images and do not use the network.

- The specification's shape and the rules the library must follow. For example, clip skip runs on three models, and
  tiling is taught on a texture and shown failing on an object.
- Resuming, the index, and the sync planning behind `upload` and `verify`. The bucket is a `botocore` stub, so the tests
  check the plan, the upload order and the headers. They do not test the behaviour of any particular storage provider.
- `tests/test_alchemy.py` runs the whole alchemy campaign through a scripted backend. It checks the manifest's shape,
  the coverage of the form and control-map vocabularies, the text results and the strip, without a GPU.
- `tests/test_image_utilities.py` does the same for the control-map service. It serves the backend's requests from a
  scripted transport mounted on a real session, so the handling of a detector that the service refuses to run is
  checked without a running service.
- `tests/test_gallery.py` runs a whole gallery campaign against a scripted transport. It checks the limits applied to a
  picture's settings, the manifest rebuilt from disk with its curation kept, a refusal recorded and skipped, and a
  conversation that includes its earlier turns. The tests do not reach the horde or spend kudos.
- `tests/test_publish.py` skips itself, with a reason, in an environment without the publish extra, because there is
  no client to stub. It also includes the test that keeps the extra optional: it imports the tool in a subprocess and
  fails if `boto3` or `botocore` is in `sys.modules`.

## Documentation

The documentation in `docs/` is organised by [Diátaxis](https://diataxis.fr/) class: `explanation/`, `reference/` and
`how-to/`. Each document starts with YAML front matter (`title`, `summary` and an optional `order`), and each class has
a `README.md` that lists its documents. When you add or rename a document, update its class `README.md` and the `nav`
in `zensical.toml`. A change in behaviour updates the documentation in the same pull request.

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
