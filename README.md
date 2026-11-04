# haidra-param-sweeps

`param-sweeps` renders the AI Horde parameter-sweep illustration library. For each generation parameter it renders a
strip of images that differ in that parameter and nothing else, with a plain-language caption on every stop. Web pages
that teach generation settings read the published library, and the AI Horde website is one of them.

The tool is for people who maintain that library: adding or changing a strip, re-rendering it, reviewing the result
and publishing it to object storage.

- One checked-in specification, [`sweep_spec.json`](sweep_spec.json), describes the whole library.
- Two runners render it. `local` renders through [horde-engine](https://pypi.org/project/horde-engine/) on your own
  GPU. `horde` renders through the public AI Horde API. Both write the same files, so a strip rendered one way can
  replace the same strip rendered the other way.
- Other subcommands render the tutorial worked examples, the alchemy (post-processing) evidence, share cards and a
  promotional gallery, and publish everything to an S3-compatible bucket.

## Install

The tool needs Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/Haidra-Org/haidra-param-sweeps.git
cd haidra-param-sweeps
uv sync
```

`uv sync` installs the tool, its tests and linters, and the `publish` extra (`boto3`). It does not install a GPU
stack. That is everything the `horde` runner, the publishing commands and the test suite need.

Two further requirements apply to some uses:

- **zbar.** QR images are decoded with `pyzbar`, which loads the zbar shared library at runtime. On Debian or Ubuntu
  install `libzbar0`, on Fedora `zbar-libs`, and on macOS `brew install zbar`. The Windows wheel includes zbar.
  [CONTRIBUTING.md](CONTRIBUTING.md#system-requirements) has the details.
- **A GPU stack, for the local runner only.** The `local` extra installs horde-engine, and one of the torch extras
  `cu126`, `cu130`, `cu132` or `cpu` selects the torch build, for example `uv sync --extra local --extra cu130`. An
  existing AI Horde worker environment can be reused instead of downloading a new one. See
  [Set up the runners](docs/how-to/set-up-runners.md).

## Quick start

Print what a run would download and how many images it would render, without generating anything:

```sh
uv run param-sweeps plan
```

Render one strip through the public horde. The runner reads your AI Horde API key from `AIHORDE_DEV_KEY`, in the
environment or in a `.env` file at the repository root:

```sh
uv run param-sweeps horde --parameter clip_skip --model nova-anime-xl --scene anime
```

The images, their sidecars and `index.json` are written under `output/param-sweeps/`. Set `PARAM_SWEEPS_OUTPUT_ROOT`
or pass `--output-root` to write somewhere else. Run the same command again and it renders only what is missing.

`uv run param-sweeps --help` lists every subcommand.

## Documentation

The [documentation](docs/README.md) is organised in three parts:

- [Explanation](docs/explanation/README.md): why each strip varies one parameter, how QR codes are made to scan, how
  the alchemy evidence and the sampler cost measurement are designed, and how publishing and caching fit together.
- [Reference](docs/reference/README.md): the specification formats, every subcommand and flag, the environment
  variables, the output tree and the publishing contract.
- [How-to guides](docs/how-to/README.md): setting up each runner, rendering and regenerating strips, reviewing a
  batch, running the alchemy, tutorial, share card and gallery campaigns, measuring sampler cost, and publishing.

To build the documentation site locally:

```sh
uv run --group docs zensical build --clean --strict
```

## Contributing

[CONTRIBUTING.md](CONTRIBUTING.md) covers the development environment, the checks CI runs and the licensing.

## License

GNU Affero General Public License v3.0 or later. See [LICENSE](LICENSE).
