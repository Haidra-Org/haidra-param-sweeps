---
title: "Set up the runners"
summary: "Install param-sweeps, give the horde runner an API key, and prepare a local GPU environment for the local runner."
order: 10
---

# Set up the runners

This guide installs the tool and prepares each runner. The `horde` runner needs only the base install and an AI Horde
API key. The `local` runner, and every other subcommand that renders on your own GPU, also needs horde-engine, a torch
build that matches your GPU, and model weights.

Before you start, you need Python 3.12 or newer and [uv](https://docs.astral.sh/uv/). When you finish,
`uv run param-sweeps plan` prints a fetch plan for the runner you set up.

## Install the tool

```sh
git clone https://github.com/Haidra-Org/haidra-param-sweeps.git
cd haidra-param-sweeps
uv sync
```

This installs the tool, its tests and linters, and the `publish` extra into `.venv/`. It does not install a GPU stack.

QR images are decoded with `pyzbar`, which needs the zbar shared library. Install it as described in
[CONTRIBUTING.md](https://github.com/Haidra-Org/haidra-param-sweeps/blob/main/CONTRIBUTING.md#system-requirements).

**Check:** `uv run param-sweeps --help` lists the subcommands.

## Set up the horde runner

The runner reads `AIHORDE_DEV_KEY` from the environment first. If it is not set there, the runner reads it from a
`.env` file: the one given with `--env-file`, else the one named by `PARAM_SWEEPS_ENV_FILE`, else the `.env` at the
repository root if it exists. The key is only put in a request header. It is never printed, logged or written into
any output.

1. Create `.env` at the repository root (git ignores it):

   ```sh
   AIHORDE_DEV_KEY=<your-api-key>
   ```

2. **Check:** `uv run param-sweeps plan` prints the image count. A render with
   `uv run param-sweeps horde --parameter clip_skip --model nova-anime-xl --scene anime` logs which source the key came from. A
   missing key stops the run with a message naming where it looked.

The same key is used by `flows`, `alchemy --backend api` and `gallery`.

## Set up the local runner

The local runner needs horde-engine and torch. Torch alone is a download of several gigabytes.

### Reuse an existing worker environment

If the machine already runs an AI Horde worker or has a horde-engine (`hordelib`) development checkout, reuse that
environment instead of building a new GPU stack. The exact torch version does not matter for this workload.

1. Install the tool into that environment without touching its dependencies:

   ```sh
   uv pip install --python <existing-env-python> -e . --no-deps
   ```

   `<existing-env-python>` is the environment's interpreter, for example `<existing-env>/bin/python` on Linux and
   macOS or `<existing-env>/Scripts/python.exe` on Windows.

2. Any environment that runs horde-engine already has pydantic, strenum, requests, pillow and horde_sdk. Check that
   its `horde_sdk` is 0.29.3 or newer, and install `pyzbar` if it is missing:

   ```sh
   uv pip install --python <existing-env-python> pyzbar
   ```

3. Run every command through that interpreter:

   ```sh
   <existing-env-python> -m param_sweeps plan
   ```

**Check:** the fetch plan lists each model and whether it is already on disk. A worker install usually has the weights
already, in its `AIWORKER_CACHE_HOME`.

To undo this, run `uv pip uninstall --python <existing-env-python> haidra-param-sweeps`.

### Fresh install

Install horde-engine (the `local` extra) together with one torch extra. The torch extras select the matching PyTorch
index for both torch and torchvision, and only one can be used at a time:

| Extra | Torch build |
| --- | --- |
| `cu126` | CUDA 12.6 |
| `cu130` | CUDA 13.0 |
| `cu132` | CUDA 13.2 |
| `cpu` | CPU only |

```sh
uv sync --extra local --extra cu130
```

Without a torch extra, `--extra local` installs whichever torch build PyPI serves for your platform, which is often not
the one your card can use.

There is no extra for ROCm or the other backends. For an AMD card, install a ROCm build of torch into the environment
first (see the [PyTorch install selector](https://pytorch.org/get-started/locally/)), then install horde-engine with
`uv pip install -e ".[local]"`, which keeps the torch already installed. A later `uv sync` replaces it with the locked
build.

**Check:** `uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` prints the
expected build.

### Choose the compute backend

The backend is never guessed. It comes from `--gpu-backend`, then `PARAM_SWEEPS_GPU_BACKEND`, then `cuda`. Supported
values are `cuda`, `rocm`, `xpu`, `npu`, `mlu`, `mps`, `directml` and `cpu`. For an AMD card, run with
`--gpu-backend rocm`. Pin a specific accelerator with `--device-index`.

### Point at the model weights

Model weights come from the standard worker cache. Set `AIWORKER_CACHE_HOME` to an existing worker's cache so models
are not downloaded twice:

```sh
export AIWORKER_CACHE_HOME=<path-to-worker-cache>
```

Pass `--no-download` to fail instead of downloading a model that is not already in the cache.

**Check:** `uv run param-sweeps plan` shows each model as already on disk.

## Next steps

- [Render the library](render-library.md) with either runner.
- [Run the alchemy campaign](run-alchemy.md), which needs `rembg` for background removal.
