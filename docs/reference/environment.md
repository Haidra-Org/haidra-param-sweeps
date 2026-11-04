---
title: "Environment variables"
summary: "Every environment variable param-sweeps reads or sets, where each is read from, and which flag overrides it."
order: 30
---

# Environment variables

`param-sweeps` reads its configuration from environment variables, and some of them also from a `.env` file. A flag on
the command line always takes precedence over the matching variable. Secrets (the API key and the bucket secret) are
never printed, logged or written into any output.

## Code map

| Concept | File | Symbol |
| --- | --- | --- |
| Output roots | `param_sweeps/cli.py` | `OUTPUT_ROOT_VARIABLE`, `PROMO_OUTPUT_ROOT_VARIABLE`, `resolve_output_root` |
| API key and `.env` lookup | `param_sweeps/horde_client.py` | `read_api_key`, `default_env_file`, `read_env_value` |
| Bucket settings | `param_sweeps/publish.py` | `PublishEnvironment`, `load_s3_settings` |
| GPU backend | `param_sweeps/local_runner.py` | `resolve_gpu_backend` |
| horde-image-utilities URL | `param_sweeps/image_utilities.py` | `resolve_service_url` |
| Beta model reference | `param_sweeps/alchemy.py` | `BETA_MODEL_CATEGORIES` |

`tests/test_cli.py` covers the output-root order.

## The .env file

Variables marked ".env" below are read from the process environment first and from a `.env` file second. The file is
the one named by `--env-file`, else the one named by `PARAM_SWEEPS_ENV_FILE`, else the `.env` at the repository root if
it exists. The tool does not look for a `.env` anywhere else. The repository `.env` is ignored by git.

Lines in the file have the form `NAME=value`, optionally prefixed with `export`. Lines starting with `#` are ignored,
and surrounding quotes are removed from the value.

## Variables

| Variable | Read from | Default | Meaning |
| --- | --- | --- | --- |
| `PARAM_SWEEPS_OUTPUT_ROOT` | environment only | `output/param-sweeps` in the repository | The library's root, for every subcommand with `--output-root`. The tutorial flows write to `tutorials/` under it. |
| `PARAM_SWEEPS_PROMO_OUTPUT_ROOT` | environment only | `output/promo` in the repository | The promo tree's root: the share cards (`--promo-root`), the gallery (`gallery/` under it, `--out`) and `--tree promo`. |
| `PARAM_SWEEPS_ENV_FILE` | environment only | the repository `.env` | The `.env` to read when `--env-file` is not given. |
| `AIHORDE_DEV_KEY` | environment, then .env | none | The AI Horde API key for `horde`, `flows`, `alchemy --backend api` and `gallery`. It is only put in a request header. |
| `PARAM_SWEEPS_GPU_BACKEND` | environment only | `cuda` | The compute backend when `--gpu-backend` is not given. |
| `PARAM_SWEEPS_HIU_URL` | environment only | `http://127.0.0.1:7860` | The horde-image-utilities service when `--hiu-url` is not given. |
| `AIWORKER_CACHE_HOME` | environment only | horde-engine's default | The worker model cache that the local runner reads weights from. Read by horde-engine. This tool does not read it. |

An empty or whitespace-only value counts as unset.

## Bucket variables

The publishing commands read these from the environment, then the .env. The secret is only passed to the S3 client.

| Variable | Required | Meaning |
| --- | --- | --- |
| `PARAM_SWEEPS_S3_ENDPOINT_URL` | yes | The S3 endpoint. For R2, `https://<account-id>.r2.cloudflarestorage.com`. |
| `PARAM_SWEEPS_S3_BUCKET` | yes | The bucket the library is published into. |
| `PARAM_SWEEPS_S3_ACCESS_KEY_ID` | yes | The access key ID of an API token with object read and write permission. |
| `PARAM_SWEEPS_S3_SECRET_ACCESS_KEY` | yes | That token's secret. |
| `PARAM_SWEEPS_S3_PREFIX` | no | A key prefix, so one bucket can hold more than one thing. Defaults to the bucket root. Overridden by `--prefix`, and by the `promo` default under `--tree promo`. |
| `PARAM_SWEEPS_S3_REGION` | no | Defaults to `auto`, which is what R2 requires. An AWS bucket needs its real region. |

A command that needs the bucket and is missing any required variable exits 1 and lists the missing names.

## Beta model reference variables

`alchemy --beta` sets three variables that horde-engine reads before it builds its model managers. A variable that is
already set in the environment is left alone, so a run with its own key or its own service keeps them.

| Variable | Value set |
| --- | --- |
| `HORDELIB_BETA_MODEL_CATEGORIES` | the opted-in categories, currently `gfpgan` |
| `HORDELIB_BETA_MODELS_API_KEY` | the anonymous key |
| `HORDE_MODEL_REFERENCE_PRIMARY_API_URL` | `https://models.aihorde.net/api` |

The categories are a tuple constant, `BETA_MODEL_CATEGORIES`, so `esrgan` can be added when the beta upscalers are worth
publishing.
