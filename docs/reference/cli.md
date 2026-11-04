---
title: "Command line"
summary: "Every param-sweeps subcommand, its flags, defaults and exit behaviour."
order: 20
---

# Command line

`param-sweeps` is the console entry point (`python -m param_sweeps` is equivalent). Every subcommand takes `--verbose`
to log at debug level. A subcommand exits 0 when everything it was asked to do succeeded and 1 otherwise.

Defaults that name a file refer to the repository root. Output roots resolve in this order: the flag, then
`PARAM_SWEEPS_OUTPUT_ROOT` or `PARAM_SWEEPS_PROMO_OUTPUT_ROOT`, then `output/param-sweeps` or `output/promo` in the
repository. See the [environment reference](environment.md).

## Code map

| Concept | File | Symbol |
| --- | --- | --- |
| Parser and every flag | `param_sweeps/cli.py` | `build_parser` |
| Entry point | `param_sweeps/cli.py` | `main` |
| Output root resolution | `param_sweeps/cli.py` | `resolve_output_root`, `resolve_promo_output_root` |
| Job selection by filter | `param_sweeps/campaign.py` | `JobFilter`, `select_jobs` |
| GPU backend resolution | `param_sweeps/local_runner.py` | `resolve_gpu_backend` |

The flag parsing and output-root order are covered by `tests/test_cli.py`.

## Subcommands at a glance

| Subcommand | What it does |
| --- | --- |
| [`plan`](#plan-local-and-horde) | Print the fetch plan and image count, without generating anything. |
| [`local`](#plan-local-and-horde) | Render through horde-engine on this machine's GPU. |
| [`horde`](#plan-local-and-horde) | Render through the public AI Horde API. |
| [`index`](#index) | Rebuild `index.json` from the images on disk. |
| [`derivatives`](#derivatives) | Cut crops and write bounded copies of published frames. Also accepts the name `crops`. |
| [`control-maps`](#control-maps) | Draw the control map beside every ControlNet render. |
| [`step-preview`](#step-preview) | Capture one render step by step and publish the two step-preview strips. |
| [`sampler-cost`](#sampler-cost) | Measure the cost of one step of each sampler. |
| [`alchemy`](#alchemy) | Apply every alchemy form and write the alchemy manifest and the upscaler strip. |
| [`flows`](#flows) | Render the tutorial worked examples and write their manifest. |
| [`review`](#review-and-choose) | Write the review contact sheet for a batch specification. |
| [`choose`](#review-and-choose) | Write picks saved from the contact sheet back into the specification. |
| [`promo`](#promo) | Draw share cards and posters from the renders on disk. |
| [`gallery`](#gallery) | Make the promo gallery through the live horde. |
| [`qr-check`](#qr-check) | Decode every published QR image and compare it with its payload. |
| [`manifest`](#manifest-upload-and-verify) | List every file a tree publishes, with sizes. |
| [`upload`](#manifest-upload-and-verify) | Sync a tree to S3-compatible storage. |
| [`verify`](#manifest-upload-and-verify) | Check a published tree, or the files on disk, against its entrypoints. |

## Shared flag groups

### Rendering filters

`plan`, `local` and `horde` take these flags. The filters combine, so one parameter, one model and one scene select a
single strip.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--spec PATH` | `sweep_spec.json` | The sweep specification. |
| `--output-root PATH` | library root | The output tree's root. |
| `--parameter NAME` | all | Restrict to this parameter. Repeatable. Must be a `SweepParameter` value. |
| `--model SLUG` | all | Restrict to this model slug. Repeatable. |
| `--scene ID` | all | Restrict to this scene id. Repeatable. |
| `--force` | off | Render selected images even when the checkpoint already accounts for them, overwriting them. |

### Local GPU flags

`plan`, `local`, `step-preview` and `sampler-cost` take all three. `flows --local` takes all three, and `alchemy` takes
the first two.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--gpu-backend NAME` | `PARAM_SWEEPS_GPU_BACKEND`, else `cuda` | One of `cuda`, `rocm`, `xpu`, `npu`, `mlu`, `mps`, `directml`, `cpu`. Never guessed. |
| `--device-index N` | `0` | The accelerator index to pin the process to. |
| `--no-download` | off | Fail instead of downloading a model that is not already in the worker cache. |

### Horde API flags

`horde`, `flows`, `alchemy` and `gallery` take `--env-file`. All of them except `gallery` also take `--api-url`.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--env-file PATH` | `PARAM_SWEEPS_ENV_FILE`, else the repository `.env` | The `.env` holding `AIHORDE_DEV_KEY`. `AIHORDE_DEV_KEY` in the environment takes precedence over the file. |
| `--api-url URL` | `https://aihorde.net/api/v2` | The API root to use instead of the public horde. |

### Published tree flags

`qr-check`, `manifest`, `upload` and `verify` take these.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--tree sweeps\|promo` | `sweeps` | The evidence library, or the promo tree published beside it. |
| `--output-root PATH` | the library root, or the promo root with `--tree promo` | The tree's root. |

`manifest`, `upload` and `verify` also take:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--env-file PATH` | `PARAM_SWEEPS_ENV_FILE`, else the repository `.env` | The `.env` with the `PARAM_SWEEPS_S3_*` variables. |
| `--prefix PREFIX` | `PARAM_SWEEPS_S3_PREFIX`, or `promo` with `--tree promo` | The key prefix to publish under. |

## plan, local and horde

`plan` prints the fetch plan (models, approximate download size, whether each is already on disk, image counts) and the
total image count, and does not generate anything. `local` and `horde` print the same plan before they render anything.

- `local`: rendering filters and local GPU flags.
- `horde`: rendering filters and horde API flags. The API key is required.

Both runners checkpoint every image as it is written, and skip images already accounted for unless `--force` is given.
See [Render the library](../how-to/render-library.md).

## index

Rebuild `index.json` from the images already on disk. Flags: `--spec`, `--output-root`.

## derivatives

Cut the full-resolution crops and write the bounded copies of the published renders, which `index` then reports. The
command also accepts the name `crops`, its former name.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--spec PATH` | `sweep_spec.json` | The sweep specification. |
| `--alchemy-spec PATH` | `alchemy_spec.json` | The alchemy specification, whose entries can declare their own crop region. |
| `--output-root PATH` | library root | Every file is written beside the frame it came from. |
| `--parameter NAME` | all | Restrict to this parameter. Repeatable. |
| `--force` | off | Rewrite files that are already on disk. |

## control-maps

Draw the control map beside every render that a ControlNet guided, using a horde-image-utilities service.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--spec PATH` | `sweep_spec.json` | The sweep specification. |
| `--output-root PATH` | library root | Each map is written beside the render it guided, with the same file name stem. |
| `--parameter NAME` | all | Restrict to this parameter. Repeatable. |
| `--hiu-url URL` | `PARAM_SWEEPS_HIU_URL`, else `http://127.0.0.1:7860` | The horde-image-utilities service. |
| `--force` | off | Redraw maps that are already on disk. |

A detector that the service will not run is recorded as a gap with the service's message, and the render is left alone.

## step-preview

Render the specification's `step_preview` stop once on the local GPU, and publish the `step_preview` and
`step_preview_guess` strips and the index. Flags: `--spec`, `--output-root`, and the local GPU flags. Needs the `local`
extra and `AIWORKER_CACHE_HOME`.

## sampler-cost

Measure the cost of one step of each sampler relative to `k_euler`. See
[Sampler cost measurement](../explanation/sampler-cost-measurement.md).

| Flag | Default | Meaning |
| --- | --- | --- |
| `--spec PATH` | `sweep_spec.json` | The specification that supplies models, sizes, seed and cfg scale. |
| `--output-root PATH` | library root | The report is written to `measurements/` under it. |
| `--model SLUG` | `deliberate`, `albedobase-xl` | Restrict to this model. Repeatable. |
| `--sampler NAME` | the whole SDK vocabulary | Restrict to this sampler. Repeatable. |
| `--steps N` | `10`, `20`, `30`, `40` | A step count to fit through. Repeatable. |
| `--repeats N` | `3` | How many renders each point is the median of. |
| `--warmups N` | `1` | How many discarded renders each model gets before timing starts. |
| `--plan-only` | off | Print the samplers, schedules and render count, without measuring anything. |

Also takes the local GPU flags.

## alchemy

Apply alchemy forms to their sources and write the alchemy manifest, the evidence tree and the `upscaler_form` strip.
See [Run the alchemy campaign](../how-to/run-alchemy.md).

| Flag | Default | Meaning |
| --- | --- | --- |
| `--spec PATH` | `alchemy_spec.json` | The alchemy specification. |
| `--sweep-spec PATH` | `sweep_spec.json` | The sweep specification with the `upscaler_form` strip. |
| `--output-root PATH` | library root | The evidence is written to `alchemy/` under it. |
| `--form FORM` | every placed form | Restrict to this form. Repeatable. Must be a form named by `horde_sdk`. |
| `--control-type TYPE` | every type the SDK names | Restrict the `annotation` form to this control type. Repeatable. |
| `--force` | off | Apply selected forms even when their result is already on disk. |
| `--plan-only` | off | Print the forms and their use cases, without applying anything. |
| `--backend hordelib\|api\|hiu` | `hordelib` | Apply forms locally through horde-engine, through the live horde, or through a horde-image-utilities service. |
| `--hiu-url URL` | `PARAM_SWEEPS_HIU_URL`, else `http://127.0.0.1:7860` | The horde-image-utilities service, for `--backend hiu`. |
| `--beta` | off | Also resolve the `gfpgan` weights from the pending model reference. |
| `--gpu-backend`, `--device-index` | | As in the local GPU flags. There is no `--no-download`, because the backend fetches post-processing models while it resolves the form. |
| `--env-file`, `--api-url` | | As in the horde API flags, for `--backend api`. |

## flows

Render the tutorial worked examples from `flows_spec.json` and rebuild `tutorials/manifest.json`. See
[Render the tutorial examples](../how-to/render-tutorial-flows.md).

| Flag | Default | Meaning |
| --- | --- | --- |
| `--spec PATH` | `flows_spec.json` | The flow specification. |
| `--output-root PATH` | `tutorials/` under the library root | The tutorial tree to write into. |
| `--flow ID` | every flow | Restrict to this flow id. Repeatable. |
| `--manifest-only` | off | Rewrite `tutorials/manifest.json` from what is on disk, without generating anything. |
| `--local` | off | Render through the worker software on this machine's GPU instead of the public horde. |
| `--gpu-backend`, `--device-index`, `--no-download` | | As in the local GPU flags. Only used with `--local`. |
| `--env-file`, `--api-url` | | As in the horde API flags. |
| `--dry-run` | off | Print what would be submitted, without generating anything. Also logs every flow with a `needs_attention` note. |

## review and choose

`review` writes `review.html` at the root of the output tree, a contact sheet of what a batch specification produced.
`choose CHOICES` validates the `choices.json` saved from the sheet, writes the picks into the specification's `chosen`
fields, and rewrites the sheet. See [Review a batch](../how-to/review-batch.md).

| Flag | Default | Meaning |
| --- | --- | --- |
| `--spec PATH` | `sweep_spec.json` | The specification to read (`review`) or update (`choose`). |
| `--output-root PATH` | library root | The output tree's root. |
| `--title TEXT` | `Review contact sheet` | The sheet's heading. |

## promo

Draw the share cards and posters that a promo specification requests, from the renders already on disk. See
[Make share cards and the promo gallery](../how-to/make-promo-material.md).

| Flag | Default | Meaning |
| --- | --- | --- |
| `--promo-spec PATH` | required | The promo specification. Font and logo paths are relative to it. |
| `--output-root PATH` | library root | The library the frames are read from. |
| `--promo-root PATH` | promo root | Where the cards and `cards.json` are written. |
| `--job KEY` | every job and set | Draw only this job or set key, such as `sweep:steps`. Repeatable. |
| `--force` | off | Redraw cards that are already on disk. |

## gallery

Make the promo gallery through the live horde: pictures, alchemy pairs, single-turn texts and conversations.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--spec PATH` | `gallery_spec.json` | The gallery specification. |
| `--out PATH` | `gallery/` under the promo root | The gallery root, where `gallery.json` is written. |
| `--env-file PATH` | `PARAM_SWEEPS_ENV_FILE`, else the repository `.env` | The `.env` holding `AIHORDE_DEV_KEY`. |
| `--only IDS` | every entry | Restrict to these entry ids, comma-separated. |
| `--skip-alchemy` | off | Do not run the alchemy forms that picture entries request. |
| `--skip-text` | off | Do not run the single-turn texts. |
| `--skip-conversations` | off | Do not run the conversations. |
| `--retry-censored` | off | Submit entries that the safety checker refused before again. |
| `--dry-run` | off | Print what would be submitted, with each picture's model and settings, and stop. |

The API root comes from the gallery specification's `api` field.

## qr-check

Decode every QR image referenced by the index and the tutorial manifest with zbar, and exit non-zero if any of them
decodes to anything other than its payload. Takes the published tree flags. See
[QR codes that scan](../explanation/qr-codes.md).

## manifest, upload and verify

These work from the rendered tree's entrypoints and do not read a specification. All take the published tree flags,
including `--env-file` and `--prefix`. `upload` and `verify` (without `--local`) need the `publish` extra. See
[Publish the library](../how-to/publish.md) and the [publishing reference](publishing.md).

| Subcommand | Flag | Meaning |
| --- | --- | --- |
| `manifest` | `--keys` | Print bucket keys under the configured prefix instead of index-relative paths. |
| `upload` | `--dry-run` | Print the plan without changing the bucket. |
| `upload` | `--delete` | Also remove objects under the prefix that neither the index nor a manifest references. |
| `verify` | `--local` | Check the files on disk instead of the bucket. Does not need credentials. |

`verify` exits non-zero when anything an entrypoint references is missing from the bucket or has a different hash.
Orphans are reported but do not fail it.
