---
title: "Render the library"
summary: "Plan and render the sweep library, regenerate a single strip, resume an interrupted run, and write crops, bounded copies, control maps and the step-preview strips."
order: 20
---

# Render the library

This guide renders the sweep library, or part of it, with either runner. Before you start, set up the runner you will
use as described in [Set up the runners](set-up-runners.md). When you finish, the images, their sidecars and
`index.json` are under the output root, and `index.json` lists every strip you rendered.

The output root is `output/param-sweeps/` in the repository unless `--output-root` or `PARAM_SWEEPS_OUTPUT_ROOT` is
set. The flag takes precedence over the variable.

## Plan the run

```sh
uv run param-sweeps plan
```

This prints the fetch plan (the models, their approximate download size, whether each is already on disk, and image
counts) and the total image count, and does not generate anything. `plan` takes the same filters as the runners, so you can check
the size of a narrowed run first.

## Render everything

```sh
# On the local GPU
uv run param-sweeps local --gpu-backend cuda

# Through the public horde, one job at a time
uv run param-sweeps horde
```

Both runners print the fetch plan and the total image count before generating anything.

**Check:** `index.json` lists each strip. A strip that is not finished has `"complete": false`.

## Regenerate one strip

The filters combine, and a single strip is all three of them:

```sh
# One strip: this parameter, this model, this scene
uv run param-sweeps horde --parameter clip_skip --model nova-anime-xl --scene anime

# Everything for one parameter, across every model and scene it runs on
uv run param-sweeps local --parameter steps

# Render a strip that is already complete again
uv run param-sweeps local --parameter tiling --model icbinp --scene thing --force
```

Without `--force`, images already on disk are left alone. With it, the selected images are rendered again and
overwritten in place. File names are deterministic, so this does not leave orphaned files behind.

If you changed the stops of a strip that is already published, read
[Renumbered stops leave orphans](../reference/publishing.md#sharp-edges) before you publish.

## Resume an interrupted run

Run the same command again. Every image is recorded in `.param-sweeps-checkpoint.json` at the root of the output tree
as soon as it is written, so a new run renders only what is missing. A job that fails is reported at the end and left
pending. The rest of the campaign still runs, and the next run retries the failed job.

An entry only counts while its image is still on disk. **To have an image rendered again, delete it.** There is no
bookkeeping to edit.

`index.json` is rewritten as the run progresses, and it only ever describes images that exist.

**Check:** the run ends with a line such as `Selected 12, rendered 3, already present 9, failed 0.`

## Write crops and bounded copies

After rendering, cut the declared crops and write the bounded copies for the images already on disk:

```sh
uv run param-sweeps derivatives
```

The command also accepts its former name, `crops`. It skips files that already exist unless `--force` is given, and
`--parameter` narrows it. Then rebuild the index so it reports the new files:

```sh
uv run param-sweeps index
```

**Check:** stops with a declared region have a `crop` entry in `index.json`, and stops over 1024 pixels on their long
edge have a `fit` entry.

## Draw the control maps

ControlNet strips publish the map that guided each render. The maps are drawn by a horde-image-utilities service,
listening on `http://127.0.0.1:7860` unless `--hiu-url` or `PARAM_SWEEPS_HIU_URL` is set:

```sh
uv run param-sweeps control-maps --parameter controlnet_gallery
uv run param-sweeps index
```

Each map is written beside the render it guided, under the same file name stem. A detector that the service will not
run is recorded as a gap, with the service's message. `--force` redraws maps already on disk. The command ends with
a count of maps drawn, reused and undrawn, and lists the reason for each undrawn map.

**Check:** the triptych strips' stops have a `control_map` entry in `index.json`.

## Render the step-preview strips

The two step-preview strips come from one render, captured step by step on the local GPU. The `local` and `horde`
runners never select them.

```sh
AIWORKER_CACHE_HOME=<path-to-worker-cache> uv run param-sweeps step-preview
```

This renders the specification's `step_preview` stop once (one 30-step render), and writes both strips, 30 frames each,
and the index. [One parameter per strip](../explanation/one-parameter-per-strip.md#one-render-step-by-step) explains the
two strips.

**Check:** `index.json` has complete `step_preview` and `step_preview_guess` strips.

## Rebuild the index

```sh
uv run param-sweeps index
```

This rebuilds `index.json` from whatever images are on disk. It does not generate anything.
