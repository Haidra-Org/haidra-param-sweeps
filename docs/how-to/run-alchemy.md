---
title: "Run the alchemy campaign"
summary: "Render the alchemy evidence with the local, API and horde-image-utilities backends, including the beta face restorers and background removal."
order: 40
---

# Run the alchemy campaign

`alchemy` applies every form placed by `alchemy_spec.json` to its use case's source, and writes the alchemy tree,
`alchemy/manifest.json`, the `upscaler_form` strip and the index. A full campaign is three runs, one per backend, each
restricted to the forms that backend can run. [Alchemy evidence](../explanation/alchemy-evidence.md) explains why.

Before you start:

- For the `hordelib` backend: the `local` extra, a torch build for your GPU and `AIWORKER_CACHE_HOME`, as in
  [Set up the runners](set-up-runners.md).
- For the `api` backend: `AIHORDE_DEV_KEY`, read in the same way as for the horde runner.
- For the `hiu` backend: a horde-image-utilities service you can reach.

When you finish, `alchemy/manifest.json` lists every placed form, and each form that could run has a result.

## 1. Check the plan

```sh
uv run param-sweeps alchemy --plan-only
```

This prints every form and the use case it is placed in, and does not apply anything.

## 2. Install background removal

`strip_background` needs `rembg`, which the base install does not include:

```sh
uv pip install "rembg[cpu]"
```

Without it, the form reports itself unavailable and the run continues.

## 3. Run the post-processors locally

```sh
AIWORKER_CACHE_HOME=<path-to-worker-cache> uv run param-sweeps alchemy --beta
```

The default `hordelib` backend applies every form that horde-engine can run on this machine's GPU, through the same
stack a worker runs. It writes the manifest, the `upscaler_form` strip and the index.

`--beta` also resolves face restorer weights from the pending model reference. Without it, GFPGANv1.3 and RestoreFormer
do not resolve and are published as gaps. If one of the variables it sets is already in your environment, that value is left
alone. See the [environment reference](../reference/environment.md#beta-model-reference-variables).

**Check:** in `alchemy/manifest.json`, the `enlarge` entries for upscalers with weights and all four `fix_faces`
entries have a `files.result`.

## 4. Run the describe family through the horde

The describe family only runs on the horde. Restrict the run to those forms, because the API also serves the
post-processors and an unrestricted run would spend kudos repeating work the GPU already did:

```sh
uv run param-sweeps alchemy --backend api --form caption --form interrogation --form nsfw
```

`--env-file` and `--api-url` work as they do for the horde runner.

**Check:** the `describe` entries have a `files.result_text` or `files.result_json`.

## 5. Draw the control maps

```sh
uv run param-sweeps alchemy --backend hiu
```

The service is at `http://127.0.0.1:7860` unless `--hiu-url` or `PARAM_SWEEPS_HIU_URL` is set. A `hiu` run
covers only the `annotation` form, so it does not overwrite what the other backends recorded. `--control-type` narrows
it to one detector. The first call to a detector that is not loaded fetches its weights, which takes minutes for the
segmentation and depth models, so each detection is allowed ten minutes.

**Check:** the `control_map` entries have a `files.result`, and each map's sidecar records the service's host and
version.

## Run one form or detector again

Add `--force` to apply selected forms even when their result is already on disk:

```sh
# One form again
AIWORKER_CACHE_HOME=<path-to-worker-cache> uv run param-sweeps alchemy --form GFPGAN --force

# One detector again
uv run param-sweeps alchemy --backend hiu --control-type openpose --force

# The face restorers from the pending model reference
AIWORKER_CACHE_HOME=<path-to-worker-cache> uv run param-sweeps alchemy --beta --form GFPGANv1.3 --form RestoreFormer
```

## Cut the alchemy crops

Entries that declare a `crop` have the region cut from both the source and the result, and results over 1024 pixels get
a bounded copy. Both are written by `derivatives`:

```sh
uv run param-sweeps derivatives
```
