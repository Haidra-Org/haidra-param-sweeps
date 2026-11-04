---
title: "Alchemy evidence"
summary: "Why every alchemy form in a use case shares one source, why a full run uses three backends, and why forms that could not run stay in the manifest."
order: 30
---

# Alchemy evidence

Alchemy is the part of the horde that takes an image as input instead of a prompt: upscalers, face restorers,
background removal, and the describe family, which returns words instead of pixels. The `alchemy` subcommand applies
every form in the horde's shared vocabulary to a source image, and writes an evidence tree with its own manifest.
Three ideas shape it: forms that are compared share one source, each form runs where the horde runs it, and a form
that could not run is reported instead of omitted.

The procedure is in [Run the alchemy campaign](../how-to/run-alchemy.md). The manifest and the specification are in
the [output tree reference](../reference/output-tree.md#alchemy-tree) and the
[campaign specifications reference](../reference/campaign-specifications.md#alchemy-specification).

## The form list comes from the SDK

The tool does not keep its own list of forms. It reads `KNOWN_ALCHEMY_TYPES` from `horde_sdk`, and `alchemy_spec.json`
places each of those forms into exactly one of five use cases: `describe`, `control_map`, `enlarge`, `fix_faces` and
`cut_out`. If the SDK gains a form that the specification has not placed, the file fails to load, so the form cannot
silently disappear from the published evidence. A few forms do not fit any of the five use cases well (the NSFW check,
the aesthetic score, the palette and the vectorizer). Each is placed in the closest use case, and its notes say so.

The `annotation` form is the exception to one entry per form. Which detector runs is the whole result, so the form is
placed once for each control type named by `horde_sdk`'s `KNOWN_ANNOTATION_CONTROL_TYPES`. Its entries have a
`control_type` field, which distinguishes them in the tree and in the manifest. A detector that the SDK gains and the
specification has not placed fails the load in the same way as a form.

## One source per use case

Two upscalers are only comparable if they started from the same picture. Every form in a use case therefore runs on the
same source. The sources are published once at the root of the alchemy tree, and every entry points at one of them:

| Use case | Source |
| --- | --- |
| `describe` | the courier at 1024 pixels, the anchor image of the txt2img tutorial example |
| `control_map` | the courier at 1024 pixels, so every detector reads the same picture |
| `enlarge` | the courier at 512 pixels, small enough that an upscaler has something to do |
| `fix_faces` | the degraded portrait from the horde-engine face-fix tests, the kind of picture blind face restoration was built for |
| `cut_out` | the robot gardener, whose silhouette is clean enough to show where the cut fell |

## Three backends

The horde runs alchemy forms in three different places, so a full campaign is three runs, each restricted to the forms
its backend can run:

- **`hordelib`** (the default) applies the post-processors on this machine's GPU, through the same stack a worker runs.
- **`api`** submits the image to the public horde's `/interrogate` endpoints and publishes the result. It is the only
  way to show what the describe family returns without inventing it. The API serves the post-processors too, so an
  unrestricted API run would spend kudos repeating work the GPU already did.
- **`hiu`** has a horde-image-utilities service draw the control maps. An alchemist worker runs that service beside
  itself for the `annotation` form. A run through it covers only the `annotation` form, because the other forms are
  produced elsewhere and an unrestricted run would overwrite what those backends recorded.

The sidecar beside each control map records the service's host and the version it reports, because a control map is
the output of a particular build of a particular detector.

## Forms that could not run

The manifest lists every placed form, whether or not it produced anything. An entry without a result reports that the
form could not be run here, and its notes say why. If the entry were left out, a reader would assume the form does not
exist. Two causes make this a routine state instead of a failure:

- **Only the post-processing forms can run through horde-engine.** Captioning, interrogation, the NSFW check and the
  other read-only forms are served by the alchemist's own backend, which is a different stack. `--backend api` requests
  them from the horde instead of approximating them with another installed model. The control maps are not on that
  backend either. They are drawn by horde-image-utilities, and the entries say so unless they were produced by a
  `--backend hiu` run.
- **The shared vocabulary names weights that the live model reference does not include.** Six of the eleven upscalers
  do not have an entry to resolve, so there are no weights to download or run. All four face restorers are
  expected to render locally. GFPGAN and CodeFormers resolve from the published model reference, and GFPGANv1.3 and
  RestoreFormer from the beta (pending) model reference.

## The beta model reference

GFPGANv1.3 and RestoreFormer are named by the shared vocabulary, but only the pending queue of the model reference
service lists them. The published reference does not. Without `--beta`, two of the four face restorers do not resolve
and are published as gaps. The variables that `--beta` sets are listed in the
[environment reference](../reference/environment.md#beta-model-reference-variables).

Beta resolution is additive. If a category is opted in but its pending fetch returns an empty result, resolution
falls back to the published reference instead of failing the run.

## Background removal

`strip_background` needs `rembg`, which the base install does not include. Without it, the form reports itself
unavailable, in the same words horde-engine uses, instead of failing the run. The horde's production background
removal is moving from this path to horde-image-utilities, which the worker runs as a separate process in its own
environment. The published cut-out is to be regenerated through horde-image-utilities once that move is complete.
