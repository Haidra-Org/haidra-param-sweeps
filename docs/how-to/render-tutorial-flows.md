---
title: "Render the tutorial examples"
summary: "Render the tutorial worked examples from flows_spec.json through the public horde or locally, rebuild their manifest, and the known recipe constraints."
order: 50
---

# Render the tutorial examples

`flows_spec.json` holds the tutorial worked examples: one committed request per technique, in each of the three worlds
where the technique can be shown. `flows` renders them into `tutorials/<flow>/` under the library's output root and
then rebuilds `tutorials/manifest.json` from what is on disk.

Before you start, set up the horde runner's API key, or the local runner for `--local`, as in
[Set up the runners](set-up-runners.md). When you finish, each rendered flow has a directory with its result and
`meta.json`, and the manifest lists it. The files are described in the
[output tree reference](../reference/output-tree.md#tutorial-tree).

## 1. Check what would be submitted

```sh
uv run param-sweeps flows --dry-run
```

This prints each flow's model and size, and logs every flow with a `needs_attention` note. It does not submit anything.

## 2. Render the flows

```sh
# Through the public horde, one flow at a time
uv run param-sweeps flows --flow img2img

# On this machine's GPU, through the worker software, with no queue or other hardware in between
uv run param-sweeps flows --local --flow inpainting
```

Every selected flow is rendered again. There is no skip for a flow that already exists, so name the flows whose
requests changed with `--flow`.

`meta.json` records which runner made the example in `runner`. A local render is not charged and does not record kudos.

**Check:** `tutorials/<flow>/meta.json` exists with the expected `runner`, and `tutorials/manifest.json` lists the flow.

## Rebuild the manifest only

```sh
uv run param-sweeps flows --manifest-only
```

This rewrites `tutorials/manifest.json` from what is on disk and does not generate anything.

## Iterate on a recipe locally

The horde is the reference runner for the published set. A recipe search through the queue is slow, and the results
depend on which worker picks up each job. `flows --local` renders through the same worker software directly, without a queue or
another worker in between, and a run of seeds takes about a minute each.

## Record what is still a compromise

A flow's `notes` say what the published recipe is and why. A `needs_attention` field says what is still a compromise,
what was tried, and what to vary next. Clear `needs_attention` when the flow is settled, and keep the lesson in `notes`.

A flow that cannot be shown properly in a world is marked `skipped` with a `skipped_reason`, and is left out of the
manifest. Consumers show the first world's example in its place and say so.

## Known recipe constraints

- **Inpainting fills from context.** An SD1.5 inpainting model fills a masked hole with what the surroundings suggest,
  so a masked sky comes back as sky at the anchor guidance, and the prompt is ignored. Raising guidance to 11 or
  weighting the subject as `(a phrase:1.4)` both make the object appear. The flows use the weight, so the guidance
  shared by every other example is unchanged.
- **Keep the mask clear of the figure.** The mask edge is grown by a few pixels before encoding, and a mask that clips
  hair puts a hat on the subject.
- **The anime inpainting checkpoint repaints the whole frame.** The checkpoint the horde serves returns a washed-out
  repaint of the whole frame instead of the hole alone, so the fox-shrine inpainting flow is skipped and has a
  `needs_attention` note.
- **The QR ControlNet fails when the prompt names geometry.** See
  [QR codes that scan](../explanation/qr-codes.md#the-prompt-must-not-describe-the-code) for the prompt rule and for
  how the QR seeds and models were chosen.
- **Not every lesson works in every world.** A world is a subject and a model. The robot does not have a face to fix, so
  face fix has no third world. The anime inpainting checkpoint fails as described above.
