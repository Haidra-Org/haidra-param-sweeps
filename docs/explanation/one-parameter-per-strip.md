---
title: "One parameter per strip"
summary: "Why each strip varies one parameter, how the shared worlds keep strips comparable, and how the presentations and the masked and hires modes are designed."
order: 10
---

# One parameter per strip

A strip is a row of images that differ in one generation parameter and nothing else. A reader who scrubs along it sees
what that parameter does, because every other value stays the same. Most of the library's design exists to keep that property: the
shared subjects, the fixed seed, the per-model sizes, and the validation that refuses a strip which would compare
identical images.

The field-level contract is in the [sweep specification reference](../reference/sweep-specification.md), and the
published files are described in the [output tree reference](../reference/output-tree.md).

## Shared worlds and scenes

The scenes belong to three worlds that the whole library shares, instead of a new subject for each lesson:

- a skyharbor courier, for the photoreal scenes;
- a fox shrine, for the anime scene;
- a rooftop gardening robot, for the scenes that need a subject that is not human.

Each world has one descriptor block that names its subject. That block is repeated byte for byte in every scene the
subject appears in. A subject described again in different words is a different subject, and a strip of it would teach
the prompt instead of the parameter it sweeps. `tests/test_spec.py` asserts that the blocks are identical wherever they
appear.

Scenes are written in the prompt idiom of the models that render them: comma-separated tags for the SD1.5 models, and
booru tags with a `rating_safe` token for the anime model. A scene is one prompt string shared by every model in a
sweep, so a scene rendered by more than one model is written in the idiom of the most literal of them.

Every image uses seed `1234567890`, except the seed strip, because the seed is what that strip teaches.

## Runners and file names

The `local` runner renders through horde-engine on your own GPU, and the `horde` runner renders through the public AI
Horde API. Both write the same files with the same names, so a strip rendered one way can replace the same strip
rendered the other way.

File names are deterministic. The same specification and seed always produce the same paths, so re-rendering a strip
overwrites it in place and does not leave orphaned files behind.

Each model renders at its native size: 512 for the SD1.5 models and 1024 for SDXL. The exceptions are stated in the
specification. The resolution strip varies size on purpose, as the example of what not to do. The hires sweeps render
above native size, for the reason given under [Hires on each baseline](#hires-on-each-baseline).

## Presentations

Most strips are shown as a scrubber, one stop after another. Some parameters only make sense in another layout, so a
strip states a presentation and consumers are expected to honour it:

- A **grid** lays out two parameters at once, for example steps against cfg scale.
- **`tile_repeat_3x3`** draws each image repeated nine times in a square, beside the single image. Tiling does not
  teach anything when the images are only shown singly, and a three-by-three plate puts whole seams in the interior instead
  of only at the corner where four tiles meet.
- **`triptych`** shows three pictures per stop: the strip's source, the stop's control map, and the render that the
  map guided. Choosing a detector means choosing what the map keeps, so the map belongs on the page instead of in a
  description of it.

The triptych strips are the ControlNet gallery. They show every control type the horde serves, on one model and one
seed, so the detector is the only thing that changes. Each of the three worlds runs the gallery on three sources, and
the strip names which one in `subject`:

- `anchor` is the world's own tutorial picture.
- `building` is a structure with straight edges and deep perspective.
- `flat` is a drawing in solid colour, without shading.

Reading one detector across the three sources separates the detectors that trace outlines from the ones that read
light and shade. The flat source does not give the depth and surface detectors anything to work with, and the building
gives the line detectors more than a person does.

A ControlNet render alone does not show what its detector drew. Two detectors that read a scene very differently can
produce renders that look alike, and a detector that did not find anything looks, from the render, like one that
worked. So the map is published as evidence in its own right, beside the render it guided.

## Strips that are not one parameter

Two kinds of strip are exceptions. The index presents them like any other strip, with an empty `varying` list.

### One render, step by step

Thirty renders stopped at thirty different step counts are thirty different pictures. They do not show what one
render looks like while it is still running. The two step-preview strips are taken from a single render instead,
captured as it sampled. The sampler holds two different values at every step, and each is published as its own strip:

- `step_preview` is the canvas, `x`: what is actually there, still holding the noise that the remaining steps remove.
  It is almost pure static a third of the way in and only becomes legible near the end.
- `step_preview_guess` is the guess, `x0`: the model's prediction of the finished picture from the current step. It
  looks like a picture from the first step and like the final one well before the last. This is what the progress
  preview in a web UI shows.

Both strips are the 30 stops of one 30-step render at seed `1234567890`, so every stop states the same parameters.
The scene is the request of the txt2img tutorial example, with the same prompt and negative prompt, and the sampler is
pinned to `k_euler`. An ancestral sampler adds noise back at every step, so "static being removed" would be the wrong
description of what the reader is watching.

The capture wraps `latent_preview.prepare_callback` and `VAEDecode.decode` in the ComfyUI copy bundled with
horde-engine for the length of the render, and restores both afterwards. A recorder sits in front of the per-step
callback that the sampling graph already builds. The stored latents are decoded with the same VAE that decoded the
delivered image. The capture does not touch `comfy.utils.PROGRESS_BAR_HOOK`, because horde-engine installs that hook
once and owns it.

### The upscaler strip

The `upscaler_form` strip's stops are not renders. They are one image put through each upscaler that has weights
available, so an upscaler page can offer a scrubber over the same comparison that the alchemy manifest lists. Like the
step-preview strips, its stops are never selected by `local` or `horde`, because they are not generated.

The strip declares only the five upscalers that can be rendered. A consumer hides a strip that reports itself
incomplete, and a scrubber with six blank stops does not teach anything. The six upscalers without weights are still listed in
the alchemy manifest, which is designed to report a gap. [Alchemy evidence](alchemy-evidence.md) explains why.

## Crops and bounded copies

A page draws a picture at the width it has, so a 512-pixel render and a 2048-pixel upscale of it look the same once
both are scaled to fit. For upscalers, the hires pass and face restoration, the whole difference is in pixels that the
reader does not see at page size. Two derived files solve this:

- A **crop** is the part of the frame the strip is about, cut at the frame's own pixels. The region is stated in
  fractions of the frame instead of pixels, so one region works for stops published at different sizes. The upscaler
  strip's stops are 1024 and 2048 pixels, and the region is the same face in both. Alchemy entries declare a region the
  same way, and it is cut from the source and the result alike, so a before and after stay comparable when the result
  is four times the size.
- A **bounded copy** is the whole frame with its long edge reduced to 1024 pixels, so a page does not download several
  megabytes only to scale most of them away. A frame already within the bound does not get one.

The derived files are written by a separate `derivatives` step, so building the index stays a pure read of the
specification and the files on disk.

## Image-to-image sources

The two checked-in sources for image-to-image cover the two things it is used for:

- `skyharbor-deck-sunset.webp` is a render of the `person` scene, so the strip that restyles it works on a picture the
  reader has already seen.
- `rooftop-garden-sketch.webp` is a deliberately crude flat-colour drawing without shading, so the source does not
  contain anything that could be mistaken for detail kept by the model.

Each strip that starts from a source image publishes its own copy inside its own directory, because two sweeps of the
same parameter can start from different images.

## Masked modes

Which pipeline runs depends on the combination of `source_processing` and a mask, and the two masked modes behave very
differently:

- `img2img` with a mask encodes the whole picture and then restricts where noise is added. The masked pixels are the
  starting point, so denoising strength is a real control over them.
- `inpainting` erases the masked pixels before encoding, so strength does not have anything to vary. Below the strength
  at which the pass can finish, it returns a flat patch in the shape of the mask instead of a gentler edit. The grid
  publishes that flat patch on purpose.

Neither mode swaps the checkpoint, so the evidence is rendered on an ordinary model, which is what a real request gets.
The horde only requires the worker to keep an inpainting model available.

**White in a mask marks the area to repaint.** The backends invert the mask into the source's alpha channel, so a mask
that looks like the region you want to keep produces the opposite of what you meant. The three masked strips use
`deck-coat-mask.png`, which covers the courier's coat and leaves her face and the railing alone.

`outpainting` selects the same pipeline as `inpainting`, byte for byte, and **the backends do not pad a canvas**. The
caller pads the picture and masks what it added. The `deck-outpaint-*.webp` inputs and their masks were made from
`skyharbor-deck-sunset.webp` by pasting it into a wider black canvas and masking everything outside it. Each stop
states the size of its own canvas, because the backends resize a source to the requested dimensions and would
otherwise squeeze the padding back out.

## Hires on each baseline

Both backends silently ignore the hires flag unless **both** requested dimensions exceed the baseline's training size,
and they drop the flag entirely while inpainting or outpainting. The specification loader validates both rules, and a
third: setting `hires_fix_denoising_strength` without requesting the hires pass is refused, because the value is not read. A
sweep that would publish identical images as a comparison fails to load instead of rendering.

The two baselines do not fail at oversize in the same way, and the hires pairs record the difference. SD1.5 stacks a
subject up a tall frame, and its pair renders tall at 768x1536. SDXL composes a single clean figure at 1152x2048, and
again at 1536x3072, three times its training height. What breaks SDXL is square area, so its pair renders at
2304x2304. Holding both baselines to one frame shape would publish a pair that does not teach anything on one of them.
