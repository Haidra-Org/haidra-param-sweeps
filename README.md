# Parameter sweeps

Renders the illustration library the Learn surfaces teach generation parameters from: for each
parameter, a strip of images that differ in that parameter and nothing else, with a plain-language
caption on every stop.

One checked-in specification, [`sweep_spec.json`](sweep_spec.json), describes the whole library.
Two runners render it: **local**, through [horde-engine](https://pypi.org/project/horde-engine/)
on your own GPU, and **horde**, through the public AI Horde API. Both write the same files, so a
strip rendered one way is interchangeable with the same strip rendered the other way.

Unlike the other tools in this directory, this one is not standard-library-only: it has its own
`pyproject.toml` and is installed as a package.

## What it produces

```text
assets/param-sweeps/
  index.json                                  everything a web app needs to draw every strip
  steps/icbinp/person/001_4.webp              the image
  steps/icbinp/person/001_4.json              its sidecar: full params, model, seed, runner, timings
  upscaler-form/albedobase-xl/courier/001_realesrgan_x4plus_crop.webp   the region worth reading at its own pixels
  upscaler-form/albedobase-xl/courier/001_realesrgan_x4plus_fit.webp    the same frame bounded to 1024 on its long edge
  denoising-strength/icbinp/garden/source.webp       the image that strip starts from
  alchemy/manifest.json                       every alchemy form, its use case and its result
  alchemy/sources/tutorial-anchor-512.webp    a source the forms in one use case share
  alchemy/enlarge/realesrgan_x4plus/result.webp      what one form made of it
  alchemy/control_map/canny/result.webp       what one detector read out of the shared source

  ...
```

`index.json` is self-sufficient. Each strip carries its model, scene, presentation, the parameters
held constant, which of them vary, and its stops in order, each with a value, a filename and a
caption. A consumer that has `index.json` never has to open a sidecar or parse a filename.

Every stop states the pixel size of its own frame, read from the file rather than from what was
asked for, and a strip that starts from a source image states that image's size too. A frame whose
long edge is over 1024 also carries `fit`, a bounded copy for a consumer to draw in place of the
full frame; a frame already within the bound carries none and is drawn as it is.

Where the difference a strip teaches lives in detail nobody can see at the size a page draws a
picture, the sweep or the stop declares a `crop`: a region in fractions of the frame, cut at the
frame's own pixels into `<stem>_crop.webp` and reported in the index as a file with its pixel box.
Fractions rather than pixels, so one region serves stops published at different sizes: the
upscaler strip's stops are 1024 and 2048 and the region is the same face in both. Alchemy entries
declare one the same way and it is cut from the source and the result alike, which is what makes a
before and after comparable when the result is four times the size.

A stop does not repeat its own resolved parameters: the strip already states what it holds and
what varies, and the full set is in the sidecar beside every image for anyone who wants it. A
strip that starts from a source image names it in `source_image`, relative to the index, so a
consumer can show it beside the strip. Each such strip carries its own copy inside its own
directory, because two sweeps of the same parameter can start from different images.

Three presentations appear in the index, and a consumer is expected to honour them:

- `strip`: show the stops in order, as a scrubber.
- `grid`: lay the stops out row-major using the strip's `grid.rows` and `grid.columns` values.
- `tile_repeat_3x3`: draw each image repeated nine times in a square, beside the single image on
  its own. The tiling strip teaches nothing at all if the images are only shown singly, and a
  three-by-three plate puts whole joins in the interior rather than only at the crossing point.

Nearly every strip is N renders that differ in one parameter. Two kinds are the exception and read
the same way in the index, stating no varying field at all:

- The two `step_preview` strips, whose stops are the steps of a **single** render. See
  [One render, step by step](#one-render-step-by-step).
- The `upscaler_form` strip, whose stops are not renders: they are one image put through each
  upscaler the horde has weights for. See [Alchemy](#alchemy).

The alchemy tree beside the index is its own thing, with its own manifest, and a consumer reads it
without going through `index.json`.

Filenames are deterministic: the same specification and seed always produce the same paths, so
re-rendering a strip overwrites it in place and never leaves an orphan behind.

## The specification

`sweep_spec.json` holds the anchor parameters every sweep holds constant, the models, the scenes,
and each sweep with a one-line rationale for the models it runs on.

The scenes belong to three worlds the whole library shares, rather than a new subject per lesson:
a skyharbor courier for the photoreal scenes, a fox shrine for the anime scene, and a rooftop
gardening robot for the scenes that need a non-human subject. Each of those worlds has one
descriptor block naming its subject, and that block is repeated byte for byte in every scene the
subject appears in: a subject re-described in different words is a different subject, and a strip
of it would teach the prompt rather than the parameter it is sweeping. `tests/test_spec.py`
asserts the blocks are identical wherever they appear.

Scenes are written in the prompt idiom of the models that render them: comma-separated tags for
the SD1.5 models, booru tags with a `rating_safe` token for the anime model. A scene is one prompt
string shared by every model in a sweep, so a scene rendered by more than one model is written in
the idiom of the more literal-minded of them.

Every image uses seed `1234567890` except the seed strip, which is what it teaches.

A sweep can state a `holds` block of parameter values every one of its stops inherits, which is
how a strip pins something the anchor sets differently (the converging-steps strips pin the
sampler that way). A stop's own values win over it, and the anchor wins over neither.

A stop can also carry a `prompt_fragment`, appended to the scene's prompt. That is how the
prompt-emphasis strips vary `(a phrase:weight)` while holding every other parameter still, and
the fragment is republished per stop in the index so a consumer can show what changed.

A sweep that names a `source_image` renders image-to-image from it, on both runners. The path is
relative to this directory, the image is checked in under `inputs/`, and it is copied into the
library so the published tree stays self-contained. The two checked-in sources are the two things
image-to-image is used for: `skyharbor-deck-sunset.webp` is a render of the `person` scene, so the
strip that restyles it works on a picture the reader has already met, and
`rooftop-garden-sketch.webp` is a deliberately crude flat-colour drawing with no shading in it at
all, so nothing in the source can be mistaken for detail the model kept. A sweep that names a
`negative_prompt` holds the text at the sweep and lets each stop turn it on or off with
`use_negative_prompt`.

A sweep or a stop can also name a `source_mask` and a `source_processing`. **White in a mask marks
what you are asking to be repainted**; the backends invert it into the source's alpha, so a mask
that looks like the region you want kept produces the opposite of what you meant. The three masked
readings run on `deck-coat-mask.png`, which covers the courier's coat and leaves her face and the
railing alone.

Which pipeline runs is decided by the combination, and the two masked modes are genuinely
different. `img2img` with a mask encodes the whole picture and then restricts where noise lands,
so the masked pixels are the starting point and the strength is a real dial over them. `inpainting`
erases them before encoding, so there is nothing to vary: below the strength at which the pass can
finish, it returns a flat patch the shape of the mask rather than a gentler edit. The grid publishes
that flat patch on purpose. Neither mode swaps the checkpoint, so the evidence renders on an
ordinary model, which is what a real request gets: the horde only requires the *worker* to keep an
inpainting model on hand.

`outpainting` selects the same pipeline as `inpainting`, byte for byte, and **nothing in the
backends pads a canvas**. The caller pads the picture and masks what it added. The
`deck-outpaint-*.webp` inputs and their masks were produced from `skyharbor-deck-sunset.webp` by
pasting it into a wider black canvas and masking everything outside it; each stop states the size
of its own canvas, because the backends resize a source to the requested dimensions and would
otherwise squeeze the padding straight back out.

Each model renders at its own native size (512 for the SD1.5 models, 1024 for SDXL). The
exceptions are stated in the specification: the resolution strip is the anti-example and varies
size on purpose, and the hires sweeps render above native, because both backends silently ignore
the flag unless **both** requested dimensions exceed the baseline's training size. The backends
also drop the flag entirely while inpainting or outpainting. Both rules are validated when the
specification loads, along with a third: stating `hires_fix_denoising_strength` without asking for
the pass is refused, because nothing reads it. A sweep that would have published identical images
as a comparison fails rather than rendering.

The two baselines do not fail at oversize in the same shape, which the hires pairs record. SD1.5
stacks a subject up a long frame and its pair renders tall at 768x1536. SDXL composed a single
clean figure at 1152x2048 and again at 1536x3072, three times its training height; what breaks it
is square area, so its pair renders at 2304x2304. Holding both to one frame shape publishes a pair
that teaches nothing on one of them.

Edit the specification to change what the library contains. Nothing about the sweeps is in the
code, apart from the row-major expansion of the grid.

## QR codes that scan

Two strips and one tutorial flow render through the `qr_code` workflow, which lays a QR ControlNet
over an ordinary render. Three things decide whether what comes out is worth publishing.

**The prompt must not describe the code.** The ControlNet already carries the geometry. A prompt
that asks for blocks, grids and hard black gaps competes with it, and the result is a barcode with
weather behind it rather than a picture with a code inside it. The prompts here ask for vapour,
falling leaves and climbing ivy: soft, flowing subjects that give the modules something to be made
of. The `cloud-sea` scene exists for exactly this reason, and has no straight line in it.

**`control_strength` is the dial.** Low, and the picture wins until no scanner can find the code;
high, and the code sits on the picture like printed ink. The usable range is narrower than it
looks and it moves with the scene: on `cloud-sea` at 1024 with AlbedoBase XL, nothing at or below
1.2 decoded on any seed tried, 1.25 is the lowest that does, and by 1.6 the code has taken over.
That range is what the `control_strength` strip publishes, stop by stop.

**Where the dial is accepted.** The horde takes `control_strength` alongside a `control_type` or the
`qr_code` workflow, and refuses a request that sets the strength with neither. A job that carries it
reaches only a recently updated worker; older ones are never handed it. The strength strips here were
rendered locally, which is the same picture a current worker returns.

**Every published code is decoded before it ships.** Not by a contrast heuristic: by `zbar`,
through `pyzbar`, against the image exactly as it is published and at the size it is published at.
A single-image render that does not decode is failed by the campaign driver and never written. A
curation batch is not failed that way, because its published frame is whichever candidate a human
picks; every candidate's decode is logged instead, and the gate over what actually shipped is:

```sh
uv run param-sweeps qr-check
```

which scans every QR image the index and the tutorial manifest reference, and exits non-zero if
any of them reads back anything other than the payload it was asked to carry.

`pyzbar` is a base dependency rather than an extra, unlike `boto3`. It is a few hundred kilobytes
including its bundled `zbar` library, it is needed wherever QR images are rendered as well as
wherever they are checked, and a rendering environment that cannot tell whether the code it just
made scans is the failure this whole section exists to prevent.

## Setup

```sh
cd docs/haidra-assets/tools/param_sweeps

# The tool, its tests and its linters. No GPU stack.
uv sync --group dev

# Add the object-storage SDK, only if you are publishing from this environment.
uv sync --group dev --extra publish
```

That is everything the `horde` runner, the specification and the index need. `boto3` is not a base
dependency: see [Publishing](#publishing).

### The local runner: reuse an existing environment first

**If the machine already runs an AI Horde worker or carries a `hordelib` dev checkout, reuse that
environment instead of building a new GPU stack** — torch alone is a multi-GB download, and exact
torch versions do not matter for this workload. Install just the tool into it, leaving its
dependencies untouched:

```sh
uv pip install --python <existing-env>/Scripts/python.exe -e . --no-deps

# then run every command through that interpreter:
<existing-env>/Scripts/python.exe -m param_sweeps plan
```

The tool's own runtime dependencies (pydantic, strenum, requests, pillow) are already satisfied by
any environment that runs `horde-engine`. The same worker install usually also means the model
weights are already in `AIWORKER_CACHE_HOME` — the fetch plan will confirm.

### Fresh install: getting the GPU backend right first

**Install the torch build for your GPU before installing `horde-engine`.** `horde-engine` depends
on torch, and if it is installed first, the default index gives you whichever build pip would pick
rather than the one your card can use. Reversing that afterwards means uninstalling torch and its
CUDA payload, so it is worth a minute up front.

For a CUDA card:

```sh
# 1. torch for CUDA first. Several GB.
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 2. then horde-engine, which will keep the torch you installed
uv sync --group dev --extra local
```

For an AMD card, install a ROCm torch build in step 1 instead, and run with `--gpu-backend rocm`.
The backend is never guessed: it comes from `--gpu-backend`, then `PARAM_SWEEPS_GPU_BACKEND`, then
CUDA. Supported values are `cuda`, `rocm`, `xpu`, `npu`, `mlu`, `mps`, `directml` and `cpu`.

Model weights come from the standard worker cache, so set `AIWORKER_CACHE_HOME` to an existing
worker's cache and nothing is downloaded twice:

```sh
export AIWORKER_CACHE_HOME=/path/to/your/worker/cache
```

### The horde runner and its key

The runner reads `AIHORDE_DEV_KEY` from the environment, and failing that from the frontpage
repository's `.env` (`../../../../.env` from this directory), which is the default. Point it
somewhere else with `--env-file` or `PARAM_SWEEPS_ENV_FILE`. The key is only ever put in a request
header: it is never printed, logged or written into any output.

## Running

```sh
# What would be downloaded, and how many images, without generating anything
uv run param-sweeps plan

# The whole library on the local GPU
uv run param-sweeps local --gpu-backend cuda

# The whole library through the horde, one job at a time
uv run param-sweeps horde

# Cut the crops and write the bounded copies for whatever images are already on disk, which
# `index` then reports; it answers to `crops` as well, which is what it used to be called
uv run param-sweeps derivatives

# Rebuild index.json from whatever images are already on disk
uv run param-sweeps index

# Sync the rendered library to object storage (see Publishing)
uv run param-sweeps upload --dry-run
```

Both runners print the fetch plan (models, approximate download size, whether each is already on
disk, image counts) and the total image count before they generate anything.

By default the library is written to `assets/param-sweeps/` at the root of this repository. Send it
elsewhere with `--output-root`.

## Signing off a batch

A specification with `batches` is reviewed from a contact sheet before anything ships:

```sh
# Write review.html at the root of the output tree
uv run param-sweeps review --spec signoff_batch_spec.json

# Write the picks saved from the sheet back into the specification, and refresh the sheet
uv run param-sweeps choose --spec signoff_batch_spec.json path/to/signoff_choices.json
```

Open `review.html` straight from disk. A job that rendered several candidates shows all of them
with the current pick outlined; click (or focus and press Enter) a candidate to pick it instead,
then use **Save choices** to download `signoff_choices.json`. Picks are kept in the browser's
local storage until saved, so a review can span sittings. The `choose` subcommand validates the
file against the specification, updates only the `chosen` fields, and rewrites the sheet so its
outlines match. The sheet renders fully without JavaScript; picking by hand in the specification
(`"chosen": <one-based candidate>`) remains equivalent.

### Rulings

A specification may also carry a `rulings` block: a list of ruling sets, each of which is the
questions one review wave has to answer and each of which the sheet renders as a panel above the
images. More than one set, because a wave that has been ruled on is usually followed by a shorter
one asking only what the first left open: answers are keyed by the set, so a follow-up set opens
empty instead of reopening decisions that were already made.

Each ruling states its question, links to the section or images that decide it, and offers the
control it is answered with (one option, a checklist, one answer per row, or free text). Every
ruling that offers options also offers **Other / see comments**, and every ruling carries a comment
box, so an answer the option set does not cover is still recorded as an answer.

```jsonc
"rulings": [
  {
    "ruling_set_id": "signoff-solver-knobs",   // what saved answers are keyed by
    "title": "Rulings this wave needs",
    "items": [
      {
        "ruling_id": "dpmpp-3m-sde-normal",
        "title": "dpmpp_3m_sde on the normal schedule",
        "question": "Does the divergence hold?",
        "control": "radio",                     // radio | checkbox-set | row-select | text
        "evidence_sections": ["f-raw-backend"], // links to the section heading
        "evidence_job_ids": ["raw-3m-sde-deliberate-normal-08"],  // links to one figure
        "options": [
          { "choice_id": "confirmed", "label": "Confirmed broken" },
          { "choice_id": "not-confirmed", "label": "Not confirmed" }
        ]
      }
    ]
  }
]
```

A `checkbox-set` may state a `master_toggle_label` to tick every listed box at once, and each
option may name an `evidence_job_id` of its own, which is what puts a **view** link beside each
approval. A `row-select` states `rows`; the rows share the ruling's `options` unless a row states
its own set. Every cited section and job id is validated against the batches, so a link that goes
nowhere fails to load rather than misleading a reviewer.

Answers live in the browser's local storage under the ruling set's id, so a review can span
sittings and neither two sets on one sheet nor two waves reviewed on one browser can collide. Each
panel counts how many of its own questions are still open (`N of M ruled`), and its **Export
rulings** downloads `<ruling_set_id>_rulings.json` and puts the same text in a box to copy out of
the page:

```jsonc
{
  "schema_version": 1,
  "ruling_set_id": "signoff-solver-knobs",
  "exported_at": "2026-08-03T19:11:43.333Z",
  "answered_count": 4,
  "total_count": 11,
  "unanswered": ["karras-at-low-steps"],
  "rulings": {
    "dpmpp-3m-sde-normal": { "control": "radio", "answer": "confirmed", "comment": "" },
    "minted-golden-correctness": { "control": "checkbox-set", "selected": ["seeds_2"], "comment": "" },
    "schedule-character-claims": { "control": "row-select", "rows": { "beta": "accurate" }, "comment": "" }
  }
}
```

Nothing is uploaded and nothing is written back into the specification: a ruling is a decision for
a human to carry out, not a field the tool edits.

## Resuming

Every image is checkpointed the moment it is written, in `.param-sweeps-checkpoint.json` at the
root of the output tree. Interrupt a run however you like and start it again: it renders only what
is missing. A job that fails is reported at the end and left pending, so the rest of the campaign
still runs and the next invocation retries it.

An entry only counts while its image is still on disk, so **deleting an image is all it takes to
have it regenerated**. There is no bookkeeping to edit.

`index.json` is rewritten as the run goes, and always describes only images that exist. A strip
that is partly rendered appears with `"complete": false`.

## Regenerating one strip

The filters compose, and a single strip is all three of them:

```sh
# One strip: this parameter, this model, this scene
uv run param-sweeps horde --parameter clip_skip --model nova-anime-xl --scene anime

# Everything for one parameter, across every model and scene it runs on
uv run param-sweeps local --parameter steps

# Re-render a strip that is already complete
uv run param-sweeps local --parameter tiling --model icbinp --scene thing --force
```

Without `--force`, images already on disk are left alone. With it, the selected images are
re-rendered and overwritten in place.

## Measuring what a sampler costs

`sampler-cost` renders no library at all. It answers one question: what does a step of each sampler
cost, relative to `k_euler`, through the same pipeline everything else here renders on.

A cost taken from a single render is three numbers added together, and only one of them is the
sampler: the model load, the fixed per-job overhead, and the sampling itself. This mode separates
them by rendering the same job at several step counts and fitting a line. The slope is the cost of
a step, the intercept is everything that does not scale with steps, and the ratio of one slope to
`k_euler`'s is the figure a cost model wants.

```sh
# What would be measured, on which schedules, and how many renders that is
uv run param-sweeps sampler-cost --plan-only

# The whole measurement: every known sampler, on one SD1.5 model and one SDXL model
uv run param-sweeps sampler-cost

# A short run to prove the harness, on two samplers and two step counts
uv run param-sweeps sampler-cost --model deliberate --sampler k_euler --sampler k_dpmpp_2m \
  --steps 10 --steps 20 --repeats 1
```

The sampler vocabulary and the schedule pairings are read from `horde_sdk`'s image sampler
constraints table rather than restated here, so a sampler added there is measured here without this
tool being touched. **The installed `horde_sdk` has to be one that carries
`horde_sdk.generation_parameters.image.constraints`**, and the installed `horde-engine` has to be
one whose sampler map knows the samplers being measured; the command says so plainly when the table
is absent.

Every sampler is measured on `karras`, because a measurement whose schedule moves with the sampler
measures two things at once. A sampler the table refuses that pairing for takes the first schedule
it advises instead, and the artifact records that it was substituted.

Everything else is held constant: one seed, one prompt, one negative prompt, one cfg scale, and
each model at its own native size. Discarded warmup renders precede each model's timed renders, so
the first render's graph building does not land in the intercept. Each point is the median of its
repeats.

The artifact lands in `measurements/` under the output root, one JSON file per run, named after the
run's start time so a re-measurement never overwrites what it is meant to be compared against. Each
record carries the schedule used, every raw timing, the fit and its R-squared, the ratio against
`k_euler`, and the ratio the sampler's evaluations per step predict. A measured ratio more than 20%
away from that prediction is flagged as discrepant and nothing is adjusted: the flag is the finding.

## One render, step by step

`step-preview` answers a question the rest of the library cannot: what does a render look like
while it is still running. Thirty renders stopped at thirty step counts are thirty different
pictures, so the frames come from one render instead, captured as it sampled.

```sh
# The whole thing: one render, sixty frames, both strips and the index
AIWORKER_CACHE_HOME=... uv run param-sweeps step-preview
```

It renders the specification's `step_preview` stop once on the local GPU with a recorder in front
of the per-step callback the sampling graph already builds, then decodes what it stashed with the
same VAE the graph decoded the delivered image with. It publishes two strips from that one render,
because the sampler carries two different things at every step:

- `step_preview`: the canvas, `x`. What is actually there, still carrying the noise the remaining
  steps burn off. It is almost pure static a third of the way in and legible only near the end.
- `step_preview_guess`: the guess, `x0`. Where the model believes the whole picture is heading from
  where it stands. It looks like a picture from the first step and like the final one well before
  the last, which is what a progress preview in any web UI is actually showing you.

Both strips are 30 stops of one 30-step render at seed `1234567890`, so every stop states the same
parameters and the index reports nothing varying across them. The scene is the txt2img tutorial's
own request, prompt and negative prompt verbatim, with the sampler pinned to `k_euler`: an
ancestral sampler adds noise back at every step, which would make "static burning off" the wrong
description of what the reader is watching.

The capture wraps `latent_preview.prepare_callback` and `VAEDecode.decode` inside the vendored
ComfyUI for the length of the render and puts both back afterwards. It never touches
`comfy.utils.PROGRESS_BAR_HOOK`, which horde-engine owns and installs once.

## Alchemy

Alchemy is the half of the horde that takes an image in rather than a prompt: upscalers, face
restorers, background removal, and the describe family that answers in words instead of pixels.
`alchemy` applies every form the horde knows to a source image and writes the evidence the site's
five alchemy pages read.

```sh
# Which forms exist and which page each one lands on. Applies nothing.
uv run param-sweeps alchemy --plan-only

# The post-processors: every form horde-engine can run, the manifest, the strip and the index
AIWORKER_CACHE_HOME=... uv run param-sweeps alchemy

# The describe family, which only the horde can answer
uv run param-sweeps alchemy --backend api --form caption --form interrogation --form nsfw

# The control maps, drawn by a horde-image-utilities service listening locally
uv run param-sweeps alchemy --backend hiu

# One detector again, discarding what is already on disk
uv run param-sweeps alchemy --backend hiu --control-type openpose --force

# One form again, discarding what is already on disk
AIWORKER_CACHE_HOME=... uv run param-sweeps alchemy --form GFPGAN --force

# The face restorers the pending model reference carries, rather than the published one
AIWORKER_CACHE_HOME=... uv run param-sweeps alchemy --beta --form GFPGANv1.3 --form RestoreFormer
```

Three backends, because the horde runs the forms in three places. `--backend hordelib`, the
default, applies the post-processors on this machine's GPU through the same stack a worker runs.
`--backend api` submits the image to the public horde's `/interrogate` endpoints and publishes what
came back, which is the only way to show what the describe family says without inventing it; it
reads `AIHORDE_DEV_KEY` exactly as `flows` does. Restrict an API run to the forms it is for: the
API serves the post-processors too, and an unrestricted run would spend kudos re-doing work the
GPU already did.

`--beta` widens where the local backend looks for weights. GFPGANv1.3 and RestoreFormer are named
by the shared vocabulary but carried by the model reference service's pending queue rather than by
the published reference, so without it two of the four face restorers resolve to nothing and
publish as gaps. The flag sets the three variables horde-engine reads before it builds its model
managers: `HORDELIB_BETA_MODEL_CATEGORIES` to the categories being opted in (`gfpgan` today, and
the constant is a tuple so `esrgan` can join it when the beta upscalers are worth publishing),
`HORDELIB_BETA_MODELS_API_KEY` to the anonymous key, and
`HORDE_MODEL_REFERENCE_PRIMARY_API_URL` to `https://models.aihorde.net/api`. A variable already
set in the environment is left alone, so a run with its own key or its own service keeps them.
Beta resolution is additive: a category opted in whose pending fetch comes back empty falls back
to the published reference rather than failing the run.

`--backend hiu` draws the control maps by asking a **horde-image-utilities** service, which is the
process an alchemist worker runs beside itself for the `annotation` form. Point it somewhere with
`--hiu-url` or `PARAM_SWEEPS_HIU_URL`; it defaults to `http://127.0.0.1:7860`. A run through it
covers the `annotation` form on its own, because the other forms are answered elsewhere and an
unrestricted run would overwrite what those backends recorded about them. `--control-type` narrows
it to one detector. The first call to a cold detector fetches its weights, which for the
segmentation and depth models takes minutes, so the service is given ten of them per detection.
The sidecar beside each map records the service's host and the version it reports, because a
control map is the output of a particular build of a particular detector.

The form list is not this tool's to invent. It is read out of `horde_sdk`'s `KNOWN_ALCHEMY_TYPES`,
and `alchemy_spec.json` places each of those forms into exactly one of five use cases: `describe`,
`control_map`, `enlarge`, `fix_faces` and `cut_out`. A form the SDK gains and the specification has
not placed fails to load the file rather than quietly vanishing from the published evidence. A few
forms fit none of the five honestly (the NSFW check, the aesthetic score, the palette, the
vectorizer), and each is placed where it lies least and says so in its notes.

The `annotation` form is the exception to one form, one entry. Which detector runs is the whole
answer, so it is placed once per control type `horde_sdk`'s `KNOWN_ANNOTATION_CONTROL_TYPES` names,
its entries carry a `control_type` and are told apart by it in the tree and in the manifest, and a
detector the SDK gains and the specification has not placed fails to load the file in the same way
a form does.

Every form in a use case runs on the **same** source, which is the whole point. Two upscalers are
only comparable if the picture underneath them is the same picture, so the sources are published
once at the root of the tree and every entry points at one of them:

| Use case | Source |
|---|---|
| `describe` | the courier at 1024, the txt2img tutorial's own anchor |
| `control_map` | the courier at 1024, so every detector reads out of one picture |
| `enlarge` | the courier at 512, small enough that an upscaler has something to do |
| `fix_faces` | the degraded portrait from hordelib's face fix tests, the kind of picture blind face restoration was built for |
| `cut_out` | the robot gardener, whose silhouette is clean enough to see where the cut fell |

`assets/param-sweeps/alchemy/manifest.json` lists every placed form whether or not it produced
anything. An entry with no result is the honest report that the form could not be run here, and its
notes say why; leaving it out would tell a reader the form does not exist. Two things make that a
routine state rather than a failure:

- **Only the post-processing forms have a horde-engine surface at all.** Captioning, interrogation,
  the NSFW check and the other read-only forms are served by the alchemist's own backend, which is
  not this stack, so `--backend api` asks the horde for them instead of approximating them with
  something else that happens to be installed. The control maps are not on that backend either:
  they are drawn by horde-image-utilities, and the entries say so unless the run that produced
  them was a `--backend hiu` one.
- **The shared vocabulary names weights the live model reference does not carry.** Six of the eleven
  upscalers have no entry to resolve, so there is nothing to download and nothing to run. All four
  face restorers are expected to render locally: GFPGAN and CodeFormers resolve from the published
  reference, and GFPGANv1.3 and RestoreFormer from the beta model reference.

Background removal needs one thing the base install does not have:

```sh
uv pip install "rembg[cpu]"
```

Without it the `strip_background` form reports itself unavailable, in the same words horde-engine
uses, rather than failing the run. The horde's production background removal is moving off this
path onto `horde-image-utilities`, which the worker runs as its own process in its own environment;
the published cut-out is regenerated through that when it lands.

The upscalers additionally publish into the sweep library as an ordinary strip, under the
`upscaler_form` parameter: one source, one stop per upscaler that has weights behind it, so the
enlarge page gets a scrubber over the same comparison the manifest lists. Like the two step-preview
arcs, its stops are never selected by `local` or `horde` — nothing about them is generated. The
strip declares only the five renderable stops, because a consumer hides a strip that reports itself
incomplete and a scrubber with six blank stops teaches nothing; the six upscalers with no weights
stay disclosed through their manifest entries, which is the surface built to admit a gap.

## The tutorial worked examples

`flows_spec.json` holds the tutorial worked examples the Learn area's technique pages and studio
walkthroughs read: one committed request per technique, in each of the three worlds where the
technique can be drawn, with a `--w2` / `--w3` suffix on the world variants. `flows` renders them
and writes `assets/param-sweeps/tutorials/<flow>/` (the result, the source and mask where there is
one, `reference-N.webp` for each extra image a remix blended in, the published request with its
image fields as placeholders, and `meta.json`) and then rebuilds `tutorials/manifest.json` from
whatever is on disk.

```sh
# Through the public horde, one flow at a time (reads AIHORDE_DEV_KEY as the horde runner does)
uv run param-sweeps flows --flow img2img

# On this machine's GPU, through the worker software, with no queue or other hardware in between
uv run param-sweeps flows --local --flow inpainting

# Rewrite tutorials/manifest.json from what is on disk, generating nothing
uv run param-sweeps flows --manifest-only
```

`meta.json` records which runner made the example (`runner`); a local render is charged nothing
and carries no kudos lines. Every selected flow is regenerated: there is no skip for a flow that
already exists, so name the flows whose requests changed. A flow that cannot honestly be drawn in
a world is marked `skipped` with a `skipped_reason` and left out of the manifest, and the Learn
surfaces show the first world's example in its place and say so.

### What the flows taught, and what is marked for further work

Some recipes took several rounds to land, and the record of that is kept in two places so the
next campaign starts from it: the flow's `notes` say what the published recipe is and why, and a
`needs_attention` field on the flow says what is still a compromise, what was tried, and what to
vary next. `flows --dry-run` prints every flow that carries the marker. Clear it when the flow is
settled; do not delete the lesson from `notes`.

- **Inpainting fills from context.** An SD1.5 inpainting model fills a masked hole with what its
  surroundings suggest, so a masked sky comes back as sky at the anchor guidance and the prompt is
  ignored. Raising guidance to 11 or weighting the subject `(a phrase:1.4)` both make the object
  appear; the flows use the weight, so the guidance every other example shares is unchanged. Keep
  the mask clear of the figure: the mask edge is grown by a few pixels before encoding, and a mask
  that clips hair puts a hat on the subject. The anime inpainting checkpoint the horde serves
  repaints the whole frame washed out rather than the hole alone, which is why the fox-shrine
  inpainting flow is skipped and marked.
- **The QR ControlNet is defeated by geometry in the prompt.** A skyline, a garden or a shrine
  named in the prompt takes the pixels the code needs, and the result does not decode; weather and
  light alone, with the place named only as somewhere far below, scans on the first seed tried.
  Even so, a decode is not guaranteed by the recipe: the flows were rendered over a run of seeds
  and the first that decoded was published, and the seed a flow carries is that one. The SDXL
  anime checkpoint draws flat areas and flattened the modules away on every seed; the SD1.5 anime
  checkpoint at 512, with the SD1.5 QR ControlNet, decoded on three seeds of four, and that is
  what the fox-shrine QR flow uses.
- **A world is a subject and a model, and not every lesson survives every world.** The robot has
  no face to fix, so face fix has no third world; the anime inpainting checkpoint fails as above.
  A skipped world falls back to the first world's example, and the surfaces say so.
- **Render locally to iterate.** The horde is the reference runner for the published set, but a
  recipe search through a queue is slow and confounded by whichever worker answers; `flows --local`
  renders through the same worker software with nothing in between, and a run of seeds is a
  minute each.

## Publishing

The library is around 70 MiB across 1400 files, so it is not committed: `assets/param-sweeps/` is
gitignored, and the published copy lives in S3-compatible object storage. The intended host is
Cloudflare R2, which is a plain S3 API behind an account endpoint URL; nothing in the tool is
specific to it, and an AWS bucket or MinIO works the same way with a different endpoint.

### The publish extra

The object-storage SDK is an optional extra, not a base dependency:

```sh
uv sync --extra publish            # or: pip install "haidra-param-sweeps[publish]"
```

Publishing is this tool's own concern. A rendering environment shares its interpreter with a GPU
stack that has nothing to do with object storage, so installing the tool there must not drag an SDK
in with it. `param_sweeps.publish` imports `boto3` where it builds a client rather than at the top
of the module, which keeps `python -m param_sweeps` working in an environment without the extra:
every command that does not reach a bucket, `manifest` and `verify --local` included, runs normally.
A command that does reach one says what to install and exits 1, rather than failing on an import.

### The addressing contract

**An object's key is the path the index or a manifest already states.** A consumer joins its
configured base URL to that string and it has the file:

```text
{evidenceBaseUrl}/index.json
{evidenceBaseUrl}/{strips[].source_image}
{evidenceBaseUrl}/{strips[].stops[].file}
{evidenceBaseUrl}/{strips[].stops[].sidecar}
{evidenceBaseUrl}/tutorials/manifest.json
{evidenceBaseUrl}/tutorials/{flows[].directory}/{flows[].files[]}
{evidenceBaseUrl}/alchemy/manifest.json
{evidenceBaseUrl}/alchemy/{entries[].files.source|result|result_text|result_json}
```

There is no translation step on either side, which is the point: moving the library between hosts,
or between a prefix and the bucket root, is a change to one configured string and nothing else. The
optional prefix is applied when the objects are written, so it belongs to the base URL, never to a
path out of the index.

Three entrypoints are the source of truth for what belongs in the library: `index.json` for the
sweeps, `tutorials/manifest.json` for the worked examples and `alchemy/manifest.json` for the
post-processor evidence. The index is required; each manifest is published with everything it names
when it is present. A file on disk that none of them references (a provenance `meta.json` beside an
alchemy result, the curation batch, the review sheet) is not published, and an object in the bucket
that none of them references is an orphan.

**Sidecars are published.** The index names one per stop, so a bucket where those keys 404 would
make the index dishonest, and `verify` would need a special case to stay quiet about it. They are
also cheap: at roughly a kilobyte each they are about 2% of the library. Nothing on the web side is
obliged to fetch them; they are there so that "everything the index names resolves" holds without
qualification.

### Configuration

Every setting is an environment variable, read from the process environment first and from a `.env`
second (by default the frontpage repository's, the same file the API key lives in, which is
gitignored; point elsewhere with `--env-file`). The secret is only ever handed to the S3 client: it
is never logged, printed or written into any output.

| Variable | Required | Meaning |
| --- | --- | --- |
| `PARAM_SWEEPS_S3_ENDPOINT_URL` | yes | The S3 endpoint. For R2, `https://<account-id>.r2.cloudflarestorage.com`. |
| `PARAM_SWEEPS_S3_BUCKET` | yes | The bucket the library is published into. |
| `PARAM_SWEEPS_S3_ACCESS_KEY_ID` | yes | The access key ID of an API token with object read and write. |
| `PARAM_SWEEPS_S3_SECRET_ACCESS_KEY` | yes | That token's secret. |
| `PARAM_SWEEPS_S3_PREFIX` | no | A key prefix, so one bucket can hold more than one thing. Defaults to the bucket root. Overridden by `--prefix`. |
| `PARAM_SWEEPS_S3_REGION` | no | Defaults to `auto`, which is what R2 wants. An AWS bucket wants its real region. |

### Runbook

```sh
# 1. What is in the library, and how large it is
uv run param-sweeps manifest

# 2. Confirm the files on disk match what the index claims. No credentials needed.
uv run param-sweeps verify --local

# 2a. Confirm every published QR code still decodes to its payload
uv run param-sweeps qr-check

# 3. What an upload would do, without touching the bucket
uv run param-sweeps upload --dry-run

# 4. Do it
uv run param-sweeps upload

# 5. Confirm the bucket now matches the index, by content hash
uv run param-sweeps verify
```

Then point the web app at the bucket's public base URL, prefix included. On the frontpage side
that is `evidenceBaseUrl` in `src/environments/*.ts` (which this repository does not edit: it is a
submodule of the frontpage, and the setting belongs to the consumer). The bucket must answer
cross-origin `GET` from the site's origin: the app fetches the index and the manifests with
`XMLHttpRequest`, which the browser refuses without an `Access-Control-Allow-Origin` header, while
the images load through `<img>` regardless. That is a CORS policy on the bucket, not anything here.

`upload` is incremental. It lists the bucket, compares each file's MD5 against the object's ETag,
and uploads only what is new or changed; a re-run after a no-op render uploads nothing. Objects are
written with `put_object` so every upload is single-part and the ETag stays equal to the MD5, which
is what makes that comparison sound.

Ordinary objects are put eight at a time: each is one small request, so a serial run is bound by
round trips rather than bytes. **The index and the manifests are uploaded last**, one at a time,
after every other object has landed. A bucket therefore never serves an entrypoint that advertises
files which have not arrived yet. The worst an interrupted upload leaves behind is objects nothing
points at, which the next run reconciles.

Orphans are reported by `upload` and `verify` but never removed unless asked:

```sh
# Remove objects under the prefix that neither the index nor a manifest references
uv run param-sweeps upload --delete --dry-run   # read this first
uv run param-sweeps upload --delete
```

That flag matters after stops are inserted into a strip, because inserting a stop renumbers the
ones after it and the old filenames become orphans.

`verify` exits non-zero when anything the index or a manifest references is missing from the bucket or has a
different hash, so it is usable as a gate. Orphans alone do not fail it: they are reported and left
to a human.

### Caching

Uploads set `Content-Type` (`image/webp`, `application/json`) and `Cache-Control`:

- Images, sidecars and everything else a manifest names: `public, max-age=604800, stale-while-revalidate=86400`.
- `index.json`, `tutorials/manifest.json`, `alchemy/manifest.json`: `public, max-age=300, must-revalidate`.

Images are **not** immutable, so the long header is a considered trade rather than a `immutable`
claim: inserting a stop renumbers the ones after it, and a path can come to hold a different image.
A week keeps consumers from refetching between publishes while letting a renumbering settle on its
own. A publish that renumbers and must be visible at once wants a cache purge at the CDN, not a
shorter header here. The index and the two manifests are the mutable entrypoints every consumer
starts from, so they are held for five minutes and no longer.

## Checks

```sh
uv run ruff check .
uv run black --check .
uv run mypy
uv run pytest
```

The tests cover the specification's shape (the rules the library is supposed to obey, such as
clip-skip running on three models and tiling being taught on a texture and counter-taught on an object), the
resume behaviour, the index, and the sync planning behind `upload` and `verify`. They do not
generate images, and they reach no network: the bucket is a `botocore` stub, so what is proven is
the plan, the ordering and the headers rather than any particular provider's behaviour.

`tests/test_alchemy.py` drives the whole alchemy campaign through a scripted backend, so the
manifest's shape, the coverage of the form and control-map vocabularies, the text results and the
strip are all checked without a GPU. `tests/test_image_utilities.py` does the same for the
control-map service, answering the backend from a scripted transport mounted on a real session, so
what the service says about a detector it will not run is checked without a service.

`tests/test_publish.py` skips itself, with a reason, in an environment without the publish extra:
there is no client to stub there. It also carries the test that keeps the extra optional, which
imports the tool in a subprocess and fails if `boto3` or `botocore` ended up in `sys.modules`.
