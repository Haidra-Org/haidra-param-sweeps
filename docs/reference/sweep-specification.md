---
title: "Sweep specification"
summary: "The format of sweep_spec.json: anchor parameters, models, scenes, strip and grid sweeps, stops, source images, masks, crops and the rules checked on load."
order: 10
---

# Sweep specification

`sweep_spec.json` describes the whole library: the anchor parameters that every sweep holds constant, the models, the
scenes, and each sweep with a rationale for the models it runs on. Edit the specification to change what the library
contains. The code does not contain anything about the sweeps, apart from the row-major expansion of a grid. The
file is validated when it loads, and an invalid file fails before anything is rendered.

The reasons behind the shared scenes, the presentations and the masked and hires rules are in
[One parameter per strip](../explanation/one-parameter-per-strip.md). Review batches and review questions, which live in
the same file, are in the [review sheet reference](review-sheet.md).

## Code map

| Concept | File | Symbol |
| --- | --- | --- |
| Whole specification | `param_sweeps/spec.py` | `SweepSpec`, `load_spec` |
| Anchor parameters | `param_sweeps/spec.py` | `AnchorParameters` |
| Models and scenes | `param_sweeps/spec.py` | `ModelSpec`, `SceneSpec` |
| Sweeps | `param_sweeps/spec.py` | `SweepBase`, `StripSweep`, `GridSweep`, `GridAxis` |
| Stops and overrides | `param_sweeps/spec.py` | `SweepStop`, `ParameterOverrides`, `FeatureOverrides` |
| Swept parameters | `param_sweeps/spec.py` | `SweepParameter` |
| Presentations and subjects | `param_sweeps/spec.py` | `StripPresentation`, `SweepSubject` |
| Crop regions | `param_sweeps/spec.py` | `CropRegion` |
| Expansion into render jobs | `param_sweeps/spec.py` | `RenderJob`, `RenderStrip`, `resolved_solver_parameters`, `resolved_features` |

`tests/test_spec.py` covers the specification's shape and the rules the library must obey. For example, it checks that
clip skip runs on three models, that tiling is taught on a texture and shown failing on an object, and that each world's
descriptor block is identical wherever it appears.

## Top level

| Field | Meaning |
| --- | --- |
| `schema_version` | Always `1`. |
| `seed` | The seed every image uses, `1234567890`, except where a stop sets its own (the seed strip). |
| `anchor` | The anchor parameters. |
| `models` | The models sweeps can run on. |
| `scenes` | The scenes sweeps can run on. |
| `sweeps` | Strip and grid sweeps. |
| `batches` | Review batches. See the [review sheet reference](review-sheet.md). |
| `rulings` | Review questions. See the [review sheet reference](review-sheet.md). |

### anchor

The values every sweep starts from: `sampler_name`, `cfg_scale`, `steps`, `karras`, `denoising_strength`, `hires_fix`,
`tiling`, and optionally `scheduler`.

### models

| Field | Meaning |
| --- | --- |
| `slug` | The short name used by `--model` and in output paths. |
| `horde_name` | The model's name on the horde. |
| `baseline` | The model's baseline, such as SD1.5 or SDXL. |
| `native_width`, `native_height` | The size the model renders at: 512 for SD1.5 and 1024 for SDXL. |
| `anchor_clip_skip` | The clip skip value used when a sweep does not vary it. |
| `approximate_download_size_gb` | Shown in the fetch plan. |
| `note` | Published in the index beside the model. |

### scenes

| Field | Meaning |
| --- | --- |
| `scene_id` | The identifier used by `--scene` and in output paths. |
| `label` | The display label. |
| `prompt` | The prompt string, shared by every model in a sweep. |
| `derived_from` | Where the scene comes from. |

Scenes are written in the prompt idiom of the models that render them: comma-separated tags for the SD1.5 models, and
booru tags with a `rating_safe` token for the anime model. A world's descriptor block must appear byte for byte in every
scene that shows that subject.

## Sweeps

Every sweep has these fields:

| Field | Meaning |
| --- | --- |
| `parameter` | The swept parameter, a `SweepParameter` value. |
| `rationale` | One line on why the sweep runs on these models. Published in the index. |
| `caption` | The strip's caption. |
| `model_slugs`, `scene_ids` | The models and scenes the sweep runs on. One strip is produced per combination. |
| `source_image` | A source image, relative to the repository root. The sweep renders image-to-image from it on both runners. |
| `negative_prompt` | A negative prompt held at the sweep. Each stop turns it on or off with `use_negative_prompt`. |
| `holds` | Parameter values every stop inherits. See [Precedence](#precedence). |
| `crop` | A region to cut from every stop. See [Crop regions](#crop-regions). |
| `subject` | For ControlNet gallery strips: `anchor`, `building` or `flat`. |
| `directory_override` | The directory the sweep's strips are filed in, instead of the parameter and model directories. Keeps several sweeps of one parameter on one model and scene apart, such as the ControlNet strength strips. |

A **strip sweep** (`kind` omitted or `"strip"`) adds:

| Field | Meaning |
| --- | --- |
| `presentation` | `strip` (default), `tile_repeat_3x3` or `triptych`. |
| `resolution_multiplier` | Scales the model's native size. Default `1.0`. |
| `stops` | The stops, in order. |
| `preview` | The two stops a collapsed card compares. `left` is `default` (the stop nearest the held value), `first` or a stop value. `right` is `last` or a stop value. Unset means the held value against the last stop. |

A **grid sweep** (`kind: "grid"`) adds `rows` and `columns`, each a `GridAxis` with `parameter`, `values` and `caption`,
and a `cell_caption`. The grid is expanded row-major into stops, and its presentation is `grid`.

### Stops

A stop has a `value` and a `caption`, and optionally a `label`, a `crop` and a `control_group`. It can also set any
parameter or feature override:

- **Generation parameters:** `steps`, `cfg_scale`, `sampler_name`, `scheduler`, `clip_skip`, `hires_fix`,
  `hires_fix_denoising_strength`, `tiling`, `karras`, `denoising_strength`, `seed`, `sampler_eta`, `sampler_s_noise`,
  `sampler_s_churn`, `sampler_solver_type`, `sampler_order`, `flow_shift`, `width`, `height`.
- **Prompt:** `use_negative_prompt`, and `prompt_fragment`, which is appended to the scene's prompt. The prompt-emphasis
  strips use `prompt_fragment` to vary `(a phrase:weight)` while every other parameter stays the same. The fragment is
  published per stop in the index.
- **Features:** `control_type`, `control_strength`, `image_is_control`, `return_control_map`, `source_processing`,
  `workflow` (`qr_code`), `extra_texts`, `extra_source_images`, `source_image`, `source_mask`, `transparent`, `loras`,
  `tis`, `post_processing`, `facefixer_strength`.
- **Candidates:** `n`, the number of candidates to render (the batch size), and `chosen`, the one-based candidate to
  publish.

### Precedence

A stop's own values take precedence over the sweep's `holds`, and `holds` take precedence over the anchor. A sweep uses
`holds` to pin something the anchor sets differently. For example, the converging-steps strips pin the sampler this way.

## Source images and masks

Paths are relative to the repository root, and the files are checked in under `inputs/`. A source image is copied into
the library, so the published tree is self-contained.

`source_processing` selects the pipeline:

| Value | Meaning |
| --- | --- |
| `img2img` | Render from the source. With a mask, noise is restricted to the masked area. |
| `inpainting` | Erase the masked area and repaint it. |
| `outpainting` | The same pipeline as `inpainting`. The caller supplies a padded canvas and a mask over the padding. |
| `remix` | Blend the source with the images in `extra_source_images`. |

In a `source_mask`, **white marks the area to repaint**. Each outpainting stop states its canvas size with `width` and
`height`, because the backends resize a source to the requested size.

## Crop regions

`crop` declares a region in fractions of the frame: `x` and `y` in `[0, 1)`, `w` and `h` in `(0, 1]`. The
`derivatives` subcommand cuts it at the frame's own pixels into `<stem>_crop.webp`, and the index reports it as a file
with its pixel box. A crop can be set on a sweep or on a single stop. `even_when_small` cuts the region even from a
frame below `MINIMUM_CROPPABLE_PIXELS` (in `param_sweeps/derivatives.py`), which is otherwise shown near its own pixels
and not cropped.

## Validation rules

`load_spec` refuses a specification that breaks these rules:

- Each model renders at its native size, except where the specification states otherwise: the resolution strip, and the
  hires sweeps.
- A hires stop must request a size where **both** dimensions exceed the baseline's training size. Below that, both
  backends silently ignore the flag.
- A stop cannot combine the hires pass with inpainting or outpainting, because the backends drop the flag in those
  modes.
- `hires_fix_denoising_strength` without the hires pass is refused, because the value is not read.
- A sweep that would publish identical images as a comparison is refused.
