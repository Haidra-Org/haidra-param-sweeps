---
title: "Output tree"
summary: "The files param-sweeps writes under the library and promo roots: index.json, sidecars, crops, the alchemy and tutorial trees, measurements and the checkpoint."
order: 40
---

# Output tree

The library is written under the output root, `output/param-sweeps/` by default. Promotional material is written under
a separate promo root, `output/promo/` by default. `index.json` describes every strip completely, so a consumer that
has the index does not need to open a sidecar or parse a file name. The alchemy and tutorial trees beside it have their
own manifests and are read without going through `index.json`.

How the roots are chosen is in the [environment reference](environment.md). Which of these files are published is in
the [publishing reference](publishing.md).

## Code map

| Concept | File | Symbol |
| --- | --- | --- |
| Index schema | `param_sweeps/outputs.py` | `SweepIndex`, `IndexStrip`, `IndexStop` |
| Index building | `param_sweeps/outputs.py` | `build_index`, `write_index` |
| Image and sidecar writing | `param_sweeps/outputs.py` | `save_render`, `ImageSidecar` |
| Source images copied into the library | `param_sweeps/outputs.py` | `publish_source_images` |
| Crops and bounded copies | `param_sweeps/derivatives.py` | `write_sweep_derivatives`, `FIT_LONG_EDGE_PIXELS` |
| Control maps | `param_sweeps/control_maps.py` | `draw_control_maps`, `ControlMapSidecar` |
| Resume checkpoint | `param_sweeps/checkpoint.py` | `CampaignCheckpoint`, `CHECKPOINT_FILENAME` |
| Alchemy manifest | `param_sweeps/alchemy.py` | `AlchemyManifest`, `ManifestEntry`, `write_manifest` |
| Tutorial manifest | `param_sweeps/flows.py` | `TutorialManifest`, `ManifestFlow`, `FlowMeta` |
| Sampler cost report | `param_sweeps/sampler_cost.py` | `SamplerCostReport`, `write_report` |
| Card manifest | `param_sweeps/promo.py` | `CardsManifest`, `CardManifestEntry` |
| Gallery manifest | `param_sweeps/gallery.py` | `build_manifest`, `write_manifest` |

The index is covered by `tests/test_campaign.py` and `tests/test_derivatives.py`, the alchemy tree by
`tests/test_alchemy.py`, the tutorial tree by `tests/test_flows.py`, and the promo tree by `tests/test_promo.py` and
`tests/test_gallery.py`.

## Library layout

```text
output/param-sweeps/
  index.json                                   every strip, its stops and their files
  .param-sweeps-checkpoint.json                the resume checkpoint
  review.html                                  the review contact sheet, when one was written
  steps/icbinp/person/001_4.webp               a rendered stop
  steps/icbinp/person/001_4.json               its sidecar
  upscaler-form/albedobase-xl/courier/001_realesrgan_x4plus_crop.webp   the region worth seeing at full resolution
  upscaler-form/albedobase-xl/courier/001_realesrgan_x4plus_fit.webp    the same frame bounded to 1024 on its long edge
  denoising-strength/icbinp/garden/source.webp the image that strip starts from
  alchemy/manifest.json                        every alchemy form, its use case and its result
  alchemy/sources/tutorial-anchor-512.webp     a source shared by the forms in one use case
  alchemy/enlarge/realesrgan_x4plus/result.webp       one form's result
  alchemy/control_map/canny/result.webp        one detector's map of the shared source
  tutorials/manifest.json                      every tutorial worked example
  tutorials/<flow>/...                         one worked example
  measurements/<start time>.json               one sampler cost report
```

File names are deterministic. The same specification and seed always produce the same paths, so re-rendering
overwrites in place.

## index.json

`index.json` has a `schema_version`, the library `seed`, and a list of `strips`. It is rewritten while a run progresses
and only describes images that exist. A strip that is partly rendered has `"complete": false`.

### Strip fields

| Field | Meaning |
| --- | --- |
| `strip_id` | The strip's identifier. |
| `parameter` | The swept parameter, a `SweepParameter` value. |
| `presentation` | `strip`, `grid`, `tile_repeat_3x3` or `triptych`. See [Presentations](#presentations). |
| `caption`, `rationale` | The strip's caption, and why it runs on this model. |
| `complete` | Whether every stop has been rendered. |
| `image_model` | The model: `slug`, `name`, `baseline`, `note`. |
| `scene` | The scene: `scene_id`, `label`, `prompt`. |
| `held` | The generation parameters that every stop shares. |
| `varying` | The names of the fields that change between stops. Empty for the step-preview and upscaler strips. |
| `stops` | The stops in order. |
| `source_image` | The image the strip starts from, relative to the index, with `source_width` and `source_height`. |
| `subject` | For ControlNet gallery strips: `anchor`, `building` or `flat`. |
| `control_map` | For a strip held to one detector: the map drawn from its source. |
| `grid` | For grids: `rows` and `columns`, each with `parameter`, `values` and `caption`. |
| `preview` | The pair of stops to show as a preview: `left_index` and `right_index`. |
| `features` | Job features shared by every stop, where there are any. |
| `batch_section` | For review batches: the section the strip belongs to. |

### Stop fields

| Field | Meaning |
| --- | --- |
| `value`, `label`, `caption` | The stop's value, its display label and its caption. |
| `file`, `sidecar` | The image and its sidecar, relative to the index. |
| `width`, `height` | The pixel size of the frame, read from the file instead of the request. |
| `prompt_fragment` | The text appended to the scene's prompt for this stop, where there is one. |
| `batch_size`, `chosen_candidate` | For a stop with several candidates: how many were rendered and which was picked. |
| `control_type`, `control_group` | For ControlNet stops: the detector and its group. |
| `control_map` | The map drawn beside this render, where there is one. |
| `crop` | The declared region, cut at the frame's own pixels: a file and its pixel box. |
| `fit` | A copy bounded to 1024 pixels on its long edge. Only present when the frame's long edge is over 1024. |

A stop does not repeat its resolved parameters. The strip states what is held and what varies, and the sidecar beside
every image has the full set.

### Presentations

Consumers are expected to honour the presentation:

- `strip`: show the stops in order, as a scrubber.
- `grid`: lay the stops out row-major using the strip's `grid.rows` and `grid.columns`.
- `tile_repeat_3x3`: draw each image repeated nine times in a square, beside the single image.
- `triptych`: show three pictures per stop, in order: the strip's source, the stop's control map, and the render.

[One parameter per strip](../explanation/one-parameter-per-strip.md#presentations) explains why.

## Sidecars

Each image has a JSON sidecar with the same stem. It records the parameter, model slug, requested model and the model
actually used, scene, prompt, stop index and value, caption, the full resolved generation parameters and features, the
runner, start and finish times, the image file name, the candidate number and whether it was chosen, the source image
and any runner note.

## Checkpoint

`.param-sweeps-checkpoint.json` at the root records every image the moment it is written. An entry only counts while
its image is still on disk, so deleting an image is enough to have it rendered again. See
[Resume an interrupted run](../how-to/render-library.md#resume-an-interrupted-run).

## Alchemy tree

`alchemy/manifest.json` has `generated_against` and a list of `entries`. Every form placed by the alchemy specification
has an entry, whether or not it produced a result.

| Field | Meaning |
| --- | --- |
| `use_case` | `describe`, `control_map`, `enlarge`, `fix_faces` or `cut_out`. |
| `form`, `title` | The alchemy form and its display title. |
| `control_type` | For `annotation` entries: the detector. |
| `files` | `source`, and where a result exists `result` (an image), `result_text` or `result_json`, relative to `alchemy/`. |
| `model` | The model that produced the result, where known. |
| `source_width`, `source_height`, `result_width`, `result_height` | Pixel sizes. |
| `source_crop`, `result_crop` | The declared region cut from the source and the result. |
| `result_fit` | A bounded copy of a result over 1024 pixels. |
| `notes` | Why the entry has no result, or how it was placed. |

Shared sources are published once under `alchemy/sources/`. Each result directory also has a provenance `meta.json`,
which is not published. Each control map has a sidecar with the service's host and reported version.

## Tutorial tree

`flows` writes one directory per flow under `tutorials/` and rebuilds `tutorials/manifest.json` from what is on disk.
A flow directory holds:

- the result;
- the source and mask, where the flow has them;
- `reference-N.webp` for each extra image a remix blended in;
- the published request, with its image fields replaced by placeholders;
- `meta.json`.

`meta.json` records the model, the seed and requested seed, the generation id, the kudos estimate and kudos charged,
wall time, whether the result was censored, the `runner` (`horde` or `local`) and the generation metadata. A local
render is not charged and does not have kudos values.

`tutorials/manifest.json` lists each flow with its `id`, `directory`, `title`, `summary`, `status`, `files`, and where
present `model`, `seed`, `result`, `source`, `mask`, `before`, `request` and `references`. A flow marked `skipped` in the
specification is left out of the manifest.

## Measurements

`sampler-cost` writes one JSON report per run to `measurements/` under the output root, named after the run's start
time, so a new measurement never overwrites an older one. Each sampler's record includes the schedule used, every raw
timing, the fit and its R-squared, the ratio against `k_euler`, the ratio predicted by the sampler's evaluations per
step, and whether the two disagree by more than 20%.

## Promo tree

```text
output/promo/
  cards.json                    every card, by key
  sweeps/<parameter>.jpg        a set card per parameter, keyed sweep:<parameter>
  tutorials/<flow>.jpg          a set card per tutorial, keyed tutorial:<flow>
  <job output>                  a job card, at its own path
  gallery/
    gallery.json                every gallery entry
    <image entry>/              a picture and its meta.json
    text/<id>/                  a single-turn text
    conversations/<id>/         a conversation, with meta.json and transcript.txt
```

`cards.json` maps each card key to its `file`, `width`, `height`, `template` and `source`.

`gallery.json` is rebuilt from what is on disk on every run. It lists every picture, text and conversation, and lists
entries refused by the safety checker under `censored`. `curated` and `note` on an entry are written by a separate
curation step and are kept when the manifest is rebuilt. A refused entry leaves a `censored.json` with the request and
the time.

Each gallery record has its request body, reply, model, seconds and kudos. It does not record the worker, so the
published files cannot identify a volunteer. The seconds are measured
with the dev key, which has queue priority, so they show what the horde can do and not the wait an anonymous user
should expect.
