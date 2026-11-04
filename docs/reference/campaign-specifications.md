---
title: "Campaign specifications"
summary: "The formats of flows_spec.json, alchemy_spec.json, gallery_spec.json and the promo card specification."
order: 12
---

# Campaign specifications

Besides `sweep_spec.json`, four specifications drive the other campaigns. `flows_spec.json`, `alchemy_spec.json` and
`gallery_spec.json` are checked in at the repository root and are the defaults for `flows`, `alchemy` and `gallery`.
The promo card specification is not part of this repository. Each caller supplies its own with `--promo-spec`, because
it holds the caller's branding.

## Code map

| Concept | File | Symbol |
| --- | --- | --- |
| Flow specification | `param_sweeps/flows.py` | `FlowsSpec`, `FlowSpec`, `FlowRequest`, `load_flows_spec` |
| Alchemy specification | `param_sweeps/alchemy.py` | `AlchemySpec`, `AlchemyEntrySpec`, `AlchemyUseCase`, `load_alchemy_spec` |
| Gallery specification | `param_sweeps/gallery.py` | `load_gallery_spec`, `apply_requirements`, `pick_model` |
| Promo specification | `param_sweeps/promo.py` | `PromoSpec`, `BrandSpec`, `FontSpec`, `CardJob`, `CardSet`, `load_promo_spec` |

The specifications are covered by `tests/test_flows.py`, `tests/test_alchemy.py`, `tests/test_gallery.py` and
`tests/test_promo.py`.

## Flow specification

`flows_spec.json` holds one committed request per tutorial technique, in each of the three worlds where the technique
can be shown. The world variants have a `--w2` or `--w3` suffix on their id. Top-level fields are `schema_version`,
`generated_against` and `flows`.

| Field | Meaning |
| --- | --- |
| `flow_id` | The flow's identifier, used by `--flow` and as its directory name. |
| `title`, `summary` | Shown in the tutorial manifest. |
| `request` | The request: `model`, `params`, `features`, and optional `source_image` and `source_mask`. |
| `before_request` | The other half of a pair, for flows whose lesson is the difference between two renders. Its result is saved as the flow's "before" image. |
| `status` | `available` (the default) or `skipped`. |
| `skipped_reason` | Why a skipped flow cannot be shown in this world. |
| `source_note` | A note about the source image. |
| `notes` | What the published recipe is and why. |
| `needs_attention` | What is still a compromise, what was tried, and what to vary next. `flows --dry-run` logs it. |

A skipped flow is left out of the manifest. Consumers show the first world's example in its place.

## Alchemy specification

`alchemy_spec.json` places every form named by `horde_sdk`'s `KNOWN_ALCHEMY_TYPES` into exactly one use case, and the
`annotation` form once per control type named by `KNOWN_ANNOTATION_CONTROL_TYPES`. A form or control type known to the
SDK but missing from the specification fails the load. Top-level fields are `schema_version`, `generated_against` and
`entries`.

| Field | Meaning |
| --- | --- |
| `form` | The alchemy form. |
| `use_case` | `describe`, `control_map`, `enlarge`, `fix_faces` or `cut_out`. |
| `title` | The display title. |
| `source` | The source image, shared by every entry in the use case. |
| `control_type` | For `annotation` entries: the detector. |
| `crop` | A region in fractions of the frame, cut from both the source and the result. Same format as a sweep `crop`. |
| `notes` | Published in the manifest. |

## Gallery specification

`gallery_spec.json` is plain JSON with these top-level fields:

| Field | Meaning |
| --- | --- |
| `api` | The API root, `https://aihorde.net/api/v2`. |
| `client_agent` | The client agent sent with every request. |
| `defaults` | Default generation parameters for each model family (`sdxl`, `sd15`, `flux`, `lightning`, `cascade`). |
| `families` | The preferred models of each family. |
| `images` | Picture entries. |
| `texts` | Single-turn text entries. |
| `conversations` | Conversation entries. |

- An **image** entry has an `id`, a `prompt`, a `family`, its preferred `models`, a fixed `seed`, the `alchemy` forms to
  apply to the result, and `tags`. Consumers query the gallery by tag.
- A **text** entry has an `id`, a `prompt` (one instruction) and a `max_length`.
- A **conversation** entry has an `id`, a `format` of `instruct` or `raw`, a list of `turns` and a `max_length`.

Before an image request is sent, steps, cfg scale, sampler and clip skip are brought within the limits that the model
reference at `models.aihorde.net` states for the chosen model. For example, a Flux entry with 25 steps at cfg 7
is submitted at 8 steps and cfg 4. The request saved beside the picture is the one that was actually sent.

## Promo specification

The promo specification describes the brand and the cards to draw. Font and logo paths are relative to the
specification file.

| Field | Meaning |
| --- | --- |
| `brand` | `name`, `url`, `tagline` (default "Same prompt. Same seed.") and an optional `logo`. |
| `fonts` | Three faces: `regular`, `bold` and `mono`. Without it, cards use Pillow's bundled face, which is what the tests use. |
| `palettes` | Colours for the `dark` and `light` themes. Has defaults. |
| `titles`, `subtitles` | The words each parameter is known by on the cards. |
| `preferred_strips` | For a parameter published as several strips, the strip to use. |
| `skip_parameters` | Parameters to leave out of a set, such as the grids, which are not a picture of one setting. |
| `pair_labels` | The labels of a `pair` card. Default `Before` and `After`. |
| `jobs` | Individual cards. |
| `sets` | Groups of cards, one per parameter or per tutorial. |

### Sets

A set draws one card per parameter (`every: "parameter"`, written to `sweeps/<parameter>.jpg` with key
`sweep:<parameter>`) or per tutorial with a result (`every: "tutorial"`, written to `tutorials/<flow>.jpg` with key
`tutorial:<flow>`). For a parameter published as several strips, the set uses the first complete strip in index order
unless `preferred_strips` names another. A set also takes `template`, `format`, `theme`, `stops` and `file_type`, and
optionally `directory` (default `sweeps` or `tutorials`) and `key_prefix` (default `sweep:` or `tutorial:`).

### Jobs

A job draws one card. It has an `id` and names its content with one of `strip`, `parameter`, `tutorial` or `gallery`. A
`gallery` job draws a promo gallery entry, from `gallery/<id>/result.webp` under the promo root.

| Field | Values |
| --- | --- |
| `template` | `strip`, `pair`, `grid` or `hero`. |
| `format` | `og` (1200×630, the default), `square`, `wide`, `story` or `poster` (2048×2048). |
| `theme` | `dark` (the default) or `light`. |
| `stops` | `all`, `preview` (the strip's own preview pair), `spread:N`, `first:N`, or a list of indices. Default `spread:6`. |
| `title`, `subtitle`, `link` | Optional text. |
| `tagline` | Replaces the brand's tagline on this card. |
| `file_type` | `jpg` (the default) or `png`. |
| `output` | The card's path under the promo root. |

A `hero` card sets its text over the picture, so each letter is drawn with a dark outline and glow. Its subtitle can
run to two lines before it is cut.
