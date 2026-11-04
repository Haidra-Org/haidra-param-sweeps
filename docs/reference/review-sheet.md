---
title: "Review sheet"
summary: "The batches and rulings fields of a specification, the review.html contact sheet, the choices.json file and the exported review answers."
order: 15
---

# Review sheet

A specification can include `batches`, renders grouped into sections for review, and `rulings`, sets of review
questions. `param-sweeps review` writes them as a static contact sheet, `review.html`, at the root of the output tree.
A reviewer picks candidates and answers the questions in the browser, and saves two kinds of file: `choices.json`, which
`param-sweeps choose` writes back into the specification, and one answers file per question set, which stays with the
reviewer.

The procedure is in [Review a batch](../how-to/review-batch.md).

## Code map

| Concept | File | Symbol |
| --- | --- | --- |
| Batches | `param_sweeps/spec.py` | `BatchSpec`, `BatchJob`, `BatchSection` |
| Review questions | `param_sweeps/spec.py` | `RulingSet`, `RulingItem`, `RulingChoice`, `RulingRow`, `RulingControl` |
| The added "Other" option | `param_sweeps/spec.py` | `RULING_OTHER_OPTION_ID`, `RULING_OTHER_OPTION_LABEL` |
| Contact sheet | `param_sweeps/review.py` | `build_contact_sheet`, `write_contact_sheet`, `CONTACT_SHEET_FILENAME` |
| Applying picks | `param_sweeps/review.py` | `apply_choices`, `CHOICES_FILENAME` |
| Answers file name | `param_sweeps/review.py` | `RULINGS_FILENAME_SUFFIX` |

`tests/test_batch.py` covers batches, review questions and the contact sheet, against the specifications in
`tests/fixtures/`.

## Batches

`batches` is a list of sections. Each has a `section`, a `title`, a `rationale`, optional `holds`, and a list of `jobs`.

| `section` value | Contents |
| --- | --- |
| `a-world-scenes` | World scenes. |
| `b-dialects` | Dialect comparison. |
| `c-feature-probes` | Feature probes. |
| `d-suspect-claims` | Renders that check a claim that has not yet been confirmed by eye. |
| `e-reference-images` | New reference images, re-rendered from their own parameters for review. |
| `f-raw-backend` | Images produced by driving the backend directly, for claims that the normal rendering path changes. |

A job has a `job_id`, a `model_slug`, a `prompt`, a `caption`, and optionally a `negative_prompt`, a `source_image` and
`requires_source_from`. It accepts every stop override from the [sweep specification](sweep-specification.md#stops),
including `n` (candidates to render) and `chosen` (the one-based candidate to publish). `requires_source_from` names the
job whose chosen render is used as this job's source image.

## Review questions

`rulings` is a list of question sets. The `rulings` field and the `RulingSet` type keep those names because they are
part of the specification format. Each set is shown as a panel above the images.

Answers are stored under the set's id, so a follow-up review can add a shorter set with only the questions the first
one left open. The new set starts empty, and decisions already made are not reopened.

```jsonc
"rulings": [
  {
    "ruling_set_id": "schedule-review",        // the key saved answers are stored under
    "title": "Questions for this review",
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

### Question set fields

| Field | Meaning |
| --- | --- |
| `ruling_set_id` | The set's identifier. Saved answers and the exported file are named after it. |
| `title`, `intro` | The panel heading and an optional introduction. |
| `items` | The questions. |

### Question fields

| Field | Meaning |
| --- | --- |
| `ruling_id`, `title`, `question`, `note` | The question's identifier, heading, text and an optional note. |
| `control` | How it is answered: `radio` (one option), `checkbox-set` (a checklist), `row-select` (one answer per row) or `text` (free text). |
| `evidence_sections` | Batch sections to link to. |
| `evidence_job_ids` | Jobs to link to. |
| `options` | The choices, each with `choice_id`, `label` and an optional `evidence_job_id`, which adds a **view** link beside that option. |
| `rows` | For `row-select`: each row has `row_id`, `label`, an optional `evidence_job_id`, and optional `options` of its own. Rows without options use the question's `options`. |
| `master_toggle_label` | For `checkbox-set`: a label for a control that ticks every box at once. |

Every question that offers options also offers **Other / see comments**, and every question has a comment box, so an
answer outside the option set is still recorded. Every cited section and job id is checked against the batches when the
specification loads, so a link without a target fails the load instead of misleading a reviewer.

## The contact sheet

`review.html` is a static page and is opened straight from disk. It renders fully without JavaScript.

- A job that rendered several candidates shows all of them, with the current pick outlined.
- Picks and answers are kept in the browser's local storage until saved, so a review can span several sittings. Picks
  are stored under `param-sweeps-review-choices`. Answers are stored under `param-sweeps-rulings:<ruling_set_id>`, so
  the answers of two sets on one sheet, or of two review rounds in one browser, do not collide.
- Each question panel shows how many of its questions are answered (`N of M ruled`).

## choices.json

**Save choices** downloads `choices.json`. `param-sweeps choose` validates it against the specification, updates only
the `chosen` fields, and rewrites the sheet so its outlines match. Editing `"chosen": <one-based candidate>` by hand in
the specification has the same effect.

## Exported answers

**Export rulings** on a panel downloads `<ruling_set_id>_rulings.json` and shows the same text in a box to copy from:

```jsonc
{
  "schema_version": 1,
  "ruling_set_id": "schedule-review",
  "exported_at": "2026-08-03T19:11:43.333Z",
  "answered_count": 4,
  "total_count": 11,
  "unanswered": ["karras-at-low-steps"],
  "rulings": {
    "dpmpp-3m-sde-normal": { "control": "radio", "answer": "confirmed", "comment": "" },
    "golden-approvals": { "control": "checkbox-set", "selected": ["seeds_2"], "comment": "" },
    "schedule-character-claims": { "control": "row-select", "rows": { "beta": "accurate" }, "comment": "" }
  }
}
```

Answers are not uploaded or written back into the specification. An answer is a decision for a person to act
on. The tool does not change any field because of it.
