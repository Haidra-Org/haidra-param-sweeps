---
title: "Review a batch"
summary: "Write the review contact sheet for a batch specification, pick candidates, answer the review questions and write the picks back into the specification."
order: 30
---

# Review a batch

A specification with `batches` is reviewed from a contact sheet before anything is published. This guide writes the
sheet, records picks and answers, and writes the picks back. Before you start, render the batch with `local` or `horde`
and the same `--spec`. When you finish, every multi-candidate job in the specification has its `chosen` field set, and
you have an answers file for each question set.

The fields involved are in the [review sheet reference](../reference/review-sheet.md).

## 1. Write the sheet

```sh
uv run param-sweeps review --spec <batch-spec.json>
```

This writes `review.html` at the root of the output tree. `--title` changes its heading.

**Check:** `review.html` exists under the output root and opens in a browser straight from disk.

## 2. Pick candidates

A job that rendered several candidates shows all of them, with the current pick outlined. Click a candidate, or focus
it and press Enter, to pick it instead. Picks are kept in the browser's local storage until saved, so a review can span
several sittings.

When you are done, use **Save choices** to download `choices.json`.

The sheet renders fully without JavaScript. Picking from the sheet needs JavaScript. Editing
`"chosen": <one-based candidate>` in the specification by hand has the same result.

## 3. Answer the review questions

If the specification has `rulings`, each question set is shown as a panel above the images. Each question links to
the sections or images it is about, and offers its control: one option, a checklist, one answer per row, or free text.
Every question with options also offers **Other / see comments**, and every question has a comment box.

Answers are kept in local storage under the set's id. The panel shows how many questions are answered
(`N of M ruled`). A review is not complete while any question is still open.

When you are done, use **Export rulings** on each panel. It downloads `<ruling_set_id>_rulings.json` and shows the same
text in a box to copy from. The export lists any questions that are still open under `unanswered`.

Answers are not uploaded and not written into the specification. Act on them by hand.

## 4. Write the picks back

```sh
uv run param-sweeps choose --spec <batch-spec.json> <path-to>/choices.json
```

`choose` validates the file against the specification, updates only the `chosen` fields, and rewrites the sheet so its
outlines match.

**Check:** the specification's diff only changes `chosen` fields, and the refreshed sheet outlines your picks.

To undo, revert the specification with version control and run `choose` again with a corrected `choices.json`, or
edit the `chosen` fields by hand and run `review` again.
