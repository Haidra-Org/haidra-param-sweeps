# SPDX-FileCopyrightText: 2026 Tazlin
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Render a contact sheet of what a batch produced, for a human to sign off before anything ships.

The sheet is one static HTML file written into the output tree, so every image reference is the
path the index already states and opening the file in a browser needs no server, no build step
and no network. Every image, caption and ruling renders with nothing running: the scripts in the
page are enhancements. One lets the reviewer click a curation batch's candidates to pick them and
save the picks as a small choices file, which [`apply_choices`][param_sweeps.review.apply_choices]
writes back into the specification. The other keeps the rulings panel's answers across reloads and
serialises them for export. A reviewer without the scripts still sees every image, every question
and every option; they just answer on paper.

Every cell states the payload beneath the image, because the question being answered is not "is
this pretty" but "is this what we asked for". A curation batch shows all of its candidates with
the chosen one marked, which is the only way the choice can be reviewed rather than trusted.

When the specification carries a [`RulingSet`][param_sweeps.spec.RulingSet], the sheet leads with
it: each question, the control it is answered with, and a link to the images that decide it. The
panel counts how many rulings are still open, so a wave cannot ship on the assumption that silence
about a question meant assent to it.

Critical public members:

- [`build_contact_sheet`][param_sweeps.review.build_contact_sheet]: the whole page, as a string.
- [`write_contact_sheet`][param_sweeps.review.write_contact_sheet]: write it into the output tree.
- [`apply_choices`][param_sweeps.review.apply_choices]: write a saved choices file into the spec.
- [`section_anchor`][param_sweeps.review.section_anchor] and
  [`job_anchor`][param_sweeps.review.job_anchor]: where a ruling's evidence link lands.
"""

from __future__ import annotations

import html
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from param_sweeps.spec import (
    BatchSection,
    RenderJob,
    RenderStrip,
    RulingControl,
    RulingItem,
    RulingSet,
    SweepSpec,
    load_spec,
)

logger = logging.getLogger(__name__)

CONTACT_SHEET_FILENAME = "review.html"
"""The sheet's name, at the root of the output tree so its image paths are the index's paths."""

CHOICES_SCHEMA_VERSION = 1
"""The choices-file layout this module writes from the sheet and reads back into the spec."""

CHOICES_FILENAME = "signoff_choices.json"
"""What the sheet's save button names its download, and what ``choose`` expects to be handed."""

SECTION_TITLES: dict[BatchSection, str] = {
    BatchSection.WORLD_SCENES: "A. World core scenes",
    BatchSection.DIALECT_COMPARISON: "B. Dialect comparison",
    BatchSection.FEATURE_PROBES: "C. Feature feasibility probes",
    BatchSection.SUSPECT_CLAIMS: "D. Claims awaiting a visual ruling",
    BatchSection.MINTED_GOLDENS: "E. Reference images minted this wave",
    BatchSection.RAW_BACKEND: "F. Raw-backend renders (no coercion in the path)",
}
"""What each batch section is called on the sheet, in the order the batch was written in."""

SWEEP_SECTION_TITLE = "Sweeps"
"""The heading the strips that are not part of any batch are collected under."""

RULINGS_SCHEMA_VERSION = 1
"""The layout of the rulings file the sheet's export button produces."""

RULINGS_FILENAME_SUFFIX = "_rulings.json"
"""What the export button appends to a set's identifier to name its download.

Named after the set rather than the sheet, because a sheet can carry more than one set and two
downloads sharing a name would leave the reviewer to work out which was which.
"""

RULINGS_DATA_ATTRIBUTE = "data-rulings-descriptor"
"""What marks the element a panel's descriptor is embedded in, which the script reads its work from.

An attribute rather than an id, because a sheet carries one panel per ruling set and an id would
have to be made unique across them for no gain: each panel finds its own descriptor within itself.
"""

RULINGS_HELP = (
    "A ruling counts as made once it carries a pick: one option for a single-answer question,"
    " at least one box or a comment for a checklist, and every row answered for a per-row question."
    " Answers are kept in this browser and are exported as a file, never sent anywhere."
)
"""What the panel says about when a question stops counting as open, stated rather than guessed."""

STYLE_SHEET = """
:root { color-scheme: light dark; --gap: 1rem; }
body {
  margin: 0; padding: var(--gap);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
  background: Canvas; color: CanvasText;
}
h1 { font-size: 1.4rem; margin: 0 0 .25rem; }
h2 { font-size: 1.1rem; margin: 2rem 0 .5rem; padding-bottom: .3rem; border-bottom: 1px solid; }
.meta { opacity: .7; font-size: .85rem; margin: 0 0 1rem; }
.grid { display: grid; gap: var(--gap); grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); }
figure { margin: 0; display: flex; flex-direction: column; gap: .4rem; }
figure img {
  width: 100%; height: auto; display: block; border-radius: 6px;
  background: repeating-conic-gradient(#8884 0% 25%, transparent 0% 50%) 50% / 20px 20px;
}
figcaption { font-size: .8rem; }
figcaption .title { font-weight: 600; }
figcaption dl { display: grid; grid-template-columns: auto 1fr; gap: 0 .5rem; margin: .3rem 0 0; }
figcaption dt { opacity: .6; }
figcaption dd { margin: 0; overflow-wrap: anywhere; }
.prompt { margin: .3rem 0 0; opacity: .8; overflow-wrap: anywhere; }
.candidates { display: grid; gap: .4rem; grid-template-columns: repeat(auto-fill, minmax(110px, 1fr)); }
.candidate { margin: 0; position: relative; }
.candidate figcaption { text-align: center; opacity: .6; }
.candidate.chosen figcaption { opacity: 1; font-weight: 600; }
.candidate.chosen img { outline: 3px solid Highlight; outline-offset: 2px; }
.missing {
  display: grid; place-items: center; aspect-ratio: 1; border: 1px dashed; border-radius: 6px;
  opacity: .5; font-size: .8rem;
}
.candidate[data-choice-key] { cursor: pointer; }
.candidate[data-choice-key]:focus-visible { outline: none; }
.candidate[data-choice-key]:focus-visible img { outline: 3px dashed Highlight; outline-offset: 2px; }
.choices-bar {
  position: fixed; right: var(--gap); bottom: var(--gap); z-index: 1;
  display: flex; align-items: center; gap: .6rem; padding: .5rem .75rem;
  background: Canvas; border: 1px solid; border-radius: 8px; box-shadow: 0 2px 8px #0004;
  font-size: .85rem;
}
.choices-bar button { font: inherit; padding: .3rem .7rem; }
.rulings { border: 1px solid; border-radius: 8px; padding: .1rem 1rem 1rem; margin: 1.5rem 0 2rem; }
.rulings h2 { border-bottom: none; margin-bottom: 0; }
.rulings-intro { margin: .2rem 0 .6rem; opacity: .8; }
.rulings-help { margin: .4rem 0 0; opacity: .7; font-size: .8rem; }
.rulings-bar {
  position: sticky; top: 0; z-index: 1; display: flex; align-items: center; gap: .6rem;
  padding: .5rem 0; background: Canvas; border-bottom: 1px solid;
}
.rulings-bar button { font: inherit; padding: .3rem .7rem; }
.rulings-count { font-weight: 600; }
.rulings-count.complete { color: green; }
.rulings-list { margin: 1rem 0 0; padding-left: 1.4rem; display: flex; flex-direction: column; gap: 1.2rem; }
.ruling { border-left: 3px solid transparent; padding-left: .8rem; }
.ruling.unruled { border-left-color: #d9822b; }
.ruling h3 { font-size: 1rem; margin: 0 0 .25rem; }
.ruling .question { margin: 0 0 .4rem; }
.ruling .note { margin: 0 0 .4rem; opacity: .75; font-size: .85rem; }
.ruling .evidence { margin: 0 0 .5rem; font-size: .85rem; }
.ruling .choices-list { display: flex; flex-direction: column; gap: .2rem; margin: 0 0 .5rem; }
.ruling .choice { display: flex; align-items: baseline; gap: .4rem; }
.ruling .choice.master { font-weight: 600; margin-bottom: .3rem; }
.ruling .ruling-row { display: grid; grid-template-columns: 12rem auto auto; gap: .4rem; align-items: baseline; }
.ruling .ruling-comment { display: block; font-size: .85rem; opacity: .8; }
.ruling textarea, .rulings-export-panel textarea {
  display: block; width: 100%; box-sizing: border-box; font: inherit; font-size: .85rem; margin-top: .2rem;
}
.rulings-export-panel { margin-top: 1rem; }
.evidence-link { font-size: .8rem; }
"""

CHOICES_SCRIPT = """
(() => {
  "use strict";
  const STORAGE_KEY = "param-sweeps-signoff-choices";
  const cells = Array.from(document.querySelectorAll(".candidate[data-choice-key]"));
  if (cells.length === 0) return;

  const groups = new Map();
  for (const cell of cells) {
    const key = cell.dataset.choiceKey;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(cell);
  }

  const baseline = new Map();
  const picks = new Map();
  for (const [key, members] of groups) {
    const initial = members.find((cell) => cell.classList.contains("chosen"));
    const value = initial ? Number(initial.dataset.candidate) : 1;
    baseline.set(key, value);
    picks.set(key, value);
  }

  let stored = null;
  try { stored = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null"); } catch { stored = null; }
  if (stored && typeof stored === "object") {
    for (const [key, value] of Object.entries(stored)) {
      const members = groups.get(key);
      if (members && members.some((cell) => Number(cell.dataset.candidate) === value)) {
        picks.set(key, value);
      }
    }
  }

  const bar = document.createElement("div");
  bar.className = "choices-bar";
  const status = document.createElement("span");
  const saveButton = document.createElement("button");
  saveButton.type = "button";
  saveButton.textContent = "Save choices";
  const resetButton = document.createElement("button");
  resetButton.type = "button";
  resetButton.textContent = "Reset";
  bar.append(status, saveButton, resetButton);
  document.body.append(bar);

  const persist = () => {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(Object.fromEntries(picks))); } catch {}
  };

  const render = () => {
    for (const [key, members] of groups) {
      for (const cell of members) {
        const selected = Number(cell.dataset.candidate) === picks.get(key);
        cell.classList.toggle("chosen", selected);
        cell.setAttribute("aria-checked", selected ? "true" : "false");
      }
    }
    let changed = 0;
    for (const [key, value] of picks) if (baseline.get(key) !== value) changed += 1;
    status.textContent = changed === 0
      ? "Click a candidate to pick it"
      : changed + " pick(s) differ from the specification";
  };

  for (const [key, members] of groups) {
    members[0].parentElement.setAttribute("role", "radiogroup");
    for (const cell of members) {
      cell.setAttribute("role", "radio");
      cell.setAttribute("tabindex", "0");
      cell.setAttribute("aria-label", "candidate " + cell.dataset.candidate);
      const select = () => { picks.set(key, Number(cell.dataset.candidate)); persist(); render(); };
      cell.addEventListener("click", select);
      cell.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") { event.preventDefault(); select(); }
      });
    }
  }

  saveButton.addEventListener("click", () => {
    const payload = { schema_version: 1, choices: Object.fromEntries(picks) };
    const blob = new Blob([JSON.stringify(payload, null, 2) + "\\n"], { type: "application/json" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "signoff_choices.json";
    link.click();
    URL.revokeObjectURL(link.href);
  });

  resetButton.addEventListener("click", () => {
    try { localStorage.removeItem(STORAGE_KEY); } catch {}
    for (const [key, value] of baseline) picks.set(key, value);
    render();
  });

  render();
})();
"""
"""The enhancement: pick candidates by click or keyboard, and save the picks as a choices file.

Picks survive a reload through ``localStorage``, so a review can be done in sittings. The saved
file's keys are each job's first-candidate path, the same key resumption uses, because it is the
one identifier that never moves when a choice changes.
"""

RULINGS_SCRIPT = """
document.querySelectorAll(".rulings").forEach((panel) => {
  "use strict";
  const dataElement = panel.querySelector("script[data-rulings-descriptor]");
  if (!dataElement) return;

  let descriptor = null;
  try { descriptor = JSON.parse(dataElement.textContent); } catch { return; }
  if (!descriptor || !Array.isArray(descriptor.items)) return;

  const storageKey = "param-sweeps-rulings:" + descriptor.ruling_set_id;
  const total = descriptor.items.length;
  const cards = new Map();
  for (const item of descriptor.items) {
    const card = panel.querySelector('.ruling[data-ruling="' + item.ruling_id + '"]');
    if (card) cards.set(item.ruling_id, { item, card });
  }

  const commentOf = (card) => {
    const box = card.querySelector("textarea[data-comment]");
    return box ? box.value.trim() : "";
  };

  const readAnswer = (item, card) => {
    const comment = commentOf(card);
    if (item.control === "checkbox-set") {
      const boxes = card.querySelectorAll("input[type=checkbox][value]:checked");
      return { control: item.control, selected: Array.from(boxes, (box) => box.value), comment };
    }
    if (item.control === "row-select") {
      const rows = {};
      for (const select of card.querySelectorAll("select[data-row]")) {
        rows[select.dataset.row] = select.value === "" ? null : select.value;
      }
      return { control: item.control, rows, comment };
    }
    if (item.control === "text") {
      const box = card.querySelector("textarea[data-answer-text]");
      const written = box ? box.value.trim() : "";
      return { control: item.control, answer: written === "" ? null : written, comment };
    }
    const picked = card.querySelector("input[type=radio]:checked");
    return { control: item.control, answer: picked ? picked.value : null, comment };
  };

  const isRuled = (answer) => {
    if (answer.control === "checkbox-set") return answer.selected.length > 0 || answer.comment !== "";
    if (answer.control === "row-select") return Object.values(answer.rows).every((value) => value !== null);
    return answer.answer !== null;
  };

  const restore = (item, card, saved) => {
    const commentBox = card.querySelector("textarea[data-comment]");
    if (commentBox && typeof saved.comment === "string") commentBox.value = saved.comment;

    if (item.control === "checkbox-set" && Array.isArray(saved.selected)) {
      for (const box of card.querySelectorAll("input[type=checkbox][value]")) {
        box.checked = saved.selected.includes(box.value);
      }
    } else if (item.control === "row-select" && saved.rows && typeof saved.rows === "object") {
      for (const select of card.querySelectorAll("select[data-row]")) {
        const value = saved.rows[select.dataset.row];
        if (Array.from(select.options).some((option) => option.value === value)) select.value = value;
      }
    } else if (item.control === "text") {
      const box = card.querySelector("textarea[data-answer-text]");
      if (box && typeof saved.answer === "string") box.value = saved.answer;
    } else if (typeof saved.answer === "string") {
      for (const input of card.querySelectorAll("input[type=radio]")) {
        if (input.value === saved.answer) input.checked = true;
      }
    }
  };

  let stored = null;
  try { stored = JSON.parse(localStorage.getItem(storageKey) || "null"); } catch { stored = null; }
  if (stored && typeof stored === "object") {
    for (const [rulingId, saved] of Object.entries(stored)) {
      const entry = cards.get(rulingId);
      if (entry && saved && typeof saved === "object") restore(entry.item, entry.card, saved);
    }
  }

  // The escape option is not part of what a master toggle approves, and never counts towards it.
  const listedBoxes = (card) => card.querySelectorAll('input[type=checkbox][value]:not([value="other-see-comments"])');

  const countLabel = panel.querySelector(".rulings-count");
  const refresh = () => {
    const answers = {};
    const unanswered = [];
    for (const [rulingId, entry] of cards) {
      const answer = readAnswer(entry.item, entry.card);
      answers[rulingId] = answer;
      const ruled = isRuled(answer);
      if (!ruled) unanswered.push(rulingId);
      entry.card.classList.toggle("unruled", !ruled);
    }
    try { localStorage.setItem(storageKey, JSON.stringify(answers)); } catch {}
    countLabel.textContent = (total - unanswered.length) + " of " + total + " ruled";
    countLabel.classList.toggle("complete", unanswered.length === 0);
    for (const master of panel.querySelectorAll("input[data-master]")) {
      const boxes = Array.from(listedBoxes(master.closest(".ruling")));
      master.checked = boxes.every((box) => box.checked);
    }
    return { answers, unanswered };
  };

  for (const master of panel.querySelectorAll("input[data-master]")) {
    master.addEventListener("change", () => {
      for (const box of listedBoxes(master.closest(".ruling"))) box.checked = master.checked;
    });
  }

  panel.addEventListener("input", refresh);
  panel.addEventListener("change", refresh);

  const exportPanel = panel.querySelector(".rulings-export-panel");
  const exportText = panel.querySelector(".rulings-export-text");
  panel.querySelector(".rulings-export").addEventListener("click", () => {
    const { answers, unanswered } = refresh();
    const payload = {
      schema_version: 1,
      ruling_set_id: descriptor.ruling_set_id,
      exported_at: new Date().toISOString(),
      answered_count: total - unanswered.length,
      total_count: total,
      unanswered: unanswered,
      rulings: answers,
    };
    const text = JSON.stringify(payload, null, 2) + "\\n";
    exportText.value = text;
    exportPanel.hidden = false;
    exportText.focus();
    exportText.select();
    const blob = new Blob([text], { type: "application/json" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = descriptor.ruling_set_id + "_rulings.json";
    link.click();
    URL.revokeObjectURL(link.href);
  });

  panel.querySelector(".rulings-reset").addEventListener("click", () => {
    if (!window.confirm("Clear every answer on this panel?")) return;
    for (const { card } of cards.values()) {
      for (const input of card.querySelectorAll("input[type=radio], input[type=checkbox]")) input.checked = false;
      for (const select of card.querySelectorAll("select[data-row]")) select.value = "";
      for (const box of card.querySelectorAll("textarea")) box.value = "";
    }
    try { localStorage.removeItem(storageKey); } catch {}
    exportPanel.hidden = true;
    refresh();
  });

  refresh();
});
"""
"""The enhancement on the rulings panels: answers that survive a reload, a count of what is still
open, and an export that hands back one file plus the same text to copy out of the page.

Each panel is driven independently from the descriptor embedded inside it, and answers are keyed
by the ruling set's own identifier, so neither two sets on one sheet nor two waves reviewed in one
browser can inherit each other's answers.
"""


def build_contact_sheet(*, spec: SweepSpec, output_root: Path, title: str) -> str:
    """Return the whole contact sheet for this specification's outputs, as one HTML document.

    Args:
        spec: The specification the outputs were rendered from.
        output_root: The root of the output tree, which the sheet is written into and whose
            layout every image reference is relative to.
        title: The heading the sheet carries.

    Returns:
        The document.
    """
    sections = _grouped_strips(spec)
    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    rendered_count = sum(
        1
        for strips in sections.values()
        for strip in strips
        for job in strip.jobs
        for candidate in job.candidate_numbers
        if (output_root / job.relative_candidate_image_path(candidate)).is_file()
    )

    body = [
        f"<h1>{html.escape(title)}</h1>",
        f'<p class="meta">{rendered_count} image(s) on disk &middot; generated {generated_at}</p>',
    ]

    body.extend(_rulings_panel(ruling_set) for ruling_set in spec.rulings)

    for section, strips in sections.items():
        section_title = SECTION_TITLES[section] if section is not None else SWEEP_SECTION_TITLE
        body.append(f'<h2 id="{section_anchor(section)}">{html.escape(section_title)}</h2>')
        body.append('<div class="grid">')
        body.extend(_strip_figure(strip=strip, output_root=output_root) for strip in strips)
        body.append("</div>")

    return _document(title=title, body="\n".join(body))


def section_anchor(section: BatchSection | None) -> str:
    """Return the element id a section's heading carries, which a ruling's evidence link lands on."""
    return f"section-{section}" if section is not None else "section-sweeps"


def job_anchor(job_id: str) -> str:
    """Return the element id a batch job's figure carries, which a ruling's evidence link lands on."""
    return f"job-{job_id}"


def write_contact_sheet(*, spec: SweepSpec, output_root: Path, title: str) -> Path:
    """Write the contact sheet into the output tree, and return its path.

    Side Effects:
        Overwrites any existing sheet.
    """
    sheet_path = output_root / CONTACT_SHEET_FILENAME
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    sheet_path.write_text(build_contact_sheet(spec=spec, output_root=output_root, title=title), encoding="utf-8")
    logger.info("Wrote the contact sheet: path=%s", sheet_path)
    return sheet_path


def _grouped_strips(spec: SweepSpec) -> dict[BatchSection | None, list[RenderStrip]]:
    """Return the specification's strips grouped under their section, in batch order.

    Sections with nothing in them are left out rather than shown empty, so a sheet for one
    section of a batch is a sheet for that section. The strips belonging to no batch come last,
    under the ``None`` key.
    """
    grouped: dict[BatchSection | None, list[RenderStrip]] = {}

    for section in SECTION_TITLES:
        section_strips = [strip for strip in spec.expand_strips() if strip.batch_section is section]
        if section_strips:
            grouped[section] = section_strips

    sweep_strips = [strip for strip in spec.expand_strips() if strip.batch_section is None]
    if sweep_strips:
        grouped[None] = sweep_strips

    return grouped


def _rulings_panel(ruling_set: RulingSet) -> str:
    """Return the rulings panel: every question this wave has to answer, and how to answer it."""
    parts: list[str] = [
        f'<section class="rulings" data-ruling-set="{html.escape(ruling_set.ruling_set_id, quote=True)}">',
        f"<h2>{html.escape(ruling_set.title)}</h2>",
    ]
    if ruling_set.intro:
        parts.append(f'<p class="rulings-intro">{html.escape(ruling_set.intro)}</p>')

    parts.append(
        '<div class="rulings-bar">'
        f'<span class="rulings-count">0 of {len(ruling_set.items)} ruled</span>'
        '<button type="button" class="rulings-export">Export rulings</button>'
        '<button type="button" class="rulings-reset">Clear answers</button>'
        "</div>",
    )
    parts.append(f'<p class="rulings-help">{html.escape(RULINGS_HELP)}</p>')

    parts.append('<ol class="rulings-list">')
    parts.extend(_ruling_item(item=item) for item in ruling_set.items)
    parts.append("</ol>")

    parts.append(
        '<div class="rulings-export-panel" hidden>'
        "<label>Exported rulings, to copy out of the page"
        '<textarea class="rulings-export-text" rows="12" readonly></textarea>'
        "</label></div>",
    )
    parts.append(_rulings_descriptor(ruling_set))
    parts.append("</section>")
    return "\n".join(parts)


def _ruling_item(*, item: RulingItem) -> str:
    """Return one ruling: what is being asked, what decides it, and what it is answered with."""
    ruling_id = html.escape(item.ruling_id, quote=True)
    parts: list[str] = [f'<li class="ruling" id="ruling-{ruling_id}" data-ruling="{ruling_id}">']
    parts.append(f"<h3>{html.escape(item.title)}</h3>")
    parts.append(f'<p class="question">{html.escape(item.question)}</p>')

    if item.note:
        parts.append(f'<p class="note">{html.escape(item.note)}</p>')

    evidence = _evidence_links(item)
    if evidence:
        parts.append(f'<p class="evidence">Evidence: {evidence}</p>')

    parts.append(_ruling_controls(item))
    parts.append(
        '<label class="ruling-comment">Comments'
        f'<textarea data-comment="{ruling_id}" rows="2"'
        ' placeholder="anything the options above cannot carry"></textarea></label>',
    )
    parts.append("</li>")
    return "\n".join(parts)


def _ruling_controls(item: RulingItem) -> str:
    """Return the form control a ruling is answered with, matching what its data asks for."""
    ruling_id = html.escape(item.ruling_id, quote=True)

    if item.control is RulingControl.TEXT:
        return f'<textarea data-answer-text="{ruling_id}" rows="3"></textarea>'

    if item.control is RulingControl.ROW_SELECT:
        rows: list[str] = []
        for row in item.rows:
            options = "".join(
                f'<option value="{html.escape(option.choice_id, quote=True)}">{html.escape(option.label)}</option>'
                for option in item.presented_options(row=row)
            )
            rows.append(
                '<label class="ruling-row">'
                f'<span class="row-label">{html.escape(row.label)}</span>'
                f'<select data-row="{html.escape(row.row_id, quote=True)}">'
                f'<option value="">not ruled</option>{options}</select>'
                f"{_evidence_link(row.evidence_job_id)}</label>",
            )
        return f'<div class="choices-list">{"".join(rows)}</div>'

    input_type = "checkbox" if item.control is RulingControl.CHECKBOX_SET else "radio"
    boxes: list[str] = []
    if item.master_toggle_label is not None:
        boxes.append(
            f'<label class="choice master"><input type="checkbox" data-master="{ruling_id}">'
            f"<span>{html.escape(item.master_toggle_label)}</span></label>",
        )
    for option in item.presented_options():
        choice_id = html.escape(option.choice_id, quote=True)
        boxes.append(
            f'<label class="choice"><input type="{input_type}" name="ruling-{ruling_id}"'
            f' id="ruling-{ruling_id}-{choice_id}" value="{choice_id}">'
            f"<span>{html.escape(option.label)}</span>{_evidence_link(option.evidence_job_id)}</label>",
        )
    return f'<div class="choices-list">{"".join(boxes)}</div>'


def _evidence_links(item: RulingItem) -> str:
    """Return the links from a ruling to the sections and images that decide it."""
    links = [
        f'<a href="#{section_anchor(section)}">{html.escape(SECTION_TITLES[section])}</a>'
        for section in item.evidence_sections
    ]
    links.extend(f'<a href="#{job_anchor(job_id)}">{html.escape(job_id)}</a>' for job_id in item.evidence_job_ids)
    return ", ".join(links)


def _evidence_link(job_id: str | None) -> str:
    """Return the link beside one option or row, when that one choice has an image of its own."""
    if job_id is None:
        return ""
    return f'<a class="evidence-link" href="#{job_anchor(job_id)}">view</a>'


def _rulings_descriptor(ruling_set: RulingSet) -> str:
    """Return the panel's descriptor, embedded as data so the script never has to infer its work.

    It states only what reading and writing an answer needs: the set's identity, and each ruling's
    identity, control and rows in the order the export has to carry them.
    """
    payload = {
        "schema_version": RULINGS_SCHEMA_VERSION,
        "ruling_set_id": ruling_set.ruling_set_id,
        "items": [
            {
                "ruling_id": item.ruling_id,
                "control": str(item.control),
                "rows": [row.row_id for row in item.rows],
            }
            for item in ruling_set.items
        ],
    }
    # A closing tag inside the data would end the script element early, whatever it is data for.
    body = json.dumps(payload, indent=2, ensure_ascii=False).replace("</", "<\\/")
    return f'<script type="application/json" {RULINGS_DATA_ATTRIBUTE}>{body}</script>'


def _strip_figure(*, strip: RenderStrip, output_root: Path) -> str:
    """Return one strip's figure: its images and the payload they were made with."""
    anchor = ""
    if strip.batch_section is not None:
        anchor = f' id="{html.escape(job_anchor(strip.jobs[0].scene_id), quote=True)}"'

    parts: list[str] = [f'<figure class="cell"{anchor}>']

    for job in strip.jobs:
        parts.append(_images_for(job=job, output_root=output_root))

    first_job = strip.jobs[0]
    parts.append("<figcaption>")
    parts.append(f'<div class="title">{html.escape(strip.caption)}</div>')
    parts.append(_payload_list(first_job, image_model_name=strip.image_model.horde_name))
    parts.append(f'<p class="prompt">{html.escape(first_job.prompt)}</p>')
    if first_job.negative_prompt:
        parts.append(f'<p class="prompt">negative: {html.escape(first_job.negative_prompt)}</p>')
    parts.append("</figcaption>")

    parts.append("</figure>")
    return "\n".join(parts)


def _images_for(*, job: RenderJob, output_root: Path) -> str:
    """Return the image or images one job produced, with a batch's chosen candidate marked."""
    if job.batch_size == 1:
        return _image_or_placeholder(relative_path=job.relative_image_path, output_root=output_root)

    candidates: list[str] = ['<div class="candidates">']
    choice_key = html.escape(job.checkpoint_key, quote=True)
    for candidate_number in job.candidate_numbers:
        chosen = " chosen" if candidate_number == job.chosen_candidate_number else ""
        image = _image_or_placeholder(
            relative_path=job.relative_candidate_image_path(candidate_number),
            output_root=output_root,
        )
        candidates.append(
            f'<figure class="candidate{chosen}" data-choice-key="{choice_key}"'
            f' data-candidate="{candidate_number}">'
            f"{image}<figcaption>{candidate_number:02d}</figcaption></figure>",
        )
    candidates.append("</div>")
    return "\n".join(candidates)


def _image_or_placeholder(*, relative_path: str, output_root: Path) -> str:
    """Return an image tag, or a placeholder saying the image has not been rendered yet."""
    if not (output_root / relative_path).is_file():
        return f'<div class="missing">not rendered<br>{html.escape(relative_path)}</div>'

    return f'<img src="{html.escape(relative_path)}" alt="{html.escape(relative_path)}" loading="lazy">'


def _payload_list(job: RenderJob, *, image_model_name: str) -> str:
    """Return the payload caption: what a reviewer has to check the image against."""
    entries: list[tuple[str, str]] = [
        ("model", image_model_name),
        ("size", f"{job.width}&times;{job.height}"),
        ("sampler", f"{job.sampler_name}{', karras' if job.karras else ''}"),
        ("steps / cfg", f"{job.steps} / {job.cfg_scale:g}"),
        ("seed", str(job.seed)),
        ("clip skip", str(job.clip_skip)),
    ]

    features = job.features()
    if not features.is_plain:
        entries.append(("features", _feature_summary(job)))
    if job.batch_size > 1:
        entries.append(("curation", f"{job.chosen_candidate_number} of {job.batch_size}"))

    rows = "".join(f"<dt>{label}</dt><dd>{value}</dd>" for label, value in entries)
    return f"<dl>{rows}</dl>"


def _feature_summary(job: RenderJob) -> str:
    """Return a one-line summary of the features a job turned on."""
    stated: list[str] = []

    if job.control_type is not None:
        stated.append(f"control {html.escape(job.control_type)}")
    if job.return_control_map:
        stated.append("control map returned")
    if job.image_is_control:
        stated.append("source is a control map")
    if job.resolved_source_processing is not None:
        stated.append(str(job.resolved_source_processing))
    if job.workflow is not None:
        stated.append(f"workflow {job.workflow}")
    if job.transparent:
        stated.append("transparent")
    if job.source_mask is not None:
        stated.append("masked")
    if job.post_processing:
        stated.append(", ".join(html.escape(step) for step in job.post_processing))
    if job.loras:
        stated.append(f"{len(job.loras)} lora(s)")
    if job.tis:
        stated.append(f"{len(job.tis)} textual inversion(s)")
    for extra_text in job.extra_texts:
        stated.append(f"{html.escape(extra_text.reference)}={html.escape(extra_text.text)}")

    return "; ".join(stated)


def _document(*, title: str, body: str) -> str:
    """Return the whole page around a body, self-contained and needing nothing external."""
    return (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{html.escape(title)}</title>\n"
        f"<style>{STYLE_SHEET}</style>\n"
        "</head>\n"
        f"<body>\n{body}\n<script>{CHOICES_SCRIPT}</script>\n<script>{RULINGS_SCRIPT}</script>\n</body>\n"
        "</html>\n"
    )


def apply_choices(*, spec_path: Path, choices_path: Path) -> int:
    """Write the picks a reviewer saved from the sheet back into the specification file.

    The choices file is what the sheet's save button downloads: ``{"schema_version": 1,
    "choices": {<first-candidate path>: <one-based pick>}}``. Only a batch job's picks are
    addressable this way; a sweep stop that renders candidates has no unique id in its
    specification and keeps being edited by hand.

    The edit is surgical: the file is rewritten field-for-field with only ``chosen`` values
    changed, and the result is validated as a specification before anything touches disk.

    Args:
        spec_path: The specification the sheet was rendered from, which is rewritten in place.
        choices_path: The choices file the reviewer saved.

    Returns:
        How many jobs' ``chosen`` fields actually changed.

    Raises:
        ValueError: If the choices file has the wrong shape, names a job the specification does
            not curate, or picks a candidate outside that job's batch.
        OSError: If either file cannot be read, or the specification cannot be rewritten.
    """
    payload = json.loads(choices_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != CHOICES_SCHEMA_VERSION:
        raise ValueError(f"{choices_path} is not a version-{CHOICES_SCHEMA_VERSION} choices file")
    choices = payload.get("choices")
    if not isinstance(choices, dict):
        raise ValueError(f"{choices_path} carries no choices mapping")

    spec = load_spec(spec_path)
    addressable = {
        job.checkpoint_key: job
        for strip in spec.expand_strips()
        if strip.batch_section is not None
        for job in strip.jobs
        if job.batch_size > 1
    }

    picked_by_job_id: dict[str, int] = {}
    for choice_key, picked in choices.items():
        job = addressable.get(choice_key)
        if job is None:
            raise ValueError(f"choice {choice_key!r} names no curation job in {spec_path.name}")
        if not isinstance(picked, int) or isinstance(picked, bool) or not 1 <= picked <= job.batch_size:
            raise ValueError(f"choice {choice_key!r} picks {picked!r}, outside a batch of {job.batch_size}")
        picked_by_job_id[job.scene_id] = picked

    raw = json.loads(spec_path.read_text(encoding="utf-8"))
    changed_count = 0
    for batch in raw.get("batches", []):
        for job_fields in batch.get("jobs", []):
            picked = picked_by_job_id.get(job_fields.get("job_id"))
            if picked is None or job_fields.get("chosen") == picked:
                continue
            job_fields["chosen"] = picked
            changed_count += 1

    if changed_count:
        SweepSpec.model_validate(raw)
        spec_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        logger.info("Applied the reviewer's picks: changed=%d path=%s", changed_count, spec_path)

    return changed_count
