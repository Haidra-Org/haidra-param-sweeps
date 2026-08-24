# SPDX-FileCopyrightText: 2026 Tazlin
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Check the review batch: its feature fields, its curation batches, and the sheet it is signed off from."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import override

import pytest
from PIL import Image
from pydantic import ValidationError

from param_sweeps.campaign import JobFilter, run_campaign, stage_dependent_source
from param_sweeps.horde_runner import HordeGenerationRequest, silently_ignored_features_in
from param_sweeps.local_runner import HordelibPayload
from param_sweeps.outputs import INDEX_FILENAME, SweepIndex
from param_sweeps.review import (
    CONTACT_SHEET_FILENAME,
    apply_choices,
    build_contact_sheet,
    job_anchor,
    section_anchor,
    write_contact_sheet,
)
from param_sweeps.runner import FetchPlanEntry, RenderOutcome, SweepRunner
from param_sweeps.spec import (
    RULING_OTHER_OPTION_ID,
    BatchSection,
    ModelSpec,
    RenderJob,
    RulingChoice,
    RulingControl,
    RulingItem,
    RulingRow,
    RulingSet,
    SourceProcessing,
    SweepParameter,
    SweepSpec,
    WorkflowName,
    load_spec,
)

BATCH_SPEC_PATH = Path(__file__).resolve().parents[1] / "signoff_batch_spec.json"

WORLD_DESCRIPTOR_BLOCKS = {
    "w1": (
        "a woman in her 30s with a sleek silver bob haircut, round copper goggles resting up in her silver hair,"
        " a patched teal canvas coat, a small brass compass on a cord around her neck"
    ),
    "w2": (
        "1girl, shrine keeper, white haori, red hakama, long black hair, single hairpin,"
        " (silver fox:1.2) sitting at her side, fox, amber eyes"
    ),
    "w3": (
        "a small rounded gardening robot with weathered orange paint, a single round blue lens for a head,"
        " a hand-painted daisy decal on its chest plate, young ivy curling around its left arm"
    ),
}
"""The descriptor blocks the batch promises are byte-identical everywhere a world's subject appears."""


class SolidColourRunner(SweepRunner):
    """A runner that produces one solid colour per candidate, so the driver runs without a GPU."""

    def __init__(self) -> None:
        """Create a runner that records every job it was handed."""
        self.rendered_paths: list[str] = []

    @property
    @override
    def name(self) -> str:
        return "test-runner"

    @override
    def fetch_plan(self, jobs: Sequence[RenderJob], models: Sequence[ModelSpec]) -> list[FetchPlanEntry]:
        return [
            FetchPlanEntry(
                model_name=image_model.horde_name,
                approximate_size_gb=0.0,
                already_present=True,
                image_count=len(jobs),
            )
            for image_model in models
        ]

    @override
    def prepare(self, jobs: Sequence[RenderJob], models: Sequence[ModelSpec]) -> None:
        return

    @override
    def render(self, job: RenderJob) -> RenderOutcome:
        self.rendered_paths.append(job.relative_image_path)
        return RenderOutcome(
            images=[Image.new("RGB", (16, 16), color=(candidate * 20, 40, 60)) for candidate in job.candidate_numbers],
        )


@pytest.fixture(scope="module")
def batch_spec() -> SweepSpec:
    return load_spec(BATCH_SPEC_PATH)


def _batch_jobs(spec: SweepSpec) -> list[RenderJob]:
    return [job for job in spec.expand_jobs() if job.parameter is SweepParameter.BATCH]


def test_the_batch_expands_to_the_jobs_and_images_it_promises(batch_spec: SweepSpec) -> None:
    jobs = batch_spec.expand_jobs()
    batch_jobs = _batch_jobs(batch_spec)

    assert len(batch_jobs) == 35
    assert len(jobs) == 40, "the five denoising-strength stops are part of the batch's sign-off too"
    assert sum(job.batch_size for job in jobs) == 122

    sections = {strip.batch_section for strip in batch_spec.expand_strips() if strip.batch_section is not None}
    assert sections == {
        BatchSection.WORLD_SCENES,
        BatchSection.DIALECT_COMPARISON,
        BatchSection.FEATURE_PROBES,
    }, "this batch covers the three sections it was written for; later batches carry the rest"


def test_every_world_descriptor_block_is_byte_identical_wherever_its_subject_appears(batch_spec: SweepSpec) -> None:
    """Assert a world's subject is described with the same bytes in every scene that shows it.

    Two images of a world are only comparable if the subject is described identically; a single
    reworded adjective is enough to make a reader think the model drifted when the prompt did.
    """
    for world, descriptor_block in WORLD_DESCRIPTOR_BLOCKS.items():
        opening = descriptor_block[:40]
        carrying_prompts = [job.prompt for job in _batch_jobs(batch_spec) if opening in job.prompt]

        assert carrying_prompts, f"no job carries the {world} descriptor block"
        for prompt in carrying_prompts:
            assert descriptor_block in prompt, f"the {world} descriptor block was reworded in: {prompt[:80]}"


def test_flux_jobs_obey_the_schedule_its_model_reference_requires(batch_spec: SweepSpec) -> None:
    """Assert every Flux job runs the sampler, step count and guidance the model can actually take."""
    flux_jobs = [job for job in batch_spec.expand_jobs() if job.model_slug == "flux-schnell"]
    assert flux_jobs

    for job in flux_jobs:
        assert job.sampler_name == "k_euler"
        # The sampler binding passes denoising strength straight through for
        # every family, so a value below 1 on a text-to-image job silently
        # truncates the schedule: thirty-step SD merely loses steps, but
        # four-step flux comes out as a washed, static-flecked sketch.
        # Verified by a controlled probe: karras on/off is indistinguishable
        # on this model, denoise 0.75 vs 1.0 is the whole failure.
        assert job.denoising_strength == 1.0
        assert job.steps == 4
        assert job.cfg_scale == 1.0
        assert job.negative_prompt == "", "Flux takes no negative prompt"


def test_the_models_that_require_clip_skip_two_get_it(batch_spec: SweepSpec) -> None:
    for job in batch_spec.expand_jobs():
        if job.model_slug in {"nova-anime-xl", "cyberrealistic-pony"}:
            assert job.clip_skip == 2


def test_the_dialect_comparison_holds_one_seed_and_one_composition(batch_spec: SweepSpec) -> None:
    dialect_jobs = [
        job
        for strip in batch_spec.expand_strips()
        if strip.batch_section is BatchSection.DIALECT_COMPARISON
        for job in strip.jobs
    ]

    assert len(dialect_jobs) == 4
    assert len({job.seed for job in dialect_jobs}) == 1
    assert len({job.model_slug for job in dialect_jobs}) == 4
    assert all(job.batch_size == 1 for job in dialect_jobs), "a comparison must not be curated"


def test_the_qr_probe_carries_the_payload_and_the_offsets(batch_spec: SweepSpec) -> None:
    qr_job = next(job for job in batch_spec.expand_jobs() if job.workflow is WorkflowName.QR_CODE)

    references = {extra_text.reference: extra_text.text for extra_text in qr_job.extra_texts}
    assert references["qr_code"] == "https://aihorde.net"
    # The SD1.5 QR path never produced a code a scanner would read (its layer
    # strengths are fixed inside the graph); the SDXL path decodes reliably at
    # this strength, verified against a real decoder. The code sits centered:
    # offsets are omitted on purpose.
    assert qr_job.model_slug == "albedobase-xl"
    assert qr_job.control_strength == 1.2
    assert (qr_job.width, qr_job.height) == (1024, 1024)
    assert qr_job.batch_size == 8


def test_the_canny_probes_declare_where_their_source_comes_from(batch_spec: SweepSpec) -> None:
    canny_jobs = [job for job in batch_spec.expand_jobs() if job.control_type == "canny"]

    assert len(canny_jobs) == 3
    assert all(job.requires_source_from == "w1-panorama-albedobase" for job in canny_jobs)
    assert sum(1 for job in canny_jobs if job.return_control_map) == 1


def test_a_qr_workflow_without_its_payload_text_is_rejected() -> None:
    with pytest.raises(ValidationError, match="qr_code"):
        RenderJob.model_validate(_minimal_job_fields() | {"workflow": "qr_code"})


def test_features_that_need_a_source_image_are_rejected_without_one() -> None:
    with pytest.raises(ValidationError, match="control_type"):
        RenderJob.model_validate(_minimal_job_fields() | {"control_type": "canny"})

    with pytest.raises(ValidationError, match="source mask"):
        RenderJob.model_validate(_minimal_job_fields() | {"source_mask": "inputs/mask.png"})

    with pytest.raises(ValidationError, match="source_processing"):
        RenderJob.model_validate(_minimal_job_fields() | {"source_processing": "inpainting"})


def test_a_chosen_candidate_outside_the_batch_is_rejected() -> None:
    with pytest.raises(ValidationError, match="outside a batch"):
        RenderJob.model_validate(_minimal_job_fields() | {"batch_size": 2, "chosen_candidate": 3})


def test_a_curation_batch_names_every_candidate_and_publishes_the_chosen_one() -> None:
    job = RenderJob.model_validate(_minimal_job_fields() | {"batch_size": 4, "chosen_candidate": 3})

    assert job.candidate_numbers == [1, 2, 3, 4]
    assert job.relative_candidate_image_path(1).endswith("001_probe_01.webp")
    assert job.relative_image_path.endswith("001_probe_03.webp")
    assert job.checkpoint_key.endswith("001_probe_01.webp"), "picking a candidate must not re-render the batch"

    single = RenderJob.model_validate(_minimal_job_fields())
    assert single.relative_image_path.endswith("001_probe.webp"), "a single render keeps its plain name"


def test_a_curation_batch_writes_every_candidate_and_indexes_the_chosen_one(
    batch_spec: SweepSpec,
    tmp_path: Path,
) -> None:
    runner = SolidColourRunner()
    result = run_campaign(
        spec=batch_spec,
        runner=runner,
        output_root=tmp_path,
        job_filter=JobFilter(scene_ids=("c-transparent-albedobase",)),
    )

    assert result.rendered_count == 1
    directory = tmp_path / "batch/c-feature-probes/albedobase-xl"
    written = sorted(path.name for path in directory.glob("*.webp"))
    assert written == [
        "005_c-transparent-albedobase_01.webp",
        "005_c-transparent-albedobase_02.webp",
        "005_c-transparent-albedobase_03.webp",
        "005_c-transparent-albedobase_04.webp",
    ]

    index = SweepIndex.model_validate_json((tmp_path / INDEX_FILENAME).read_text(encoding="utf-8"))
    stop = index.strips[0].stops[0]
    assert stop.batch_size == 4
    job = next(job for job in batch_spec.expand_jobs() if job.scene_id == "c-transparent-albedobase")
    assert stop.file.endswith(f"_{job.chosen_candidate_number:02d}.webp"), "the recorded choice is what publishes"
    assert index.strips[0].features is not None
    assert index.strips[0].features.transparent is True
    assert index.strips[0].batch_section is BatchSection.FEATURE_PROBES

    sidecars = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.glob("*.json"))]
    assert sum(1 for sidecar in sidecars if sidecar["is_chosen"]) == 1


def test_a_dependent_job_is_staged_from_the_render_it_names(batch_spec: SweepSpec, tmp_path: Path) -> None:
    dependent = next(job for job in batch_spec.expand_jobs() if job.requires_source_from is not None)
    producer = next(job for job in batch_spec.expand_jobs() if job.scene_id == dependent.requires_source_from)

    with pytest.raises(RuntimeError, match="has not been rendered yet"):
        stage_dependent_source(job=dependent, spec=batch_spec, output_root=tmp_path)

    produced_path = tmp_path / producer.relative_image_path
    produced_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=(1, 2, 3)).save(produced_path, format="WEBP")

    staged = stage_dependent_source(job=dependent, spec=batch_spec, output_root=tmp_path)
    assert staged is not None
    try:
        assert staged.is_file()
        assert staged == dependent.source_image_path
    finally:
        staged.unlink()


def test_both_runners_carry_the_same_features_into_their_own_payloads(batch_spec: SweepSpec) -> None:
    """Assert a feature stated once in the specification reaches both backends.

    The two payloads have different field names and different shapes, which is exactly why this
    is worth asserting: a feature added to one mapping and forgotten in the other would only show
    up as a silently plainer image.
    """
    qr_job = next(job for job in batch_spec.expand_jobs() if job.workflow is WorkflowName.QR_CODE)

    hordelib_mapping = HordelibPayload.from_job(qr_job).as_hordelib_mapping()
    assert hordelib_mapping["workflow"] == "qr_code"
    assert hordelib_mapping["extra_texts"][0]["reference"] == "qr_code"
    assert hordelib_mapping["n_iter"] == 8

    api_params = HordeGenerationRequest.from_job(qr_job).as_request_body()["params"]
    assert api_params["workflow"] == "qr_code"
    assert api_params["extra_texts"][0]["reference"] == "qr_code"
    assert api_params["n"] == 8


def test_the_control_strength_a_job_states_reaches_the_request_body(batch_spec: SweepSpec) -> None:
    """Assert a stated control strength is posted and an unstated one leaves the key out.

    The API takes ``control_strength`` in its params, so nothing is dropped on the way in and the
    runner has no ignored feature left to name.
    """
    qr_job = next(job for job in batch_spec.expand_jobs() if job.workflow is WorkflowName.QR_CODE)
    plain_job = next(job for job in batch_spec.expand_jobs() if job.features().is_plain)

    qr_params = HordeGenerationRequest.from_job(qr_job).as_request_body()["params"]
    assert qr_params["control_strength"] == qr_job.control_strength

    plain_params = HordeGenerationRequest.from_job(plain_job).as_request_body()["params"]
    assert "control_strength" not in plain_params

    assert silently_ignored_features_in(qr_job) == []
    assert silently_ignored_features_in(plain_job) == []


def test_a_plain_job_still_produces_the_payload_it_always_did(batch_spec: SweepSpec) -> None:
    """Assert the feature fields stay out of the payload of a job that turns none of them on."""
    plain_job = next(job for job in batch_spec.expand_jobs() if job.features().is_plain)

    api_params = HordeGenerationRequest.from_job(plain_job).as_request_body()["params"]
    assert set(api_params) == {
        "sampler_name",
        "cfg_scale",
        "width",
        "height",
        "steps",
        "seed",
        "karras",
        "hires_fix",
        "tiling",
        "clip_skip",
        "denoising_strength",
        "n",
    }

    hordelib_mapping = HordelibPayload.from_job(plain_job).as_hordelib_mapping()
    assert "workflow" not in hordelib_mapping
    assert "control_type" not in hordelib_mapping


def test_a_source_image_without_a_stated_processing_is_still_image_to_image() -> None:
    job = RenderJob.model_validate(_minimal_job_fields() | {"source_image": "inputs/whatever.webp"})
    assert job.resolved_source_processing is SourceProcessing.IMG2IMG


def test_the_contact_sheet_groups_the_batch_by_section_and_states_the_payload(
    batch_spec: SweepSpec,
    tmp_path: Path,
) -> None:
    run_campaign(
        spec=batch_spec,
        runner=SolidColourRunner(),
        output_root=tmp_path,
        job_filter=JobFilter(scene_ids=("b-deck-prose", "c-qr-forest")),
    )
    sheet_path = write_contact_sheet(spec=batch_spec, output_root=tmp_path, title="Sign-off")

    assert sheet_path.name == CONTACT_SHEET_FILENAME
    sheet = sheet_path.read_text(encoding="utf-8")

    assert "B. Dialect comparison" in sheet
    assert "C. Feature feasibility probes" in sheet
    assert "batch/b-dialects/flux-schnell/004_b-deck-prose.webp" in sheet
    assert "workflow qr_code" in sheet
    assert "not rendered" in sheet, "a job with no image yet is shown as missing rather than broken"


def test_the_contact_sheet_marks_which_candidate_was_chosen(batch_spec: SweepSpec, tmp_path: Path) -> None:
    sheet = build_contact_sheet(spec=batch_spec, output_root=tmp_path, title="Sign-off")
    assert 'class="candidate chosen"' in sheet


def test_the_contact_sheet_makes_every_curation_candidate_pickable(batch_spec: SweepSpec, tmp_path: Path) -> None:
    """Assert each candidate carries the key and number the choose subcommand maps back to a job."""
    sheet = build_contact_sheet(spec=batch_spec, output_root=tmp_path, title="Sign-off")
    curation_jobs = [job for job in batch_spec.expand_jobs() if job.batch_size > 1]

    assert curation_jobs, "the batch is expected to curate something"
    for job in curation_jobs:
        assert f'data-choice-key="{job.checkpoint_key}"' in sheet
    assert sheet.count("data-candidate=") == sum(job.batch_size for job in curation_jobs)
    assert "Save choices" in sheet


def test_apply_choices_writes_the_picks_back_into_the_specification(batch_spec: SweepSpec, tmp_path: Path) -> None:
    spec_copy = tmp_path / "spec.json"
    spec_copy.write_text(BATCH_SPEC_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    target = next(job for job in batch_spec.expand_jobs() if job.scene_id == "c-transparent-nova")
    choices_path = tmp_path / "choices.json"
    choices_path.write_text(
        json.dumps({"schema_version": 1, "choices": {target.checkpoint_key: 3}}),
        encoding="utf-8",
    )

    assert apply_choices(spec_path=spec_copy, choices_path=choices_path) == 1

    updated = next(job for job in load_spec(spec_copy).expand_jobs() if job.scene_id == "c-transparent-nova")
    assert updated.chosen_candidate == 3
    assert apply_choices(spec_path=spec_copy, choices_path=choices_path) == 0, "the same picks twice change nothing"


def test_apply_choices_rejects_a_pick_the_batch_cannot_honour(batch_spec: SweepSpec, tmp_path: Path) -> None:
    spec_copy = tmp_path / "spec.json"
    spec_copy.write_text(BATCH_SPEC_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    target = next(job for job in batch_spec.expand_jobs() if job.scene_id == "c-transparent-nova")
    before = spec_copy.read_text(encoding="utf-8")

    out_of_range = tmp_path / "out_of_range.json"
    out_of_range.write_text(
        json.dumps({"schema_version": 1, "choices": {target.checkpoint_key: target.batch_size + 1}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="outside a batch"):
        apply_choices(spec_path=spec_copy, choices_path=out_of_range)

    unknown_job = tmp_path / "unknown_job.json"
    unknown_job.write_text(
        json.dumps({"schema_version": 1, "choices": {"batch/nowhere/001_nothing_01.webp": 1}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="names no curation job"):
        apply_choices(spec_path=spec_copy, choices_path=unknown_job)

    assert spec_copy.read_text(encoding="utf-8") == before, "a rejected choices file must touch nothing"


def _probe_ruling_set() -> RulingSet:
    """Return a ruling set exercising every control the sheet can render.

    Its evidence names jobs the checked-in batch really renders, so a link that goes nowhere is a
    test failure rather than something a reviewer discovers by clicking it.
    """
    return RulingSet(
        ruling_set_id="probe-wave",
        title="Rulings this probe needs",
        intro="Nothing here ships until every question below carries an answer.",
        items=[
            RulingItem(
                ruling_id="single-answer",
                title="A claim to affirm or reject",
                question="Does the render show what the claim says it shows?",
                control=RulingControl.RADIO,
                evidence_sections=[BatchSection.FEATURE_PROBES],
                evidence_job_ids=["c-qr-forest"],
                options=[
                    RulingChoice(choice_id="affirm", label="Affirm"),
                    RulingChoice(choice_id="reject", label="Reject"),
                ],
            ),
            RulingItem(
                ruling_id="approvals",
                title="Images to approve one by one",
                question="Which of these are correct?",
                control=RulingControl.CHECKBOX_SET,
                master_toggle_label="Approve all listed",
                options=[
                    RulingChoice(choice_id="deck", label="the deck", evidence_job_id="b-deck-prose"),
                    RulingChoice(choice_id="forest", label="the forest", evidence_job_id="c-qr-forest"),
                ],
            ),
            RulingItem(
                ruling_id="per-row",
                title="One answer per row",
                question="Is each description accurate?",
                note="The shared options answer the first row; the second states its own.",
                control=RulingControl.ROW_SELECT,
                options=[
                    RulingChoice(choice_id="accurate", label="Accurate"),
                    RulingChoice(choice_id="inaccurate", label="Inaccurate"),
                ],
                rows=[
                    RulingRow(row_id="shared", label="answered from the shared set"),
                    RulingRow(
                        row_id="its_own",
                        label="answered from its own set",
                        evidence_job_id="b-deck-prose",
                        options=[RulingChoice(choice_id="authorised", label="Authorised")],
                    ),
                ],
            ),
            RulingItem(
                ruling_id="written-out",
                title="An answer with no options at all",
                question="What should the copy say?",
                control=RulingControl.TEXT,
            ),
        ],
    )


@pytest.fixture(scope="module")
def ruled_spec(batch_spec: SweepSpec) -> SweepSpec:
    return batch_spec.model_copy(update={"rulings": [_probe_ruling_set()]})


def _rulings_panel_of(sheet: str) -> str:
    """Return only the rulings panel, so the script's own text cannot satisfy an assertion."""
    return sheet.split('<section class="rulings"')[1].split("</section>")[0]


def test_the_sheet_asks_every_ruling_with_the_control_it_states(ruled_spec: SweepSpec, tmp_path: Path) -> None:
    sheet = build_contact_sheet(spec=ruled_spec, output_root=tmp_path, title="Sign-off")

    assert "Rulings this probe needs" in sheet
    assert '<span class="rulings-count">0 of 4 ruled</span>' in sheet
    assert "Export rulings" in sheet

    assert 'id="ruling-single-answer-affirm"' in sheet
    assert '<input type="radio" name="ruling-single-answer"' in sheet
    assert '<input type="checkbox" name="ruling-approvals"' in sheet
    assert 'data-master="approvals"' in sheet
    assert '<select data-row="shared">' in sheet
    assert '<select data-row="its_own">' in sheet
    assert 'data-answer-text="written-out"' in sheet

    for ruling_id in ("single-answer", "approvals", "per-row", "written-out"):
        assert f'<textarea data-comment="{ruling_id}"' in sheet, "every ruling takes a comment"


def test_every_ruling_that_offers_options_offers_the_escape(ruled_spec: SweepSpec, tmp_path: Path) -> None:
    """Assert an answer the option set does not cover is still recordable, on every such ruling."""
    panel = _rulings_panel_of(build_contact_sheet(spec=ruled_spec, output_root=tmp_path, title="Sign-off"))

    # Two option sets on the radio and checkbox rulings, and one per row of the per-row ruling.
    assert panel.count(f'value="{RULING_OTHER_OPTION_ID}"') == 4
    assert "Other / see comments" in panel


def test_a_row_answers_from_its_own_options_when_it_states_them(ruled_spec: SweepSpec, tmp_path: Path) -> None:
    sheet = build_contact_sheet(spec=ruled_spec, output_root=tmp_path, title="Sign-off")
    own_row = sheet.split('<select data-row="its_own">')[1].split("</select>")[0]

    assert 'value="authorised"' in own_row
    assert 'value="accurate"' not in own_row, "the shared set does not leak into a row that stated its own"
    assert f'value="{RULING_OTHER_OPTION_ID}"' in own_row


def test_every_ruling_links_to_evidence_the_sheet_actually_carries(ruled_spec: SweepSpec, tmp_path: Path) -> None:
    """Assert each evidence link lands on an element the same page renders."""
    sheet = build_contact_sheet(spec=ruled_spec, output_root=tmp_path, title="Sign-off")

    for anchor in (
        section_anchor(BatchSection.FEATURE_PROBES),
        job_anchor("c-qr-forest"),
        job_anchor("b-deck-prose"),
    ):
        assert f'href="#{anchor}"' in sheet
        assert f'id="{anchor}"' in sheet


def test_the_panel_states_its_rulings_as_data_for_the_script(ruled_spec: SweepSpec, tmp_path: Path) -> None:
    sheet = build_contact_sheet(spec=ruled_spec, output_root=tmp_path, title="Sign-off")
    descriptor = json.loads(
        sheet.split('<script type="application/json" data-rulings-descriptor>')[1].split("</script>")[0],
    )

    assert descriptor["ruling_set_id"] == "probe-wave"
    assert [item["ruling_id"] for item in descriptor["items"]] == [
        "single-answer",
        "approvals",
        "per-row",
        "written-out",
    ]
    assert descriptor["items"][2]["rows"] == ["shared", "its_own"]
    assert 'data-ruling-set="probe-wave"' in sheet, "the answers are keyed by the set, not by the browser"


def test_two_ruling_sets_render_as_two_independently_keyed_panels(
    batch_spec: SweepSpec,
    tmp_path: Path,
) -> None:
    """Assert a follow-up set is a panel of its own, keyed apart from the set it follows."""
    follow_up = _probe_ruling_set().model_copy(
        update={"ruling_set_id": "probe-follow-up", "title": "What the first set left open"},
    )
    spec = batch_spec.model_copy(update={"rulings": [_probe_ruling_set(), follow_up]})

    sheet = build_contact_sheet(spec=spec, output_root=tmp_path, title="Sign-off")

    assert sheet.count('<section class="rulings"') == 2
    assert sheet.count('<script type="application/json" data-rulings-descriptor>') == 2
    assert 'data-ruling-set="probe-wave"' in sheet
    assert 'data-ruling-set="probe-follow-up"' in sheet


def test_two_ruling_sets_may_not_share_an_identifier() -> None:
    """Assert a repeated set id is rejected, since answers are keyed by it and would collide."""
    raw = json.loads(BATCH_SPEC_PATH.read_text(encoding="utf-8"))
    one_set = {
        "ruling_set_id": "probe-wave",
        "title": "Rulings this probe needs",
        "items": [
            {
                "ruling_id": "probe",
                "title": "A probe",
                "question": "Well?",
                "control": "radio",
                "options": [{"choice_id": "affirm", "label": "Affirm"}],
            },
        ],
    }
    raw["rulings"] = [one_set, one_set]

    with pytest.raises(ValidationError, match="ruling set ids must be unique"):
        SweepSpec.model_validate(raw)


def test_a_sheet_for_a_specification_with_no_rulings_carries_no_panel(
    batch_spec: SweepSpec,
    tmp_path: Path,
) -> None:
    sheet = build_contact_sheet(spec=batch_spec, output_root=tmp_path, title="Sign-off")

    assert 'class="rulings"' not in sheet
    assert "Save choices" in sheet, "the candidate-choose flow does not depend on there being rulings"


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"control": "radio", "options": []}, "with nothing to pick from"),
        ({"control": "row-select", "rows": []}, "with no rows to answer for"),
        ({"control": "text", "options": [{"choice_id": "affirm", "label": "Affirm"}]}, "answered in free text"),
        (
            {"control": "radio", "options": [{"choice_id": RULING_OTHER_OPTION_ID, "label": "Other"}]},
            "which is reserved",
        ),
        (
            {"control": "radio", "options": [{"choice_id": "Affirm It", "label": "Affirm"}]},
            "is not a lowercase identifier",
        ),
        (
            {
                "control": "radio",
                "options": [{"choice_id": "affirm", "label": "Affirm"}],
                "rows": [{"row_id": "one", "label": "one"}],
            },
            "which only a row-select has",
        ),
        (
            {
                "control": "radio",
                "options": [{"choice_id": "affirm", "label": "Affirm"}],
                "master_toggle_label": "Approve all",
            },
            "which only a checkbox set has",
        ),
    ],
)
def test_a_ruling_the_sheet_could_not_render_is_rejected(fields: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        RulingItem.model_validate(
            {
                "ruling_id": "probe",
                "title": "A probe",
                "question": "Well?",
                "options": [{"choice_id": "affirm", "label": "Affirm"}],
            }
            | fields,
        )


def test_a_ruling_citing_evidence_the_batch_does_not_render_is_rejected() -> None:
    """Assert a dead evidence link is caught here rather than by the reviewer clicking it."""
    raw = json.loads(BATCH_SPEC_PATH.read_text(encoding="utf-8"))
    raw["rulings"] = [
        {
            "ruling_set_id": "probe-wave",
            "title": "Rulings this probe needs",
            "items": [
                {
                    "ruling_id": "probe",
                    "title": "A probe",
                    "question": "Well?",
                    "control": "radio",
                    "evidence_job_ids": ["a-job-nothing-renders"],
                    "options": [{"choice_id": "affirm", "label": "Affirm"}],
                },
            ],
        },
    ]

    with pytest.raises(ValidationError, match="cites jobs no batch renders"):
        SweepSpec.model_validate(raw)


def _minimal_job_fields() -> dict[str, object]:
    """Return the fields every render job needs, for the validation cases to build on."""
    return {
        "parameter": SweepParameter.BATCH,
        "model_slug": "probe-model",
        "horde_model_name": "Probe",
        "scene_id": "probe",
        "prompt": "a probe",
        "stop_index": 1,
        "stop_label": "probe",
        "stop_value": "probe",
        "caption": "a probe",
        "width": 512,
        "height": 512,
        "steps": 30,
        "cfg_scale": 7.0,
        "sampler_name": "k_euler_a",
        "clip_skip": 1,
        "hires_fix": False,
        "tiling": False,
        "karras": True,
        "denoising_strength": 0.75,
        "seed": 1,
    }
