from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from PIL import Image, ImageDraw
from pydantic import ValidationError
import pytest

from student_analyze import __version__
from student_analyze.assets import artifact_digest
from student_analyze.atomic import serialize_model
from student_analyze.config import AppConfig
from student_analyze.document_mapper import map_documents
from student_analyze.document_models import (
    DecisionMethod,
    DocumentDecision,
    DocumentDecisionProvenance,
    DocumentGraphDecisionSet,
    DocumentRole,
    PageClassificationDecision,
    QuestionDecision,
    QuestionVersionDecision,
    RegionDecision,
    RegionKind,
    RelationDecision,
    RelationDecisionStatus,
    RelationType,
)
from student_analyze.errors import CaseValidationError
from student_analyze.exam_master_models import (
    AnnotationActor,
    AnswerEntryDecision,
    AnswerEvidenceDecision,
    AnswerProvenance,
    AnswerSource,
    AnswerSupport,
    ApprovalStatus,
    ChoiceOptionDecision,
    EndorsementKind,
    EvidenceDecisionStatus,
    EvidenceSourceKind,
    ExamMaster,
    QuestionReconstructionDecision,
    QuestionType,
    RubricCriterionDecision,
    RubricSupport,
    ScopeKind,
    SolverInputManifest,
    SolverQuestionInput,
    SolverRouteReason,
    SolverTask,
    VerificationLevel,
    VerificationMethod,
    VerificationResult,
    VerificationStatus,
    MasterQuestion,
)
from student_analyze.fingerprint import digest_value
from student_analyze.image_preprocess import prepare_logical_pages
from student_analyze.models import ImplementationVersions, PipelineStage, SCHEMA_VERSION
from student_analyze.page_models import (
    LogicalPageDecision,
    PageDecisionProvenance,
    PageDecisionSet,
    PageLayout,
    PagePosition,
    PixelBox,
    SourcePageDecision,
)
from student_analyze.pipeline import (
    build_stage_fingerprint,
    commit_stage_artifact,
    ingest_case,
    verify_case,
)
from student_analyze.submission import (
    build_submission,
    prepare_submission_inputs,
    prepare_submission_structure,
)
from student_analyze.submission_models import (
    SubmissionMappingDecision,
    SubmissionMappingDecisionSet,
    SubmissionSourceRole,
    SubmissionTranscriptionDecision,
    SubmissionTranscriptionDecisionSet,
    VisibleContentRole,
)


NOW = datetime(2026, 8, 11, tzinfo=UTC)


def _provenance(
    *, prompt: str, skill: str = "test-skill-v1"
) -> DocumentDecisionProvenance:
    return DocumentDecisionProvenance(
        method=DecisionMethod.DETERMINISTIC_TEST_FIXTURE,
        model_identifier="test-model",
        prompt_version=prompt,
        skill_version=skill,
        decided_at=NOW,
    )


def _ready_case(
    tmp_path: Path,
    *,
    teacher_annotation: bool = False,
    question_type: QuestionType = QuestionType.OBJECTIVE_SINGLE,
):
    source = tmp_path / "raw"
    source.mkdir()
    Image.new("RGB", (40, 24), "white").save(source / "question.jpg", "JPEG")
    answer = Image.new("RGB", (40, 24), "white")
    draw = ImageDraw.Draw(answer)
    draw.text((6, 6), "B", fill="black")
    answer.save(source / "answer.jpg", "JPEG")
    config = AppConfig(cases_dir=tmp_path / "cases")
    ingested = ingest_case(source, config)
    case_manifest_sha256, _ = artifact_digest(ingested.case_dir / "case_manifest.json")
    assets = {item.relative_path: item for item in ingested.manifest.source_assets}
    page_decisions = PageDecisionSet(
        case_id=ingested.manifest.case_id,
        source_manifest_sha256=case_manifest_sha256,
        provenance=PageDecisionProvenance(
            method="deterministic_test_fixture",
            model_identifier="test-model",
            prompt_version="test-pages-v1",
            decided_at=NOW,
        ),
        assets=[
            SourcePageDecision(
                source_asset_id=asset.asset_id,
                source_sha256=asset.sha256,
                layout=PageLayout.SINGLE_PAGE,
                layout_confidence=1.0,
                pages=[
                    LogicalPageDecision(
                        position=PagePosition.SINGLE,
                        crop_box=PixelBox(left=0, top=0, right=40, bottom=24),
                        rotation_clockwise=0,
                        orientation_confidence=1.0,
                        boundary_confidence=1.0,
                        evidence=["single-page test fixture"],
                    )
                ],
            )
            for asset in assets.values()
        ],
    )
    pages = prepare_logical_pages(ingested.case_dir, page_decisions, config)
    page_manifest_sha256, _ = artifact_digest(
        ingested.case_dir / pages.state.completed_stages[-1].artifacts[0].relative_path
    )
    page_by_source = {page.source_relative_path: page for page in pages.manifest.pages}
    question_page = page_by_source["question.jpg"]
    answer_page = page_by_source["answer.jpg"]
    mapping_decisions = DocumentGraphDecisionSet(
        case_id=ingested.manifest.case_id,
        page_manifest_sha256=page_manifest_sha256,
        provenance=_provenance(prompt="test-map-v1"),
        documents=[
            DocumentDecision(
                ref="question-booklet",
                role=DocumentRole.QUESTION_BOOKLET,
                confidence=1.0,
                evidence=["question fixture"],
            ),
            DocumentDecision(
                ref="answer-sheet",
                role=DocumentRole.ANSWER_SHEET,
                confidence=1.0,
                evidence=["answer fixture"],
            ),
        ],
        pages=[
            PageClassificationDecision(
                page_id=question_page.page_id,
                document_ref="question-booklet",
                order=1,
                confidence=1.0,
                evidence=["question page"],
            ),
            PageClassificationDecision(
                page_id=answer_page.page_id,
                document_ref="answer-sheet",
                order=1,
                confidence=1.0,
                evidence=["answer page"],
            ),
        ],
        questions=[
            QuestionDecision(
                ref="q1",
                printed_label="1",
                order=1,
                confidence=1.0,
                evidence=["question 1"],
            )
        ],
        question_versions=[
            QuestionVersionDecision(
                ref="q1-original",
                question_ref="q1",
                label="original",
                region_refs=["q1-region"],
                confidence=1.0,
                evidence=["original question"],
            )
        ],
        regions=[
            RegionDecision(
                ref="q1-region",
                page_id=question_page.page_id,
                kind=RegionKind.PRINTED_QUESTION,
                bbox=PixelBox(left=2, top=2, right=38, bottom=22),
                order=1,
                confidence=1.0,
                evidence=["printed question region"],
            ),
            RegionDecision(
                ref="q1-answer",
                page_id=answer_page.page_id,
                kind=RegionKind.ANSWER_AREA,
                bbox=PixelBox(left=2, top=2, right=38, bottom=22),
                order=1,
                confidence=1.0,
                evidence=["answer region"],
            ),
        ],
        relations=[
            RelationDecision(
                type=RelationType.ANSWERS,
                from_ref="q1-answer",
                to_ref="q1-original",
                status=RelationDecisionStatus.ACCEPTED,
                confidence=1.0,
                evidence=["answer box is labeled 1"],
                requires_review=False,
            )
        ],
    )
    mapped = map_documents(ingested.case_dir, mapping_decisions, config)
    graph_sha256, _ = artifact_digest(
        ingested.case_dir / mapped.state.completed_stages[-1].artifacts[0].relative_path
    )
    question = mapped.graph.questions[0]
    version_id = question.effective_version_id
    assert version_id is not None
    objective = question_type in {
        QuestionType.OBJECTIVE_SINGLE,
        QuestionType.OBJECTIVE_MULTIPLE,
    }
    reconstruction = QuestionReconstructionDecision(
        question_id=question.question_id,
        version_id=version_id,
        prompt_text="Choose one option.",
        question_type=question_type,
        points=1,
        options=(
            [
                ChoiceOptionDecision(label="A", text="first"),
                ChoiceOptionDecision(label="B", text="second"),
            ]
            if objective
            else []
        ),
        source_contains_student_content=False,
        source_contains_teacher_annotation=False,
        confidence=1.0,
        evidence=["clean deterministic fixture"],
        requires_review=False,
    )
    solver_manifest = SolverInputManifest(
        case_id=ingested.manifest.case_id,
        document_graph_sha256=graph_sha256,
        answer_evidence_decision_sha256="0" * 64,
        question_reconstruction_decision_sha256="1" * 64,
        crop_decision_sha256="2" * 64,
        manifest_fingerprint="3" * 64,
        created_at=NOW,
        question_reconstruction_provenance=_provenance(
            prompt="test-reconstruction-v1"
        ),
        questions=[
            SolverQuestionInput(
                question_id=question.question_id,
                version_id=version_id,
                printed_label="1",
                task=SolverTask.RECONSTRUCT_AND_SOLVE,
                route_reason=SolverRouteReason.NO_RELIABLE_EVIDENCE,
                reconstruction=reconstruction,
            )
        ],
    )
    solver_manifest_sha256 = sha256(serialize_model(solver_manifest)).hexdigest()
    answer_region = next(
        item for item in mapped.graph.regions if item.decision_ref == "q1-answer"
    )
    answer_evidence = []
    if teacher_annotation:
        answer_evidence.append(
            AnswerEvidenceDecision(
                ref="possible-teacher-mark",
                source_kind=EvidenceSourceKind.TEACHER_ANNOTATION,
                page_id=answer_page.page_id,
                bbox=PixelBox(left=16, top=4, right=20, bottom=8),
                observed_content="tick-shaped stroke",
                actor=AnnotationActor.UNKNOWN,
                endorsement_kind=EndorsementKind.UNKNOWN,
                scope_kind=ScopeKind.QUESTION,
                question_ids=[question.question_id],
                answer_support=AnswerSupport.NONE,
                rubric_support=RubricSupport.NONE,
                status=EvidenceDecisionStatus.CANDIDATE,
                confidence=0.3,
                evidence=["actor and meaning are uncertain"],
                requires_review=True,
            )
        )
    versions = ImplementationVersions(
        code=__version__,
        base_schema=SCHEMA_VERSION,
        config=config.config_version,
        model="test-model",
        prompt="test-master-v1",
        skill="test-master-skill-v1",
    )
    stage_inputs = [{"fixture": "phase-5-ready-case"}]
    stage_config = config.master_fingerprint_payload()
    stage_fingerprint, _ = build_stage_fingerprint(
        stage=PipelineStage.MASTER_READY,
        model_type=ExamMaster,
        schema_id="exam_master.schema.json",
        config=stage_config,
        versions=versions,
        inputs=stage_inputs,
    )
    provenance = _provenance(prompt="test-master-v1", skill="test-master-skill-v1")
    master = ExamMaster(
        case_id=ingested.manifest.case_id,
        document_graph_sha256=graph_sha256,
        answer_evidence_decision_sha256="0" * 64,
        question_reconstruction_decision_sha256="1" * 64,
        crop_decision_sha256="2" * 64,
        solver_input_manifest_sha256=solver_manifest_sha256,
        exam_master_decision_sha256="4" * 64,
        exam_review_decision_sha256="5" * 64,
        stage_fingerprint=stage_fingerprint,
        created_at=NOW,
        evidence_provenance=provenance,
        reconstruction_provenance=provenance,
        solver_provenance=provenance,
        review_provenance=provenance,
        answer_evidence=answer_evidence,
        solver_input_manifest=solver_manifest,
        questions=[
            MasterQuestion(
                question_id=question.question_id,
                version_id=version_id,
                printed_label="1",
                prompt_text="Choose one option.",
                question_type=question_type,
                points=1,
                options=reconstruction.options,
                solver_required=True,
                route_reason=SolverRouteReason.NO_RELIABLE_EVIDENCE,
                reference_answers=[AnswerEntryDecision(answer="A")],
                rubric=[
                    RubricCriterionDecision(
                        ref="rubric-q1", description="Select A.", points=1
                    )
                ],
                solution_summary="A is the independently verified answer.",
                answer_provenance=[
                    AnswerProvenance(
                        source=AnswerSource.INDEPENDENT_SOLUTION,
                        source_scope=[question.question_id],
                        verification_level=VerificationLevel.INDEPENDENTLY_REVIEWED,
                    )
                ],
                verification_results=[
                    VerificationResult(
                        method=(
                            VerificationMethod.OPTION_MEMBERSHIP
                            if objective
                            else VerificationMethod.LOGIC_REVIEW
                        ),
                        status=(
                            VerificationStatus.PASSED
                            if objective
                            else VerificationStatus.NOT_APPLICABLE
                        ),
                        details=(
                            "Reference answer is a listed option."
                            if objective
                            else "The fixture rubric is reviewed semantically."
                        ),
                    )
                ],
                approval_status=ApprovalStatus.APPROVED,
                requires_review=False,
            )
        ],
    )
    committed = commit_stage_artifact(
        ingested.case_dir,
        stage=PipelineStage.MASTER_READY,
        artifact_name="exam_master.json",
        payload=master,
        model_type=ExamMaster,
        schema_id="exam_master.schema.json",
        config=stage_config,
        versions=versions,
        inputs=stage_inputs,
    )
    assert committed.state.current_stage == PipelineStage.MASTER_READY
    return ingested.case_dir, config, question, version_id, answer_region, answer_page


def _mapping_set(
    structure,
    question,
    version_id,
    answer_region,
    answer_page,
    *,
    teacher=False,
    include_formal=True,
    include_scratch=False,
):
    structure_sha256 = digest_value(structure.model_dump(mode="json"))
    items = []
    if include_formal:
        items.append(
            SubmissionMappingDecision(
                ref="q1-formal-answer",
                question_id=question.question_id,
                version_id=version_id,
                slot_label="answer",
                slot_order=1,
                source_role=SubmissionSourceRole.ANSWER_SHEET,
                source_region_id=answer_region.region_id,
                page_id=answer_page.page_id,
                bbox=PixelBox(left=4, top=4, right=20, bottom=16),
                visible_roles=(
                    [
                        VisibleContentRole.STUDENT_HANDWRITING,
                        VisibleContentRole.TEACHER_ANNOTATION,
                    ]
                    if teacher
                    else [VisibleContentRole.STUDENT_HANDWRITING]
                ),
                excluded_annotation_refs=(
                    ["possible-teacher-mark"] if teacher else []
                ),
                confidence=1.0,
                evidence=["response unit is inside the printed answer box"],
            )
        )
    if include_scratch:
        question_page = next(
            page
            for page in structure.pages
            if page.document_role == DocumentRole.QUESTION_BOOKLET
        )
        items.append(
            SubmissionMappingDecision(
                ref="q1-scratch-answer",
                question_id=question.question_id,
                version_id=version_id,
                slot_label="scratch",
                slot_order=1,
                source_role=SubmissionSourceRole.QUESTION_BOOKLET_SCRATCH,
                page_id=question_page.page_id,
                bbox=PixelBox(left=4, top=4, right=20, bottom=16),
                visible_roles=[VisibleContentRole.STUDENT_HANDWRITING],
                confidence=1.0,
                evidence=["student working is visible on the question page"],
            )
        )
    return SubmissionMappingDecisionSet(
        case_id=structure.case_id,
        structure_manifest_sha256=structure_sha256,
        provenance=_provenance(
            prompt="submission-mapping-v1.0.0",
            skill="exam-submission-transcription-v1.0.0",
        ),
        items=items,
    )


def _transcriptions(manifest, decisions):
    return SubmissionTranscriptionDecisionSet(
        case_id=manifest.case_id,
        submission_input_manifest_sha256=sha256(
            serialize_model(manifest)
        ).hexdigest(),
        provenance=_provenance(
            prompt="submission-transcription-v1.0.0",
            skill="exam-submission-transcription-v1.0.0",
        ),
        decisions=decisions,
    )


def test_submission_commits_with_unresolved_teacher_mark_in_review_queue(
    tmp_path: Path,
) -> None:
    case_dir, config, question, version_id, answer_region, answer_page = _ready_case(
        tmp_path, teacher_annotation=True
    )
    structure_result = prepare_submission_structure(case_dir, config)
    mappings = _mapping_set(
        structure_result.manifest,
        question,
        version_id,
        answer_region,
        answer_page,
        teacher=True,
    )
    inputs = prepare_submission_inputs(
        case_dir, structure_result.manifest, mappings, config
    )
    transcriptions = _transcriptions(
        inputs.manifest,
        [
            SubmissionTranscriptionDecision(
                mapping_ref="q1-formal-answer",
                observed_content="b",
                normalized_answer="B",
                is_blank=False,
                has_erasure=False,
                confidence=0.99,
                evidence=["a lowercase b is visibly written"],
            )
        ],
    )

    built = build_submission(case_dir, inputs.manifest, transcriptions, config)

    assert built.state.current_stage == PipelineStage.SUBMISSION_READY
    assert built.submission.requires_review
    assert len(built.submission.review_items) == 1
    assert built.submission.items[0].normalized_answer == "B"
    assert built.submission.items[0].crop.page_bbox == PixelBox(
        left=4, top=4, right=20, bottom=16
    )
    assert built.submission.items[0].crop.raw_bbox == PixelBox(
        left=4, top=4, right=20, bottom=16
    )
    _, verified_state = verify_case(case_dir)
    assert verified_state.current_stage == PipelineStage.SUBMISSION_READY

    reused = build_submission(case_dir, inputs.manifest, transcriptions, config)
    assert reused.reused


def test_formal_blank_is_not_replaced_by_question_booklet_scratch(
    tmp_path: Path,
) -> None:
    case_dir, config, question, version_id, answer_region, answer_page = _ready_case(
        tmp_path
    )
    structure = prepare_submission_structure(case_dir, config).manifest
    mappings = _mapping_set(
        structure,
        question,
        version_id,
        answer_region,
        answer_page,
        include_scratch=True,
    )
    inputs = prepare_submission_inputs(case_dir, structure, mappings, config)
    transcriptions = _transcriptions(
        inputs.manifest,
        [
            SubmissionTranscriptionDecision(
                mapping_ref="q1-formal-answer",
                is_blank=True,
                has_erasure=False,
                confidence=1.0,
                evidence=["the answer-sheet slot is untouched"],
            ),
            SubmissionTranscriptionDecision(
                mapping_ref="q1-scratch-answer",
                observed_content="A",
                normalized_answer="A",
                is_blank=False,
                has_erasure=False,
                confidence=1.0,
                evidence=["A is visible in scratch work"],
            ),
        ],
    )

    built = build_submission(case_dir, inputs.manifest, transcriptions, config)

    formal = next(
        item
        for item in built.submission.items
        if item.source_role == SubmissionSourceRole.ANSWER_SHEET
    )
    scratch = next(
        item
        for item in built.submission.items
        if item.source_role == SubmissionSourceRole.QUESTION_BOOKLET_SCRATCH
    )
    assert formal.is_blank and formal.observed_content is None
    assert scratch.normalized_answer == "A"
    assert not built.submission.requires_review


def test_missing_formal_answer_mapping_is_rejected(tmp_path: Path) -> None:
    case_dir, config, question, version_id, answer_region, answer_page = _ready_case(
        tmp_path
    )
    structure = prepare_submission_structure(case_dir, config).manifest
    mappings = _mapping_set(
        structure,
        question,
        version_id,
        answer_region,
        answer_page,
        include_formal=False,
        include_scratch=True,
    )

    with pytest.raises(CaseValidationError, match="cover every response target"):
        prepare_submission_inputs(case_dir, structure, mappings, config)


def test_objective_normalization_cannot_expand_option_meaning(tmp_path: Path) -> None:
    case_dir, config, question, version_id, answer_region, answer_page = _ready_case(
        tmp_path
    )
    structure = prepare_submission_structure(case_dir, config).manifest
    mappings = _mapping_set(
        structure, question, version_id, answer_region, answer_page
    )
    inputs = prepare_submission_inputs(case_dir, structure, mappings, config)
    transcriptions = _transcriptions(
        inputs.manifest,
        [
            SubmissionTranscriptionDecision(
                mapping_ref="q1-formal-answer",
                observed_content="b",
                normalized_answer="B: second",
                is_blank=False,
                has_erasure=False,
                confidence=1.0,
                evidence=["lowercase b is visible"],
            )
        ],
    )

    with pytest.raises(CaseValidationError, match="option label"):
        build_submission(case_dir, inputs.manifest, transcriptions, config)


def test_low_confidence_and_erased_blank_review_invariants() -> None:
    with pytest.raises(ValidationError, match="low-confidence"):
        SubmissionTranscriptionDecision(
            mapping_ref="q1-answer",
            observed_content="A",
            normalized_answer="A",
            is_blank=False,
            has_erasure=False,
            confidence=0.79,
            evidence=["unclear mark"],
        )
    with pytest.raises(ValidationError, match="only after human review"):
        SubmissionTranscriptionDecision(
            mapping_ref="q1-answer",
            is_blank=True,
            has_erasure=True,
            confidence=1.0,
            evidence=["cancelled writing"],
        )
    reviewed_blank = SubmissionTranscriptionDecision(
        mapping_ref="q1-answer",
        is_blank=True,
        has_erasure=True,
        confidence=1.0,
        evidence=["cancelled writing was reviewed as blank"],
        human_confirmed=True,
        review_note="Human reviewer directed the cancelled response to be scored blank.",
        blank_after_erasure_review=True,
    )
    assert reviewed_blank.is_blank and reviewed_blank.has_erasure
