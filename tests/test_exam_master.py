from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from PIL import Image
import pytest

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
)
from student_analyze.errors import ReviewRequiredError
from student_analyze.exam_master import build_exam_master, prepare_solver_inputs
from student_analyze.exam_master_models import (
    AnnotationActor,
    AnswerEntryDecision,
    AnswerEvidenceDecision,
    AnswerEvidenceDecisionSet,
    AnswerSource,
    AnswerSupport,
    ChoiceOptionDecision,
    CropReviewStatus,
    EndorsementKind,
    EvidenceAnswerDecision,
    EvidenceDecisionStatus,
    EvidenceSourceKind,
    ExamMasterDecisionSet,
    ExamReviewDecisionSet,
    IndependentReviewStatus,
    QuestionMasterDecision,
    QuestionReconstructionDecision,
    QuestionReconstructionDecisionSet,
    QuestionReviewDecision,
    QuestionType,
    RubricCriterionDecision,
    RubricSupport,
    ScopeKind,
    SolverCropDecision,
    SolverInputDecisionSet,
    SolverRouteReason,
    SolverTask,
    VerificationMethod,
    VerificationStatus,
)
from student_analyze.fingerprint import digest_value
from student_analyze.image_preprocess import prepare_logical_pages
from student_analyze.models import PipelineStage
from student_analyze.page_models import (
    LogicalPageDecision,
    PageDecisionProvenance,
    PageDecisionSet,
    PageLayout,
    PagePosition,
    PixelBox,
    SourcePageDecision,
)
from student_analyze.pipeline import ingest_case, verify_case


NOW = datetime(2026, 8, 10, tzinfo=UTC)


def _provenance(
    *,
    method: DecisionMethod = DecisionMethod.DETERMINISTIC_TEST_FIXTURE,
    prompt: str = "test-prompt-v1",
    skill: str = "test-skill-v1",
) -> DocumentDecisionProvenance:
    return DocumentDecisionProvenance(
        method=method,
        model_identifier=(None if method == DecisionMethod.HUMAN_REVIEW else "test-model"),
        model_identifier_unavailable_reason=(
            "human review" if method == DecisionMethod.HUMAN_REVIEW else None
        ),
        prompt_version=prompt,
        skill_version=skill,
        decided_at=NOW,
    )


def _mapped_case(tmp_path: Path):
    source = tmp_path / "raw"
    source.mkdir()
    Image.new("RGB", (40, 24), "white").save(source / "exam.jpg", format="JPEG")
    config = AppConfig(cases_dir=tmp_path / "cases")
    ingested = ingest_case(source, config)
    asset = ingested.manifest.source_assets[0]
    case_manifest_sha256, _ = artifact_digest(ingested.case_dir / "case_manifest.json")
    page_decisions = PageDecisionSet(
        case_id=ingested.manifest.case_id,
        source_manifest_sha256=case_manifest_sha256,
        provenance=PageDecisionProvenance(
            method="deterministic_test_fixture",
            model_identifier="test-model",
            prompt_version="test-page-v1",
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
                        evidence=["single question page fixture"],
                    )
                ],
            )
        ],
    )
    pages = prepare_logical_pages(ingested.case_dir, page_decisions, config)
    page_reference = pages.state.completed_stages[-1].artifacts[0]
    page_manifest_sha256, _ = artifact_digest(
        ingested.case_dir / page_reference.relative_path
    )
    page = pages.manifest.pages[0]
    mapping_decisions = DocumentGraphDecisionSet(
        case_id=ingested.manifest.case_id,
        page_manifest_sha256=page_manifest_sha256,
        provenance=_provenance(prompt="test-map-v1", skill="test-map-skill-v1"),
        documents=[
            DocumentDecision(
                ref="question-booklet",
                role=DocumentRole.QUESTION_BOOKLET,
                confidence=1.0,
                evidence=["question booklet fixture"],
            )
        ],
        pages=[
            PageClassificationDecision(
                page_id=page.page_id,
                document_ref="question-booklet",
                order=1,
                printed_page_number="1",
                confidence=1.0,
                evidence=["question page fixture"],
            )
        ],
        questions=[
            QuestionDecision(
                ref="q1",
                printed_label="1",
                order=1,
                confidence=1.0,
                evidence=["printed question 1"],
            )
        ],
        question_versions=[
            QuestionVersionDecision(
                ref="q1-original",
                question_ref="q1",
                label="original",
                region_refs=["q1-region"],
                confidence=1.0,
                evidence=["original question 1"],
            )
        ],
        regions=[
            RegionDecision(
                ref="q1-region",
                page_id=page.page_id,
                kind=RegionKind.PRINTED_QUESTION,
                bbox=PixelBox(left=2, top=2, right=38, bottom=22),
                order=1,
                confidence=1.0,
                evidence=["printed question region"],
            )
        ],
    )
    mapped = map_documents(ingested.case_dir, mapping_decisions, config)
    graph_reference = mapped.state.completed_stages[-1].artifacts[0]
    graph_sha256, _ = artifact_digest(
        ingested.case_dir / graph_reference.relative_path
    )
    question = mapped.graph.questions[0]
    version = next(
        item
        for item in mapped.graph.question_versions
        if item.version_id == question.effective_version_id
    )
    region = next(
        item for item in mapped.graph.regions if item.region_id in version.region_ids
    )
    return ingested, mapped, config, graph_sha256, question, version, region


def _crop_decisions(mapped, graph_sha256, question, version, region, **overrides):
    values = {
        "contains_student_handwriting": False,
        "contains_teacher_annotation": False,
        "contains_answer_content": False,
        "requires_review": False,
    }
    values.update(overrides)
    return SolverInputDecisionSet(
        case_id=mapped.graph.case_id,
        document_graph_sha256=graph_sha256,
        provenance=_provenance(prompt="test-crop-v1", skill="test-crop-skill-v1"),
        crops=[
            SolverCropDecision(
                ref="q1-prompt-crop",
                question_id=question.question_id,
                version_id=version.version_id,
                source_region_id=region.region_id,
                page_id=region.page_id,
                bbox=region.bbox,
                review_status=CropReviewStatus.DETERMINISTIC_TEST_FIXTURE,
                confidence=1.0,
                evidence=["clean printed prompt fixture"],
                **values,
            )
        ],
    )


def _reconstruction_decisions(
    mapped,
    graph_sha256,
    question,
    version,
    region,
    *,
    contaminated: bool = False,
    human_confirmed: bool = False,
    requires_review: bool = False,
):
    method = (
        DecisionMethod.HUMAN_REVIEW
        if human_confirmed
        else DecisionMethod.DETERMINISTIC_TEST_FIXTURE
    )
    return QuestionReconstructionDecisionSet(
        case_id=mapped.graph.case_id,
        document_graph_sha256=graph_sha256,
        provenance=_provenance(
            method=method,
            prompt="test-reconstruction-v1",
            skill="test-reconstruction-skill-v1",
        ),
        questions=[
            QuestionReconstructionDecision(
                question_id=question.question_id,
                version_id=version.version_id,
                prompt_text="Which option is correct?",
                question_type=QuestionType.OBJECTIVE_SINGLE,
                points=3,
                options=[
                    ChoiceOptionDecision(label="A", text="Option A"),
                    ChoiceOptionDecision(label="B", text="Option B"),
                ],
                visual_region_ids=[region.region_id],
                source_contains_student_content=contaminated,
                source_contains_teacher_annotation=False,
                human_confirmed=human_confirmed,
                confidence=1.0,
                evidence=["printed question reconstruction fixture"],
                requires_review=requires_review,
            )
        ],
    )


def _teacher_evidence(mapped, graph_sha256, question, region):
    return AnswerEvidenceDecisionSet(
        case_id=mapped.graph.case_id,
        document_graph_sha256=graph_sha256,
        provenance=_provenance(
            method=DecisionMethod.HUMAN_REVIEW,
            prompt="test-evidence-v1",
            skill="test-evidence-skill-v1",
        ),
        items=[
            AnswerEvidenceDecision(
                ref="teacher-check-q1",
                source_kind=EvidenceSourceKind.TEACHER_ANNOTATION,
                page_id=region.page_id,
                bbox=region.bbox,
                observed_content="teacher tick confirms selected option B",
                actor=AnnotationActor.TEACHER,
                endorsement_kind=EndorsementKind.CORRECT,
                scope_kind=ScopeKind.QUESTION,
                question_ids=[question.question_id],
                answer_support=AnswerSupport.ANSWER_KEY,
                rubric_support=RubricSupport.EXACT_MATCH,
                answers=[
                    EvidenceAnswerDecision(
                        question_id=question.question_id,
                        answer="B",
                    )
                ],
                status=EvidenceDecisionStatus.ACCEPTED,
                confidence=1.0,
                evidence=["human-confirmed teacher tick and scope"],
                human_confirmed=True,
                requires_review=False,
            )
        ],
    )


def _empty_evidence(mapped, graph_sha256):
    return AnswerEvidenceDecisionSet(
        case_id=mapped.graph.case_id,
        document_graph_sha256=graph_sha256,
        provenance=_provenance(prompt="test-evidence-v1", skill="test-evidence-skill-v1"),
    )


def _master_decisions(manifest, graph_sha256, *, solved: bool):
    manifest_sha256 = sha256(serialize_model(manifest)).hexdigest()
    reconstruction = manifest.questions[0].reconstruction
    return ExamMasterDecisionSet(
        case_id=manifest.case_id,
        document_graph_sha256=graph_sha256,
        solver_input_manifest_sha256=manifest_sha256,
        provenance=_provenance(prompt="test-solver-v1", skill="test-solver-skill-v1"),
        questions=[
            QuestionMasterDecision(
                question_id=manifest.questions[0].question_id,
                version_id=manifest.questions[0].version_id,
                prompt_text=reconstruction.prompt_text,
                question_type=reconstruction.question_type,
                points=reconstruction.points,
                options=reconstruction.options,
                subparts=reconstruction.subparts,
                knowledge_points=reconstruction.knowledge_points,
                solver_performed=solved,
                reference_answers=(
                    [AnswerEntryDecision(answer="B")] if solved else []
                ),
                rubric=(
                    [
                        RubricCriterionDecision(
                            ref="correct-choice",
                            description="Select B.",
                            points=3,
                        )
                    ]
                    if solved
                    else []
                ),
                solution_summary=("B follows from the prompt." if solved else None),
                verification_requests=(
                    [VerificationMethod.OPTION_MEMBERSHIP] if solved else []
                ),
            )
        ],
    )


def _review_decisions(manifest, master_decisions, *, solved: bool):
    return ExamReviewDecisionSet(
        case_id=manifest.case_id,
        solver_input_manifest_sha256=sha256(serialize_model(manifest)).hexdigest(),
        exam_master_decision_sha256=digest_value(
            master_decisions.model_dump(mode="json")
        ),
        provenance=_provenance(prompt="test-review-v1", skill="test-solver-skill-v1"),
        reviews=(
            [
                QuestionReviewDecision(
                    question_id=manifest.questions[0].question_id,
                    version_id=manifest.questions[0].version_id,
                    status=IndependentReviewStatus.CONFIRMED,
                    independently_derived_answer="B",
                    evidence=["independent solution also gives B"],
                )
            ]
            if solved
            else []
        ),
    )


def test_confirmed_teacher_answer_skips_solving_and_commits_master(tmp_path: Path) -> None:
    ingested, mapped, config, graph_sha256, question, version, region = _mapped_case(
        tmp_path
    )
    evidence = _teacher_evidence(mapped, graph_sha256, question, region)
    reconstructions = _reconstruction_decisions(
        mapped, graph_sha256, question, version, region
    )
    crops = _crop_decisions(mapped, graph_sha256, question, version, region)
    prepared = prepare_solver_inputs(
        ingested.case_dir, evidence, reconstructions, crops, config
    )

    assert prepared.manifest.questions[0].task == SolverTask.RECONSTRUCT_ONLY
    assert (
        prepared.manifest.questions[0].route_reason
        == SolverRouteReason.RELIABLE_EVIDENCE
    )
    decisions = _master_decisions(prepared.manifest, graph_sha256, solved=False)
    reviews = _review_decisions(prepared.manifest, decisions, solved=False)
    built = build_exam_master(
        ingested.case_dir,
        evidence,
        prepared.manifest,
        decisions,
        reviews,
        config,
    )

    assert built.state.current_stage == PipelineStage.MASTER_READY
    assert built.master.questions[0].reference_answers[0].answer == "B"
    assert (
        built.master.questions[0].answer_provenance[0].source
        == AnswerSource.TEACHER_ENDORSED_SUBMISSION
    )
    assert all(
        result.status != VerificationStatus.FAILED
        for result in built.master.questions[0].verification_results
    )
    verify_case(ingested.case_dir)


def test_missing_evidence_routes_to_blind_solution_and_review(tmp_path: Path) -> None:
    ingested, mapped, config, graph_sha256, question, version, region = _mapped_case(
        tmp_path
    )
    evidence = _empty_evidence(mapped, graph_sha256)
    reconstructions = _reconstruction_decisions(
        mapped, graph_sha256, question, version, region
    )
    crops = _crop_decisions(mapped, graph_sha256, question, version, region)
    prepared = prepare_solver_inputs(
        ingested.case_dir, evidence, reconstructions, crops, config
    )

    assert prepared.manifest.questions[0].task == SolverTask.RECONSTRUCT_AND_SOLVE
    decisions = _master_decisions(prepared.manifest, graph_sha256, solved=True)
    reviews = _review_decisions(prepared.manifest, decisions, solved=True)
    built = build_exam_master(
        ingested.case_dir,
        evidence,
        prepared.manifest,
        decisions,
        reviews,
        config,
    )

    question_result = built.master.questions[0]
    assert question_result.solver_required
    assert question_result.answer_provenance[0].source == AnswerSource.INDEPENDENT_SOLUTION
    assert not question_result.requires_review


def test_unconfirmed_teacher_mark_does_not_skip_blind_solving(tmp_path: Path) -> None:
    ingested, mapped, config, graph_sha256, question, version, region = _mapped_case(
        tmp_path
    )
    evidence = AnswerEvidenceDecisionSet(
        case_id=mapped.graph.case_id,
        document_graph_sha256=graph_sha256,
        provenance=_provenance(prompt="test-evidence-v1", skill="test-evidence-skill-v1"),
        items=[
            AnswerEvidenceDecision(
                ref="ambiguous-large-tick",
                source_kind=EvidenceSourceKind.TEACHER_ANNOTATION,
                page_id=region.page_id,
                bbox=region.bbox,
                observed_content="large tick near answer block",
                actor=AnnotationActor.UNKNOWN,
                endorsement_kind=EndorsementKind.UNKNOWN,
                scope_kind=ScopeKind.QUESTION,
                question_ids=[question.question_id],
                answer_support=AnswerSupport.NONE,
                rubric_support=RubricSupport.NONE,
                status=EvidenceDecisionStatus.CANDIDATE,
                confidence=0.5,
                evidence=["actor and meaning are not visually unique"],
                requires_review=True,
            )
        ],
    )
    prepared = prepare_solver_inputs(
        ingested.case_dir,
        evidence,
        _reconstruction_decisions(
            mapped, graph_sha256, question, version, region
        ),
        _crop_decisions(mapped, graph_sha256, question, version, region),
        config,
    )
    assert prepared.manifest.questions[0].task == SolverTask.RECONSTRUCT_AND_SOLVE
    assert (
        prepared.manifest.questions[0].route_reason
        == SolverRouteReason.NO_RELIABLE_EVIDENCE
    )


def test_solver_crop_with_student_handwriting_is_rejected(tmp_path: Path) -> None:
    ingested, mapped, config, graph_sha256, question, version, region = _mapped_case(
        tmp_path
    )
    with pytest.raises(ReviewRequiredError, match="question-only"):
        prepare_solver_inputs(
            ingested.case_dir,
            _empty_evidence(mapped, graph_sha256),
            _reconstruction_decisions(
                mapped, graph_sha256, question, version, region
            ),
            _crop_decisions(
                mapped,
                graph_sha256,
                question,
                version,
                region,
                contains_student_handwriting=True,
            ),
            config,
        )


def test_marked_question_reconstruction_requires_human_confirmation(
    tmp_path: Path,
) -> None:
    ingested, mapped, config, graph_sha256, question, version, region = _mapped_case(
        tmp_path
    )
    reconstructions = _reconstruction_decisions(
        mapped,
        graph_sha256,
        question,
        version,
        region,
        contaminated=True,
        requires_review=True,
    )
    with pytest.raises(ReviewRequiredError, match="requires review"):
        prepare_solver_inputs(
            ingested.case_dir,
            _empty_evidence(mapped, graph_sha256),
            reconstructions,
            _crop_decisions(mapped, graph_sha256, question, version, region),
            config,
        )


def test_reviewed_only_teacher_mark_cannot_claim_answer_support() -> None:
    with pytest.raises(ValueError, match="reviewed-only"):
        AnswerEvidenceDecision(
            ref="reviewed-only-q1",
            source_kind=EvidenceSourceKind.TEACHER_ANNOTATION,
            page_id="page-1",
            bbox=PixelBox(left=0, top=0, right=10, bottom=10),
            observed_content="tick meaning only reviewed",
            actor=AnnotationActor.UNKNOWN,
            endorsement_kind=EndorsementKind.REVIEWED_ONLY,
            scope_kind=ScopeKind.QUESTION,
            question_ids=["question-1"],
            answer_support=AnswerSupport.ANSWER_KEY,
            rubric_support=RubricSupport.EXACT_MATCH,
            answers=[EvidenceAnswerDecision(question_id="question-1", answer="B")],
            status=EvidenceDecisionStatus.CANDIDATE,
            confidence=0.5,
            evidence=["ambiguous tick"],
            requires_review=True,
        )


def test_unconfirmed_independent_review_blocks_master_ready(tmp_path: Path) -> None:
    ingested, mapped, config, graph_sha256, question, version, region = _mapped_case(
        tmp_path
    )
    evidence = _empty_evidence(mapped, graph_sha256)
    prepared = prepare_solver_inputs(
        ingested.case_dir,
        evidence,
        _reconstruction_decisions(
            mapped, graph_sha256, question, version, region
        ),
        _crop_decisions(mapped, graph_sha256, question, version, region),
        config,
    )
    decisions = _master_decisions(prepared.manifest, graph_sha256, solved=True)
    reviews = _review_decisions(prepared.manifest, decisions, solved=True)
    reviews.reviews[0].status = IndependentReviewStatus.UNCERTAIN
    with pytest.raises(ReviewRequiredError, match="Exam Master requires review"):
        build_exam_master(
            ingested.case_dir,
            evidence,
            prepared.manifest,
            decisions,
            reviews,
            config,
        )
