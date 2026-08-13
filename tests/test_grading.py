from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest

from student_analyze.assets import artifact_digest
from student_analyze.atomic import serialize_model
from student_analyze.document_models import DecisionMethod
from student_analyze.errors import CaseValidationError, ReviewRequiredError
from student_analyze.exam_master_models import QuestionType
from student_analyze.grading import (
    build_grading,
    finalize_grading_review,
    prepare_grading_context,
    prepare_grading_review,
)
from student_analyze.grading_models import (
    AcademicErrorDiagnosis,
    AcademicErrorType,
    GradingDecisionSet,
    GradingMethod,
    GradingPhase,
    GradingReviewDecisionSet,
    GradingReviewStatus,
    GradingRoute,
    GradingTargetDecision,
    ReviewedTargetDecision,
    RubricEvaluation,
    RubricEvaluationStatus,
)
from student_analyze.models import PipelineStage
from student_analyze.pipeline import verify_case
from student_analyze.submission import (
    build_submission,
    prepare_submission_inputs,
    prepare_submission_structure,
)
from student_analyze.submission_models import SubmissionTranscriptionDecision
from tests.test_submission import (
    _mapping_set,
    _provenance,
    _ready_case,
    _transcriptions,
)


def _submission_ready(
    tmp_path: Path,
    *,
    observed_content: str | None = "A",
    is_blank: bool = False,
    question_type: QuestionType = QuestionType.OBJECTIVE_SINGLE,
    teacher_annotation: bool = False,
):
    case_dir, config, question, version_id, answer_region, answer_page = _ready_case(
        tmp_path,
        teacher_annotation=teacher_annotation,
        question_type=question_type,
    )
    structure = prepare_submission_structure(case_dir, config).manifest
    mappings = _mapping_set(
        structure,
        question,
        version_id,
        answer_region,
        answer_page,
        teacher=teacher_annotation,
    )
    inputs = prepare_submission_inputs(case_dir, structure, mappings, config)
    decision = SubmissionTranscriptionDecision(
        mapping_ref="q1-formal-answer",
        observed_content=None if is_blank else observed_content,
        normalized_answer=None if is_blank else observed_content,
        is_blank=is_blank,
        has_erasure=False,
        confidence=1,
        evidence=["reviewed grading fixture"],
    )
    submission = build_submission(
        case_dir,
        inputs.manifest,
        _transcriptions(inputs.manifest, [decision]),
        config,
    )
    return case_dir, config, submission


def _grading_decisions(manifest, decisions):
    return GradingDecisionSet(
        case_id=manifest.case_id,
        grading_input_manifest_sha256=sha256(serialize_model(manifest)).hexdigest(),
        provenance=_provenance(
            prompt="grading-v1.0.0", skill="exam-grading-v1.0.0"
        ),
        decisions=decisions,
    )


def _source_grading_sha256(case_dir: Path) -> str:
    _, state = verify_case(case_dir)
    completion = next(
        item for item in state.completed_stages if item.stage == PipelineStage.GRADED
    )
    reference = next(
        item for item in completion.artifacts if item.schema_id == "grading.schema.json"
    )
    return artifact_digest(case_dir / reference.relative_path)[0]


def test_objective_is_scored_deterministically_and_reviewed(tmp_path: Path) -> None:
    case_dir, config, _ = _submission_ready(tmp_path, observed_content="A")
    context = prepare_grading_context(case_dir, config)

    assert context.manifest.targets[0].route == GradingRoute.AUTO_OBJECTIVE
    assert context.manifest.llm_target_ids == []
    built = build_grading(
        case_dir,
        context.manifest,
        _grading_decisions(context.manifest, []),
        config,
    )

    assert built.state.current_stage == PipelineStage.GRADED
    assert built.grading.phase == GradingPhase.GRADED
    assert built.grading.provisional_score == 1
    assert built.grading.targets[0].method == GradingMethod.AUTO_OBJECTIVE
    assert built.grading.questions[0].proposed_score == 1
    assert not built.grading.requires_review

    review = GradingReviewDecisionSet(
        case_id=built.grading.case_id,
        source_grading_sha256=_source_grading_sha256(case_dir),
        provenance=_provenance(
            prompt="grading-human-review-v1.0.0",
            skill="exam-grading-v1.0.0",
        ).model_copy(update={"method": DecisionMethod.HUMAN_REVIEW}),
        decisions=[],
    )
    finalized = finalize_grading_review(case_dir, review, config)

    assert finalized.state.current_stage == PipelineStage.REVIEWED
    assert finalized.grading.phase == GradingPhase.REVIEWED
    assert finalized.grading.final_score == 1
    assert finalized.grading.targets[0].final_score == 1
    _, verified = verify_case(case_dir)
    assert verified.current_stage == PipelineStage.REVIEWED


def test_wrong_objective_has_only_generic_objective_error(tmp_path: Path) -> None:
    case_dir, config, _ = _submission_ready(tmp_path, observed_content="B")
    context = prepare_grading_context(case_dir, config)
    built = build_grading(
        case_dir,
        context.manifest,
        _grading_decisions(context.manifest, []),
        config,
    )

    result = built.grading.targets[0]
    assert result.proposed_score == 0
    assert [item.error_type for item in result.error_diagnoses] == [
        AcademicErrorType.INCORRECT_OBJECTIVE
    ]


def test_blank_is_scored_by_python_without_llm(tmp_path: Path) -> None:
    case_dir, config, _ = _submission_ready(tmp_path, is_blank=True)
    context = prepare_grading_context(case_dir, config)

    assert context.manifest.targets[0].route == GradingRoute.AUTO_BLANK
    built = build_grading(
        case_dir,
        context.manifest,
        _grading_decisions(context.manifest, []),
        config,
    )

    assert built.grading.targets[0].proposed_score == 0
    assert built.grading.targets[0].error_diagnoses[0].error_type == (
        AcademicErrorType.UNANSWERED
    )


def test_nonblank_subjective_target_requires_llm_rubric_decision(
    tmp_path: Path,
) -> None:
    case_dir, config, _ = _submission_ready(
        tmp_path,
        observed_content="partially correct explanation",
        question_type=QuestionType.SHORT_ANSWER,
    )
    context = prepare_grading_context(case_dir, config)
    target = context.manifest.targets[0]
    rubric_ref = target.rubric[0].ref
    evidence_id = target.responses[0].submission_item_id
    decisions = _grading_decisions(
        context.manifest,
        [
            GradingTargetDecision(
                target_id=target.target_id,
                rubric_evaluations=[
                    RubricEvaluation(
                        rubric_ref=rubric_ref,
                        status=RubricEvaluationStatus.PARTIALLY_MET,
                        awarded_points=0.5,
                        evidence_item_ids=[evidence_id],
                        rationale="The response contains only part of the required explanation.",
                    )
                ],
                error_diagnoses=[
                    AcademicErrorDiagnosis(
                        error_type=AcademicErrorType.REASONING_OMISSION,
                        rubric_refs=[rubric_ref],
                        evidence_item_ids=[evidence_id],
                        diagnosis="A required reasoning step is absent.",
                    )
                ],
                confidence=0.9,
            )
        ],
    )

    assert target.route == GradingRoute.LLM_RUBRIC
    built = build_grading(case_dir, context.manifest, decisions, config)
    assert built.grading.provisional_score == 0.5
    assert built.grading.targets[0].method == GradingMethod.LLM_RUBRIC
    assert not built.grading.requires_review


def test_undetermined_llm_result_generates_review_packet_and_human_override(
    tmp_path: Path,
) -> None:
    case_dir, config, _ = _submission_ready(
        tmp_path,
        observed_content="ambiguous response",
        question_type=QuestionType.SHORT_ANSWER,
    )
    context = prepare_grading_context(case_dir, config)
    target = context.manifest.targets[0]
    rubric_ref = target.rubric[0].ref
    evidence_id = target.responses[0].submission_item_id
    decisions = _grading_decisions(
        context.manifest,
        [
            GradingTargetDecision(
                target_id=target.target_id,
                rubric_evaluations=[
                    RubricEvaluation(
                        rubric_ref=rubric_ref,
                        status=RubricEvaluationStatus.UNDETERMINED,
                        rationale="The response does not support a stable rubric decision.",
                    )
                ],
                confidence=0.7,
                requires_review=True,
                review_reasons=["Partial-credit boundary is ambiguous."],
            )
        ],
    )
    built = build_grading(case_dir, context.manifest, decisions, config)

    assert built.grading.provisional_score is None
    assert built.grading.requires_review
    assert built.grading.targets[0].review_status == GradingReviewStatus.REQUIRED
    packet = prepare_grading_review(case_dir)
    assert packet.manifest_path.is_file()
    assert packet.html_path.is_file()
    assert len(packet.manifest.items) == 1
    html = packet.html_path.read_text(encoding="utf-8")
    assert "Editable review decision JSON" in html
    assert "Download review JSON" in html

    review = GradingReviewDecisionSet(
        case_id=built.grading.case_id,
        source_grading_sha256=_source_grading_sha256(case_dir),
        provenance=_provenance(
            prompt="grading-human-review-v1.0.0",
            skill="exam-grading-v1.0.0",
        ).model_copy(update={"method": DecisionMethod.HUMAN_REVIEW}),
        decisions=[
            ReviewedTargetDecision(
                target_id=target.target_id,
                rubric_evaluations=[
                    RubricEvaluation(
                        rubric_ref=rubric_ref,
                        status=RubricEvaluationStatus.MET,
                        awarded_points=1,
                        evidence_item_ids=[evidence_id],
                        rationale="Human review confirms the required answer is present.",
                    )
                ],
                decision_reason="Reviewed against the original response crop.",
            )
        ],
    )
    finalized = finalize_grading_review(case_dir, review, config)

    assert finalized.grading.final_score == 1
    assert finalized.grading.targets[0].method == GradingMethod.HUMAN_REVIEW
    assert finalized.grading.targets[0].review_status == GradingReviewStatus.RESOLVED


def test_phase_6_rejects_unresolved_phase_5_submission(tmp_path: Path) -> None:
    case_dir, config, built = _submission_ready(
        tmp_path,
        observed_content="B",
        teacher_annotation=True,
    )
    assert built.submission.requires_review

    with pytest.raises(ReviewRequiredError, match="completed in phase 5"):
        prepare_grading_context(case_dir, config)


def test_every_deduction_requires_an_academic_error_diagnosis(tmp_path: Path) -> None:
    case_dir, config, _ = _submission_ready(
        tmp_path,
        observed_content="wrong explanation",
        question_type=QuestionType.SHORT_ANSWER,
    )
    context = prepare_grading_context(case_dir, config)
    target = context.manifest.targets[0]
    decisions = _grading_decisions(
        context.manifest,
        [
            GradingTargetDecision(
                target_id=target.target_id,
                rubric_evaluations=[
                    RubricEvaluation(
                        rubric_ref=target.rubric[0].ref,
                        status=RubricEvaluationStatus.NOT_MET,
                        awarded_points=0,
                        evidence_item_ids=[target.responses[0].submission_item_id],
                        rationale="The required answer is absent.",
                    )
                ],
                confidence=1,
            )
        ],
    )

    with pytest.raises(CaseValidationError, match="academic error diagnosis"):
        build_grading(case_dir, context.manifest, decisions, config)
