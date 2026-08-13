"""Phase 6 deterministic routing, rubric grading, and human review."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from html import escape
from pathlib import Path
import json
import re
import unicodedata

from student_analyze import __version__
from student_analyze.assets import artifact_digest
from student_analyze.atomic import (
    InterruptHook,
    atomic_write_bytes,
    atomic_write_model,
    serialize_model,
)
from student_analyze.config import AppConfig
from student_analyze.errors import (
    CaseValidationError,
    InvalidTransitionError,
    ReviewRequiredError,
)
from student_analyze.exam_master import read_exam_master
from student_analyze.exam_master_models import (
    ApprovalStatus,
    ExamMaster,
    QuestionType,
)
from student_analyze.fingerprint import digest_value
from student_analyze.grading_models import (
    AcademicErrorDiagnosis,
    AcademicErrorType,
    Grading,
    GradingDecisionSet,
    GradingInputManifest,
    GradingMethod,
    GradingPhase,
    GradingResponseEvidence,
    GradingReviewContextItem,
    GradingReviewDecisionSet,
    GradingReviewItem,
    GradingReviewManifest,
    GradingReviewStatus,
    GradingRoute,
    GradingTargetDecision,
    GradingTargetInput,
    GradingTargetResult,
    QuestionScoreSummary,
    ReviewedTargetDecision,
    RubricEvaluation,
    RubricEvaluationStatus,
)
from student_analyze.models import (
    ImplementationVersions,
    PipelineStage,
    PipelineState,
    SCHEMA_VERSION,
)
from student_analyze.pipeline import (
    build_stage_fingerprint,
    commit_stage_artifact,
    verify_case,
)
from student_analyze.submission import read_submission
from student_analyze.submission_models import (
    Submission,
    SubmissionItem,
    SubmissionSourceRole,
)
from student_analyze.validation import validate_json


@dataclass(frozen=True, slots=True)
class GradingContextPreparationResult:
    case_dir: Path
    manifest_path: Path
    manifest: GradingInputManifest
    reused: bool


@dataclass(frozen=True, slots=True)
class GradingBuildResult:
    case_dir: Path
    grading: Grading
    state: PipelineState
    reused: bool


@dataclass(frozen=True, slots=True)
class GradingReviewPreparationResult:
    case_dir: Path
    manifest_path: Path
    html_path: Path
    manifest: GradingReviewManifest
    reused: bool


def load_grading_input_manifest(path: Path) -> GradingInputManifest:
    return _read_model_file(path, GradingInputManifest, "grading input manifest")


def load_grading_decisions(path: Path) -> GradingDecisionSet:
    return _read_model_file(path, GradingDecisionSet, "grading decisions")


def read_grading(path: Path) -> Grading:
    return _read_model_file(path, Grading, "grading artifact")


def load_grading_review_decisions(path: Path) -> GradingReviewDecisionSet:
    return _read_model_file(path, GradingReviewDecisionSet, "grading review decisions")


def read_grading_review_manifest(path: Path) -> GradingReviewManifest:
    return _read_model_file(path, GradingReviewManifest, "grading review manifest")


def prepare_grading_context(
    case_dir: Path,
    config: AppConfig,
    *,
    interrupt_hook: InterruptHook | None = None,
) -> GradingContextPreparationResult:
    """Build the complete, reviewed context used for automatic and LLM grading."""

    case_dir = case_dir.resolve(strict=True)
    _, state = verify_case(case_dir)
    master_path, master = _active_exam_master(case_dir, state)
    submission_path, submission = _active_submission(case_dir, state)
    _reject_unresolved_inputs(master, submission)
    master_sha256, _ = artifact_digest(master_path)
    submission_sha256, _ = artifact_digest(submission_path)
    fingerprint = digest_value(
        {
            "exam_master_sha256": master_sha256,
            "submission_sha256": submission_sha256,
            "config": config.grading_context_fingerprint_payload(),
        }
    )
    output_dir = case_dir / "work" / "grading_context" / fingerprint
    manifest_path = output_dir / "grading_input_manifest.json"

    if manifest_path.exists():
        manifest = load_grading_input_manifest(manifest_path)
        verify_grading_input_manifest(
            manifest,
            master,
            submission,
            exam_master_sha256=master_sha256,
            submission_sha256=submission_sha256,
        )
        return GradingContextPreparationResult(
            case_dir, manifest_path, manifest, reused=True
        )

    targets = _build_grading_targets(master, submission)
    manifest = GradingInputManifest(
        case_id=master.case_id,
        exam_master_sha256=master_sha256,
        submission_sha256=submission_sha256,
        manifest_fingerprint=fingerprint,
        created_at=datetime.now(UTC),
        targets=targets,
        llm_target_ids=[
            item.target_id for item in targets if item.route == GradingRoute.LLM_RUBRIC
        ],
    )
    atomic_write_model(
        manifest_path,
        manifest,
        model_type=GradingInputManifest,
        interrupt_hook=interrupt_hook,
    )
    verify_grading_input_manifest(
        manifest,
        master,
        submission,
        exam_master_sha256=master_sha256,
        submission_sha256=submission_sha256,
    )
    return GradingContextPreparationResult(
        case_dir, manifest_path, manifest, reused=False
    )


def build_grading(
    case_dir: Path,
    input_manifest: GradingInputManifest,
    decisions: GradingDecisionSet,
    config: AppConfig,
    *,
    force: bool = False,
    interrupt_hook: InterruptHook | None = None,
) -> GradingBuildResult:
    """Combine deterministic results and LLM rubric decisions into graded output."""

    case_dir = case_dir.resolve(strict=True)
    _, state = verify_case(case_dir)
    master_path, master = _active_exam_master(case_dir, state)
    submission_path, submission = _active_submission(case_dir, state)
    _reject_unresolved_inputs(master, submission)
    master_sha256, _ = artifact_digest(master_path)
    submission_sha256, _ = artifact_digest(submission_path)
    verify_grading_input_manifest(
        input_manifest,
        master,
        submission,
        exam_master_sha256=master_sha256,
        submission_sha256=submission_sha256,
    )
    input_sha256 = sha256(serialize_model(input_manifest)).hexdigest()
    _validate_grading_decisions(
        decisions,
        input_manifest,
        input_manifest_sha256=input_sha256,
    )
    decision_sha256 = digest_value(decisions.model_dump(mode="json"))
    versions = ImplementationVersions(
        code=__version__,
        base_schema=SCHEMA_VERSION,
        config=config.config_version,
        model=decisions.provenance.model_identifier,
        prompt=decisions.provenance.prompt_version,
        skill=decisions.provenance.skill_version,
    )
    stage_inputs = [
        {
            "exam_master_sha256": master_sha256,
            "submission_sha256": submission_sha256,
            "grading_input_manifest_sha256": input_sha256,
            "grading_decision_sha256": decision_sha256,
        }
    ]
    stage_config = config.grading_fingerprint_payload()
    stage_fingerprint, _ = build_stage_fingerprint(
        stage=PipelineStage.GRADED,
        model_type=Grading,
        schema_id="grading.schema.json",
        config=stage_config,
        versions=versions,
        inputs=stage_inputs,
    )
    target_results = _compile_graded_results(input_manifest, decisions)
    provisional_scores = [item.proposed_score for item in target_results]
    review_items = [
        GradingReviewItem(target_id=item.target_id, reasons=[item.review_reason])
        for item in target_results
        if item.review_status == GradingReviewStatus.REQUIRED
        and item.review_reason is not None
    ]
    grading = Grading(
        phase=GradingPhase.GRADED,
        case_id=master.case_id,
        exam_master_sha256=master_sha256,
        submission_sha256=submission_sha256,
        grading_input_manifest_fingerprint=input_manifest.manifest_fingerprint,
        grading_input_manifest_sha256=input_sha256,
        grading_decision_sha256=decision_sha256,
        stage_fingerprint=stage_fingerprint,
        created_at=datetime.now(UTC),
        grading_provenance=decisions.provenance,
        input_manifest=input_manifest,
        targets=target_results,
        questions=_build_question_summaries(target_results),
        provisional_score=(
            None
            if any(item is None for item in provisional_scores)
            else sum(item for item in provisional_scores if item is not None)
        ),
        max_score=sum(item.max_points for item in target_results),
        review_items=review_items,
        requires_review=bool(review_items),
    )
    verify_grading(
        grading,
        master,
        submission,
        exam_master_sha256=master_sha256,
        submission_sha256=submission_sha256,
    )
    grading = _reuse_or_validate_orphan(
        case_dir,
        PipelineStage.GRADED,
        stage_fingerprint,
        grading,
    )
    committed = commit_stage_artifact(
        case_dir,
        stage=PipelineStage.GRADED,
        artifact_name="grading.json",
        payload=grading,
        model_type=Grading,
        schema_id="grading.schema.json",
        config=stage_config,
        versions=versions,
        inputs=stage_inputs,
        force=force,
        interrupt_hook=interrupt_hook,
    )
    reference = committed.artifacts[0]
    active = read_grading(case_dir / reference.relative_path)
    return GradingBuildResult(case_dir, active, committed.state, committed.reused)


def prepare_grading_review(
    case_dir: Path,
    *,
    interrupt_hook: InterruptHook | None = None,
) -> GradingReviewPreparationResult:
    """Render a local, static review packet for required grading decisions."""

    case_dir = case_dir.resolve(strict=True)
    manifest, state = verify_case(case_dir)
    grading_path, grading = _active_grading(case_dir, state, PipelineStage.GRADED)
    grading_sha256, _ = artifact_digest(grading_path)
    review_by_id = {item.target_id: item for item in grading.review_items}
    input_by_id = {item.target_id: item for item in grading.input_manifest.targets}
    result_by_id = {item.target_id: item for item in grading.targets}
    items = [
        GradingReviewContextItem(
            target=input_by_id[item.target_id],
            proposed_result=result_by_id[item.target_id],
            reasons=item.reasons,
        )
        for item in grading.review_items
    ]
    review_manifest = GradingReviewManifest(
        case_id=grading.case_id,
        source_grading_sha256=grading_sha256,
        created_at=datetime.now(UTC),
        items=items,
    )
    output_dir = case_dir / "work" / "grading_review" / grading_sha256
    manifest_path = output_dir / "grading_review_manifest.json"
    html_path = output_dir / "index.html"
    reused = manifest_path.exists() and html_path.exists()
    if reused:
        persisted = read_grading_review_manifest(manifest_path)
        if _review_manifest_signature(persisted) != _review_manifest_signature(
            review_manifest
        ):
            raise CaseValidationError(
                f"existing grading review manifest conflicts with {grading_path}"
            )
        review_manifest = persisted
    else:
        atomic_write_model(
            manifest_path,
            review_manifest,
            model_type=GradingReviewManifest,
            interrupt_hook=interrupt_hook,
        )
        html = _render_review_html(case_dir, manifest.source_root, review_manifest)
        atomic_write_bytes(
            html_path,
            html.encode("utf-8"),
            validator=_validate_html,
            interrupt_hook=interrupt_hook,
        )
    return GradingReviewPreparationResult(
        case_dir,
        manifest_path,
        html_path,
        review_manifest,
        reused,
    )


def finalize_grading_review(
    case_dir: Path,
    review_decisions: GradingReviewDecisionSet,
    config: AppConfig,
    *,
    force: bool = False,
    interrupt_hook: InterruptHook | None = None,
) -> GradingBuildResult:
    """Apply complete human decisions and commit the reviewed grading artifact."""

    case_dir = case_dir.resolve(strict=True)
    _, state = verify_case(case_dir)
    master_path, master = _active_exam_master(case_dir, state)
    submission_path, submission = _active_submission(case_dir, state)
    grading_path, source = _active_grading(case_dir, state, PipelineStage.GRADED)
    master_sha256, _ = artifact_digest(master_path)
    submission_sha256, _ = artifact_digest(submission_path)
    source_sha256, _ = artifact_digest(grading_path)
    _validate_review_decisions(review_decisions, source, source_sha256=source_sha256)
    review_sha256 = digest_value(review_decisions.model_dump(mode="json"))
    versions = ImplementationVersions(
        code=__version__,
        base_schema=SCHEMA_VERSION,
        config=config.config_version,
        model=review_decisions.provenance.model_identifier,
        prompt=review_decisions.provenance.prompt_version,
        skill=review_decisions.provenance.skill_version,
    )
    stage_inputs = [
        {
            "source_grading_sha256": source_sha256,
            "review_decision_sha256": review_sha256,
        }
    ]
    stage_config = config.grading_review_fingerprint_payload()
    stage_fingerprint, _ = build_stage_fingerprint(
        stage=PipelineStage.REVIEWED,
        model_type=Grading,
        schema_id="grading.schema.json",
        config=stage_config,
        versions=versions,
        inputs=stage_inputs,
    )
    reviewed_targets = _apply_review_decisions(source, review_decisions)
    grading = Grading(
        phase=GradingPhase.REVIEWED,
        case_id=source.case_id,
        exam_master_sha256=source.exam_master_sha256,
        submission_sha256=source.submission_sha256,
        grading_input_manifest_fingerprint=source.grading_input_manifest_fingerprint,
        grading_input_manifest_sha256=source.grading_input_manifest_sha256,
        grading_decision_sha256=source.grading_decision_sha256,
        source_grading_sha256=source_sha256,
        review_decision_sha256=review_sha256,
        stage_fingerprint=stage_fingerprint,
        created_at=datetime.now(UTC),
        grading_provenance=source.grading_provenance,
        review_provenance=review_decisions.provenance,
        input_manifest=source.input_manifest,
        targets=reviewed_targets,
        questions=_build_question_summaries(reviewed_targets),
        provisional_score=source.provisional_score,
        final_score=sum(item.final_score or 0 for item in reviewed_targets),
        max_score=source.max_score,
        review_items=[],
        requires_review=False,
        warnings=source.warnings,
    )
    verify_grading(
        grading,
        master,
        submission,
        exam_master_sha256=master_sha256,
        submission_sha256=submission_sha256,
        source_grading=source,
        source_grading_sha256=source_sha256,
    )
    grading = _reuse_or_validate_orphan(
        case_dir,
        PipelineStage.REVIEWED,
        stage_fingerprint,
        grading,
    )
    committed = commit_stage_artifact(
        case_dir,
        stage=PipelineStage.REVIEWED,
        artifact_name="grading.json",
        payload=grading,
        model_type=Grading,
        schema_id="grading.schema.json",
        config=stage_config,
        versions=versions,
        inputs=stage_inputs,
        force=force,
        human_confirmed=True,
        interrupt_hook=interrupt_hook,
    )
    reference = committed.artifacts[0]
    active = read_grading(case_dir / reference.relative_path)
    return GradingBuildResult(case_dir, active, committed.state, committed.reused)


def verify_grading_input_manifest(
    manifest: GradingInputManifest,
    master: ExamMaster,
    submission: Submission,
    *,
    exam_master_sha256: str,
    submission_sha256: str,
) -> None:
    _reject_unresolved_inputs(master, submission)
    if manifest.case_id != master.case_id or manifest.case_id != submission.case_id:
        raise CaseValidationError("grading input belongs to another case")
    if manifest.exam_master_sha256 != exam_master_sha256:
        raise CaseValidationError("grading input references another Exam Master")
    if manifest.submission_sha256 != submission_sha256:
        raise CaseValidationError("grading input references another Submission")
    expected = _build_grading_targets(master, submission)
    if manifest.targets != expected:
        raise CaseValidationError("grading input does not match active master and submission")


def verify_grading(
    grading: Grading,
    master: ExamMaster,
    submission: Submission,
    *,
    exam_master_sha256: str,
    submission_sha256: str,
    source_grading: Grading | None = None,
    source_grading_sha256: str | None = None,
) -> None:
    if grading.case_id != master.case_id or grading.case_id != submission.case_id:
        raise CaseValidationError("grading artifact belongs to another case")
    if grading.exam_master_sha256 != exam_master_sha256:
        raise CaseValidationError("grading artifact references another Exam Master")
    if grading.submission_sha256 != submission_sha256:
        raise CaseValidationError("grading artifact references another Submission")
    embedded_sha256 = sha256(serialize_model(grading.input_manifest)).hexdigest()
    if embedded_sha256 != grading.grading_input_manifest_sha256:
        raise CaseValidationError("embedded grading input hash is invalid")
    if (
        grading.input_manifest.manifest_fingerprint
        != grading.grading_input_manifest_fingerprint
    ):
        raise CaseValidationError("embedded grading input fingerprint is invalid")
    verify_grading_input_manifest(
        grading.input_manifest,
        master,
        submission,
        exam_master_sha256=exam_master_sha256,
        submission_sha256=submission_sha256,
    )
    inputs = {item.target_id: item for item in grading.input_manifest.targets}
    results = {item.target_id: item for item in grading.targets}
    if set(inputs) != set(results):
        raise CaseValidationError("grading results must cover every input target")
    for target_id, result in results.items():
        target = inputs[target_id]
        if (
            result.question_id != target.question_id
            or result.version_id != target.version_id
            or result.part_id != target.part_id
            or result.submission_item_ids
            != [item.submission_item_id for item in target.responses]
            or result.max_points != target.max_points
        ):
            raise CaseValidationError(f"grading result changes target identity: {target_id}")
        _validate_evaluations(
            result.rubric_evaluations,
            result.error_diagnoses,
            target,
            allow_undetermined=(grading.phase == GradingPhase.GRADED),
        )
        if grading.phase == GradingPhase.GRADED:
            expected_method = {
                GradingRoute.AUTO_OBJECTIVE: GradingMethod.AUTO_OBJECTIVE,
                GradingRoute.AUTO_BLANK: GradingMethod.AUTO_BLANK,
                GradingRoute.LLM_RUBRIC: GradingMethod.LLM_RUBRIC,
            }[target.route]
            if result.method != expected_method:
                raise CaseValidationError(f"grading method differs from route: {target_id}")

    if grading.phase == GradingPhase.REVIEWED:
        if source_grading is None or source_grading_sha256 is None:
            raise CaseValidationError("reviewed grading verification needs source grading")
        if grading.source_grading_sha256 != source_grading_sha256:
            raise CaseValidationError("reviewed grading references another source grading")
        if grading.input_manifest != source_grading.input_manifest:
            raise CaseValidationError("review cannot change the grading input manifest")
        source_by_id = {item.target_id: item for item in source_grading.targets}
        for result in grading.targets:
            previous = source_by_id[result.target_id]
            if previous.review_status == GradingReviewStatus.REQUIRED:
                if result.review_status != GradingReviewStatus.RESOLVED:
                    raise CaseValidationError("required grading review was not resolved")
            elif (
                result.method != previous.method
                or result.rubric_evaluations != previous.rubric_evaluations
                or result.error_diagnoses != previous.error_diagnoses
                or result.proposed_score != previous.proposed_score
                or result.review_status != GradingReviewStatus.NOT_REQUIRED
            ):
                raise CaseValidationError("human review changed an unflagged grading target")


def _build_grading_targets(
    master: ExamMaster, submission: Submission
) -> list[GradingTargetInput]:
    formal = [
        item
        for item in submission.items
        if item.source_role == SubmissionSourceRole.ANSWER_SHEET
    ]
    grouped: dict[tuple[str, str | None], list[SubmissionItem]] = {}
    for item in formal:
        grouped.setdefault((item.question_id, item.part_id), []).append(item)
    targets: list[GradingTargetInput] = []
    consumed_rubric_refs: set[str] = set()
    for question in master.questions:
        parts = question.subparts or [None]
        for part in parts:
            part_id = part.part_id if part is not None else None
            key = (question.question_id, part_id)
            items = sorted(grouped.get(key, []), key=lambda item: item.slot_order)
            if not items:
                raise CaseValidationError(
                    f"grading target has no formal submission evidence: {key}"
                )
            answers = [
                item
                for item in question.reference_answers
                if item.part_ref == part_id
            ]
            rubric = [item for item in question.rubric if item.part_ref == part_id]
            if not answers or not rubric:
                raise CaseValidationError(
                    f"grading target lacks an approved answer or rubric: {key}"
                )
            if any(item.points is None for item in rubric):
                raise CaseValidationError(
                    f"grading target has rubric criteria without points: {key}"
                )
            consumed_rubric_refs.update(item.ref for item in rubric)
            responses = [
                GradingResponseEvidence(
                    submission_item_id=item.item_id,
                    slot_label=item.slot_label,
                    observed_content=item.observed_content,
                    normalized_answer=item.normalized_answer,
                    is_blank=item.is_blank,
                    crop=item.crop,
                )
                for item in items
            ]
            all_blank = all(item.is_blank for item in items)
            objective = question.question_type in {
                QuestionType.OBJECTIVE_SINGLE,
                QuestionType.OBJECTIVE_MULTIPLE,
            }
            route = (
                GradingRoute.AUTO_BLANK
                if all_blank
                else GradingRoute.AUTO_OBJECTIVE
                if objective
                else GradingRoute.LLM_RUBRIC
            )
            targets.append(
                GradingTargetInput(
                    target_id=_stable_id(
                        "grading-target", question.question_id, part_id or "root"
                    ),
                    question_id=question.question_id,
                    version_id=question.version_id,
                    part_id=part_id,
                    printed_label=(
                        question.printed_label
                        if part is None
                        else f"{question.printed_label}{part.printed_label}"
                    ),
                    prompt_text=(question.prompt_text if part is None else part.prompt_text),
                    question_type=question.question_type,
                    max_points=sum(item.points or 0 for item in rubric),
                    options=question.options,
                    reference_answers=answers,
                    rubric=rubric,
                    solution_summary=question.solution_summary,
                    assumptions=question.assumptions,
                    knowledge_points=question.knowledge_points,
                    responses=responses,
                    route=route,
                    mandatory_review=(question.question_type == QuestionType.DIAGRAM),
                )
            )
    all_rubric_refs = {item.ref for question in master.questions for item in question.rubric}
    if consumed_rubric_refs != all_rubric_refs:
        raise CaseValidationError("some Exam Master rubric criteria cannot be routed")
    if set(grouped) != {
        (item.question_id, item.part_id)
        for item in targets
    }:
        raise CaseValidationError("formal Submission contains an unknown grading target")
    return targets


def _compile_graded_results(
    manifest: GradingInputManifest, decisions: GradingDecisionSet
) -> list[GradingTargetResult]:
    decisions_by_id = {item.target_id: item for item in decisions.decisions}
    results: list[GradingTargetResult] = []
    for target in manifest.targets:
        if target.route == GradingRoute.AUTO_BLANK:
            results.append(_grade_blank(target))
        elif target.route == GradingRoute.AUTO_OBJECTIVE:
            results.append(_grade_objective(target))
        else:
            results.append(_compile_llm_result(target, decisions_by_id[target.target_id]))
    return results


def _build_question_summaries(
    targets: list[GradingTargetResult],
) -> list[QuestionScoreSummary]:
    grouped: dict[str, list[GradingTargetResult]] = {}
    for target in targets:
        grouped.setdefault(target.question_id, []).append(target)
    summaries: list[QuestionScoreSummary] = []
    for question_id, items in grouped.items():
        proposed = [item.proposed_score for item in items]
        final = [item.final_score for item in items]
        summaries.append(
            QuestionScoreSummary(
                question_id=question_id,
                target_ids=[item.target_id for item in items],
                proposed_score=(
                    None if any(item is None for item in proposed) else sum(proposed)
                ),
                final_score=None if any(item is None for item in final) else sum(final),
                max_score=sum(item.max_points for item in items),
            )
        )
    return summaries


def _grade_blank(target: GradingTargetInput) -> GradingTargetResult:
    evidence_ids = [item.submission_item_id for item in target.responses]
    evaluations = [
        RubricEvaluation(
            rubric_ref=criterion.ref,
            status=RubricEvaluationStatus.NOT_MET,
            awarded_points=0,
            evidence_item_ids=evidence_ids,
            rationale="All formal response slots for this target are blank.",
        )
        for criterion in target.rubric
    ]
    diagnosis = AcademicErrorDiagnosis(
        error_type=AcademicErrorType.UNANSWERED,
        rubric_refs=[item.ref for item in target.rubric],
        evidence_item_ids=evidence_ids,
        diagnosis="The reviewed formal response is blank.",
    )
    return GradingTargetResult(
        target_id=target.target_id,
        question_id=target.question_id,
        version_id=target.version_id,
        part_id=target.part_id,
        submission_item_ids=evidence_ids,
        method=GradingMethod.AUTO_BLANK,
        rubric_evaluations=evaluations,
        error_diagnoses=[diagnosis],
        proposed_score=0,
        max_points=target.max_points,
        confidence=1,
        review_status=GradingReviewStatus.NOT_REQUIRED,
    )


def _grade_objective(target: GradingTargetInput) -> GradingTargetResult:
    labels = [item.label for item in target.options]
    student_text = "".join(
        (item.normalized_answer or item.observed_content or "")
        for item in target.responses
        if not item.is_blank
    )
    student = _canonical_option_answer(student_text, labels)
    acceptable = {
        parsed
        for answer in target.reference_answers
        for candidate in [answer.answer, *answer.acceptable_alternatives]
        if (parsed := _canonical_option_answer(candidate, labels)) is not None
    }
    if not acceptable:
        raise CaseValidationError(
            f"objective target has no machine-readable reference answer: {target.target_id}"
        )
    correct = student in acceptable
    evidence_ids = [item.submission_item_id for item in target.responses]
    status = (
        RubricEvaluationStatus.MET if correct else RubricEvaluationStatus.NOT_MET
    )
    evaluations = [
        RubricEvaluation(
            rubric_ref=criterion.ref,
            status=status,
            awarded_points=(criterion.points or 0) if correct else 0,
            evidence_item_ids=evidence_ids,
            rationale=(
                "The normalized selected option set exactly matches the approved answer."
                if correct
                else "The normalized selected option set does not match the approved answer."
            ),
        )
        for criterion in target.rubric
    ]
    errors = []
    if not correct:
        errors.append(
            AcademicErrorDiagnosis(
                error_type=AcademicErrorType.INCORRECT_OBJECTIVE,
                rubric_refs=[item.ref for item in target.rubric],
                evidence_item_ids=evidence_ids,
                diagnosis="The selected option set is incorrect; no deeper cause is inferred.",
            )
        )
    return GradingTargetResult(
        target_id=target.target_id,
        question_id=target.question_id,
        version_id=target.version_id,
        part_id=target.part_id,
        submission_item_ids=evidence_ids,
        method=GradingMethod.AUTO_OBJECTIVE,
        rubric_evaluations=evaluations,
        error_diagnoses=errors,
        proposed_score=target.max_points if correct else 0,
        max_points=target.max_points,
        confidence=1,
        review_status=GradingReviewStatus.NOT_REQUIRED,
    )


def _compile_llm_result(
    target: GradingTargetInput, decision: GradingTargetDecision
) -> GradingTargetResult:
    _validate_evaluations(
        decision.rubric_evaluations,
        decision.error_diagnoses,
        target,
        allow_undetermined=True,
    )
    undetermined = any(
        item.status == RubricEvaluationStatus.UNDETERMINED
        for item in decision.rubric_evaluations
    )
    proposed_score = (
        None
        if undetermined
        else sum(item.awarded_points or 0 for item in decision.rubric_evaluations)
    )
    reasons = list(decision.review_reasons)
    if target.mandatory_review:
        reasons.append("Diagram responses require human score review.")
    if decision.visual_judgment_required:
        reasons.append("The score depends on direct visual judgment of the response crop.")
    review_reason = "; ".join(dict.fromkeys(reasons)) or None
    return GradingTargetResult(
        target_id=target.target_id,
        question_id=target.question_id,
        version_id=target.version_id,
        part_id=target.part_id,
        submission_item_ids=[item.submission_item_id for item in target.responses],
        method=GradingMethod.LLM_RUBRIC,
        rubric_evaluations=decision.rubric_evaluations,
        error_diagnoses=decision.error_diagnoses,
        proposed_score=proposed_score,
        max_points=target.max_points,
        confidence=decision.confidence,
        review_status=(
            GradingReviewStatus.REQUIRED
            if review_reason is not None
            else GradingReviewStatus.NOT_REQUIRED
        ),
        review_reason=review_reason,
    )


def _validate_grading_decisions(
    decisions: GradingDecisionSet,
    manifest: GradingInputManifest,
    *,
    input_manifest_sha256: str,
) -> None:
    if decisions.case_id != manifest.case_id:
        raise CaseValidationError("grading decisions belong to another case")
    if decisions.grading_input_manifest_sha256 != input_manifest_sha256:
        raise CaseValidationError("grading decisions reference another input manifest")
    actual = [item.target_id for item in decisions.decisions]
    if actual != manifest.llm_target_ids:
        raise CaseValidationError(
            "grading decisions must cover every LLM target in manifest order"
        )
    targets = {item.target_id: item for item in manifest.targets}
    for decision in decisions.decisions:
        _validate_evaluations(
            decision.rubric_evaluations,
            decision.error_diagnoses,
            targets[decision.target_id],
            allow_undetermined=True,
        )


def _validate_evaluations(
    evaluations: list[RubricEvaluation],
    diagnoses: list[AcademicErrorDiagnosis],
    target: GradingTargetInput,
    *,
    allow_undetermined: bool,
) -> None:
    rubric = {item.ref: item for item in target.rubric}
    actual_refs = [item.rubric_ref for item in evaluations]
    if set(actual_refs) != set(rubric) or len(actual_refs) != len(rubric):
        raise CaseValidationError(
            f"rubric evaluations must cover target exactly: {target.target_id}"
        )
    evidence_ids = {item.submission_item_id for item in target.responses}
    deducted: set[str] = set()
    for evaluation in evaluations:
        criterion = rubric[evaluation.rubric_ref]
        maximum = criterion.points or 0
        if not set(evaluation.evidence_item_ids).issubset(evidence_ids):
            raise CaseValidationError("rubric evaluation cites another response")
        if evaluation.status == RubricEvaluationStatus.MET:
            if evaluation.awarded_points != maximum:
                raise CaseValidationError("met rubric criteria must award full points")
        elif evaluation.status == RubricEvaluationStatus.PARTIALLY_MET:
            if (
                evaluation.awarded_points is None
                or evaluation.awarded_points <= 0
                or evaluation.awarded_points >= maximum
            ):
                raise CaseValidationError(
                    "partially-met rubric criteria must award an interior point value"
                )
            if not evaluation.evidence_item_ids:
                raise CaseValidationError(
                    "partially-met rubric criteria must cite student evidence"
                )
            deducted.add(evaluation.rubric_ref)
        elif evaluation.status == RubricEvaluationStatus.NOT_MET:
            if not evaluation.evidence_item_ids:
                raise CaseValidationError(
                    "not-met rubric criteria must cite student evidence"
                )
            deducted.add(evaluation.rubric_ref)
        elif not allow_undetermined:
            raise CaseValidationError("final review cannot retain undetermined rubrics")
    diagnosed: set[str] = set()
    for diagnosis in diagnoses:
        if not set(diagnosis.rubric_refs).issubset(deducted):
            raise CaseValidationError("error diagnosis must cite deducted rubric criteria")
        if not set(diagnosis.evidence_item_ids).issubset(evidence_ids):
            raise CaseValidationError("error diagnosis cites another response")
        diagnosed.update(diagnosis.rubric_refs)
    if deducted != diagnosed:
        raise CaseValidationError(
            "every deducted rubric criterion requires an academic error diagnosis"
        )


def _validate_review_decisions(
    decisions: GradingReviewDecisionSet,
    source: Grading,
    *,
    source_sha256: str,
) -> None:
    if decisions.case_id != source.case_id:
        raise CaseValidationError("grading review decisions belong to another case")
    if decisions.source_grading_sha256 != source_sha256:
        raise CaseValidationError("grading review decisions reference another grading")
    expected = [item.target_id for item in source.review_items]
    actual = [item.target_id for item in decisions.decisions]
    if actual != expected:
        raise CaseValidationError(
            "human review decisions must cover every required target in order"
        )
    targets = {item.target_id: item for item in source.input_manifest.targets}
    for decision in decisions.decisions:
        _validate_evaluations(
            decision.rubric_evaluations,
            decision.error_diagnoses,
            targets[decision.target_id],
            allow_undetermined=False,
        )


def _apply_review_decisions(
    source: Grading, decisions: GradingReviewDecisionSet
) -> list[GradingTargetResult]:
    reviewed = {item.target_id: item for item in decisions.decisions}
    results: list[GradingTargetResult] = []
    for item in source.targets:
        decision: ReviewedTargetDecision | None = reviewed.get(item.target_id)
        if decision is None:
            if item.proposed_score is None:
                raise CaseValidationError("unreviewed grading target has no proposed score")
            results.append(
                GradingTargetResult.model_validate(
                    {
                        **item.model_dump(mode="json"),
                        "final_score": item.proposed_score,
                    }
                )
            )
            continue
        final_score = sum(
            evaluation.awarded_points or 0
            for evaluation in decision.rubric_evaluations
        )
        results.append(
            GradingTargetResult.model_validate(
                {
                    **item.model_dump(mode="json"),
                    "method": GradingMethod.HUMAN_REVIEW,
                    "rubric_evaluations": decision.rubric_evaluations,
                    "error_diagnoses": decision.error_diagnoses,
                    "final_score": final_score,
                    "confidence": 1,
                    "review_status": GradingReviewStatus.RESOLVED,
                    "review_reason": None,
                    "review_note": decision.decision_reason,
                }
            )
        )
    return results


def _canonical_option_answer(
    value: str, option_labels: list[str]
) -> tuple[str, ...] | None:
    normalized_labels = [
        unicodedata.normalize("NFKC", item).strip().upper()
        for item in option_labels
    ]
    text = unicodedata.normalize("NFKC", value).strip().upper()
    if not text:
        return None
    if all(len(item) == 1 for item in normalized_labels):
        compact = re.sub(r"[\s,，、;；/|+（）()\[\]{}]", "", text)
        if not compact or any(char not in normalized_labels for char in compact):
            return None
        if len(compact) != len(set(compact)):
            return None
        return tuple(sorted(compact))
    parts = [
        item
        for item in re.split(r"[\s,，、;；/|+]+", text)
        if item
    ]
    if not parts or any(item not in normalized_labels for item in parts):
        return None
    if len(parts) != len(set(parts)):
        return None
    return tuple(sorted(parts))


def _reject_unresolved_inputs(master: ExamMaster, submission: Submission) -> None:
    if master.requires_review or any(
        item.approval_status != ApprovalStatus.APPROVED for item in master.questions
    ):
        raise ReviewRequiredError(
            "phase 6 requires a fully approved Exam Master before grading"
        )
    if submission.requires_review or any(item.requires_review for item in submission.items):
        raise ReviewRequiredError(
            "phase 6 requires all Submission mapping and transcription review to be "
            "completed in phase 5"
        )


def _active_exam_master(
    case_dir: Path, state: PipelineState
) -> tuple[Path, ExamMaster]:
    completion = next(
        (item for item in state.completed_stages if item.stage == PipelineStage.MASTER_READY),
        None,
    )
    if completion is None:
        raise InvalidTransitionError("phase 6 requires an active Exam Master")
    references = [
        item for item in completion.artifacts if item.schema_id == "exam_master.schema.json"
    ]
    if len(references) != 1:
        raise CaseValidationError("master_ready must have exactly one active Exam Master")
    path = case_dir / references[0].relative_path
    return path, read_exam_master(path)


def _active_submission(
    case_dir: Path, state: PipelineState
) -> tuple[Path, Submission]:
    completion = next(
        (
            item
            for item in state.completed_stages
            if item.stage == PipelineStage.SUBMISSION_READY
        ),
        None,
    )
    if completion is None:
        raise InvalidTransitionError("phase 6 requires an active Submission")
    references = [
        item for item in completion.artifacts if item.schema_id == "submission.schema.json"
    ]
    if len(references) != 1:
        raise CaseValidationError("submission_ready must have exactly one active Submission")
    path = case_dir / references[0].relative_path
    return path, read_submission(path)


def _active_grading(
    case_dir: Path, state: PipelineState, stage: PipelineStage
) -> tuple[Path, Grading]:
    completion = next(
        (item for item in state.completed_stages if item.stage == stage),
        None,
    )
    if completion is None:
        raise InvalidTransitionError(f"phase 6 requires an active {stage.value} artifact")
    references = [
        item for item in completion.artifacts if item.schema_id == "grading.schema.json"
    ]
    if len(references) != 1:
        raise CaseValidationError(f"{stage.value} must have exactly one grading artifact")
    path = case_dir / references[0].relative_path
    return path, read_grading(path)


def _reuse_or_validate_orphan(
    case_dir: Path,
    stage: PipelineStage,
    stage_fingerprint: str,
    candidate: Grading,
) -> Grading:
    path = (
        case_dir
        / "artifacts"
        / stage.value
        / stage_fingerprint
        / "grading.json"
    )
    if not path.exists():
        return candidate
    persisted = read_grading(path)
    candidate_payload = candidate.model_dump(mode="json")
    persisted_payload = persisted.model_dump(mode="json")
    candidate_payload.pop("created_at", None)
    persisted_payload.pop("created_at", None)
    if persisted_payload != candidate_payload:
        raise CaseValidationError(f"conflicting uncommitted grading artifact at {path}")
    return persisted


def _render_review_html(
    case_dir: Path, source_root: str, manifest: GradingReviewManifest
) -> str:
    cards: list[str] = []
    for item in manifest.items:
        target = item.target
        result = item.proposed_result
        response_blocks = []
        for response in target.responses:
            crop_uri = (case_dir / response.crop.relative_path).resolve().as_uri()
            source_uri = (
                Path(source_root) / response.crop.source_relative_path
            ).resolve().as_uri()
            response_blocks.append(
                "<div class='response'>"
                f"<p><strong>{escape(response.slot_label)}</strong>: "
                f"{escape(response.observed_content or '[blank]')}</p>"
                f"<img src='{escape(crop_uri)}' alt='student response crop'>"
                f"<p><a href='{escape(source_uri)}'>Open original image context</a></p>"
                "</div>"
            )
        rubric_rows = "".join(
            "<tr>"
            f"<td>{escape(criterion.ref)}</td>"
            f"<td>{escape(criterion.description)}</td>"
            f"<td>{criterion.points}</td>"
            "</tr>"
            for criterion in target.rubric
        )
        proposed_rows = "".join(
            "<li>"
            f"{escape(evaluation.rubric_ref)}: {escape(evaluation.status.value)}, "
            f"points={evaluation.awarded_points}; {escape(evaluation.rationale)}"
            "</li>"
            for evaluation in result.rubric_evaluations
        )
        cards.append(
            "<section>"
            f"<h2>{escape(target.printed_label)} — {escape(target.target_id)}</h2>"
            f"<p><strong>Review reason:</strong> {escape('; '.join(item.reasons))}</p>"
            f"<p><strong>Question:</strong> {escape(target.prompt_text)}</p>"
            f"<p><strong>Reference answer:</strong> "
            f"{escape(' | '.join(answer.answer for answer in target.reference_answers))}</p>"
            "<table><thead><tr><th>Rubric</th><th>Description</th><th>Points</th>"
            f"</tr></thead><tbody>{rubric_rows}</tbody></table>"
            f"{''.join(response_blocks)}"
            f"<p><strong>Proposed score:</strong> {result.proposed_score} / "
            f"{result.max_points}</p><ul>{proposed_rows}</ul>"
            "</section>"
        )
    body = "".join(cards) or "<p>No grading decisions require human review.</p>"
    decision_template = {
        "schema_version": "1.0.0",
        "case_id": manifest.case_id,
        "source_grading_sha256": manifest.source_grading_sha256,
        "provenance": {
            "method": "human_review",
            "model_identifier": None,
            "model_identifier_unavailable_reason": "Human review; no model used.",
            "prompt_version": "grading-human-review-v1.0.0",
            "skill_version": "exam-grading-v1.0.0",
            "decided_at": datetime.now(UTC).isoformat(),
        },
        "decisions": [
            {
                "target_id": item.target.target_id,
                "rubric_evaluations": [
                    evaluation.model_dump(mode="json")
                    for evaluation in item.proposed_result.rubric_evaluations
                ],
                "error_diagnoses": [
                    diagnosis.model_dump(mode="json")
                    for diagnosis in item.proposed_result.error_diagnoses
                ],
                "decision_reason": "REPLACE WITH HUMAN REVIEW REASON",
            }
            for item in manifest.items
        ],
    }
    editable_json = escape(
        json.dumps(decision_template, ensure_ascii=False, indent=2), quote=False
    )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Grading review</title><style>"
        "body{font-family:system-ui,sans-serif;margin:2rem;max-width:1100px}"
        "section{border:1px solid #bbb;border-radius:8px;padding:1rem;margin:1rem 0}"
        "img{max-width:100%;max-height:420px;border:1px solid #ddd}"
        "table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:.4rem}"
        "textarea{box-sizing:border-box;width:100%;min-height:32rem;font-family:monospace}"
        "</style></head><body><h1>Phase 6 grading review</h1>"
        f"<p>Case: {escape(manifest.case_id)}</p>{body}"
        "<h2>Editable review decision JSON</h2>"
        "<p>Edit all required rubric decisions and reasons, then download the JSON "
        "and pass it to the <code>review</code> command.</p>"
        f"<textarea id='decision-json'>{editable_json}</textarea>"
        "<p><button type='button' onclick='downloadReview()'>Download review JSON</button></p>"
        "<script>function downloadReview(){const text=document.getElementById('decision-json').value;"
        "const blob=new Blob([text],{type:'application/json'});const url=URL.createObjectURL(blob);"
        "const link=document.createElement('a');link.href=url;link.download='grading_review_decisions.json';"
        "link.click();URL.revokeObjectURL(url);}</script></body></html>"
    )


def _review_manifest_signature(manifest: GradingReviewManifest) -> dict[str, object]:
    payload = manifest.model_dump(mode="json")
    payload.pop("created_at", None)
    return payload


def _validate_html(path: Path) -> None:
    content = path.read_text(encoding="utf-8")
    if not content.startswith("<!doctype html>") or "</html>" not in content:
        raise CaseValidationError(f"invalid grading review HTML at {path}")


def _read_model_file(path: Path, model_type, label: str):
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise CaseValidationError(f"cannot read {label} file {path}: {exc}") from exc
    return validate_json(model_type, raw)


def _stable_id(kind: str, *parts: str) -> str:
    digest = digest_value({"kind": kind, "parts": list(parts)})
    return f"{kind}-{digest[:20]}"
