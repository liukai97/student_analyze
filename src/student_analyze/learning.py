"""Deterministic phase 7 context, validation, evidence, and mastery compilation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from html import escape
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from student_analyze.assets import artifact_digest
from student_analyze.atomic import (
    InterruptHook,
    atomic_write_bytes,
    atomic_write_model,
    serialize_model,
)
from student_analyze.config import AppConfig
from student_analyze.database import HistoryEvidence, load_active_history
from student_analyze.errors import CaseValidationError, InvalidTransitionError
from student_analyze.fingerprint import digest_value
from student_analyze.grading_models import AcademicErrorType, Grading, GradingPhase
from student_analyze.learning_models import (
    AttributionScope,
    ClaimStrength,
    ExamDatePrecision,
    HistoricalKnowledgeEvidence,
    KnowledgeCatalog,
    KnowledgeChangeRecord,
    KnowledgeChangeType,
    KnowledgeEvidence,
    KnowledgeMappingDecision,
    KnowledgePoint,
    KnowledgePointStatus,
    LearningAnalysis,
    LearningAnalysisDecision,
    LearningInputManifest,
    LearningReviewItem,
    LearningReviewItemKind,
    LearningReviewManifest,
    LearningTargetInput,
    MappingStatus,
    MasterySnapshot,
    MasteryState,
    TrendStatus,
    ExamMetadataDecision,
)
from student_analyze.models import PipelineStage, PipelineState
from student_analyze.pipeline import verify_case
from student_analyze.validation import validate_json


ModelT = TypeVar("ModelT", bound=BaseModel)
NON_DIAGNOSTIC_ERRORS = {
    AcademicErrorType.UNANSWERED,
    AcademicErrorType.INCORRECT_OBJECTIVE,
}
LOW_MAPPING_CONFIDENCE = 0.8


@dataclass(frozen=True, slots=True)
class LearningContextPreparationResult:
    case_dir: Path
    manifest_path: Path
    manifest: LearningInputManifest
    reused: bool


@dataclass(frozen=True, slots=True)
class LearningCompilationResult:
    analysis: LearningAnalysis
    catalog_sha256: str
    history_evidence_count: int


@dataclass(frozen=True, slots=True)
class LearningReviewPreparationResult:
    manifest_path: Path
    html_path: Path
    manifest: LearningReviewManifest
    reused: bool


def load_exam_metadata(path: Path) -> ExamMetadataDecision:
    return _read_model_file(path, ExamMetadataDecision, "exam metadata")


def load_knowledge_catalog(path: Path) -> KnowledgeCatalog:
    return _read_model_file(path, KnowledgeCatalog, "knowledge catalog")


def load_learning_input_manifest(path: Path) -> LearningInputManifest:
    return _read_model_file(path, LearningInputManifest, "learning input manifest")


def load_learning_decisions(path: Path) -> LearningAnalysisDecision:
    return _read_model_file(path, LearningAnalysisDecision, "learning analysis decisions")


def read_learning_analysis(path: Path) -> LearningAnalysis:
    return _read_model_file(path, LearningAnalysis, "learning analysis")


def prepare_learning_review(
    manifest_path: Path,
    decision_path: Path,
) -> LearningReviewPreparationResult:
    """Render unresolved semantic candidates without changing the formal decision."""

    manifest_path = manifest_path.resolve(strict=True)
    decision_path = decision_path.resolve(strict=True)
    manifest = load_learning_input_manifest(manifest_path)
    decision = load_learning_decisions(decision_path)
    manifest_sha256, _ = artifact_digest(manifest_path)
    decision_sha256, _ = artifact_digest(decision_path)
    if decision.case_id != manifest.case_id:
        raise CaseValidationError("learning decisions belong to another case")
    if decision.learning_input_manifest_sha256 != manifest_sha256:
        raise CaseValidationError("learning decisions reference another input manifest")
    items = _learning_review_items(manifest, decision)
    review = LearningReviewManifest(
        case_id=manifest.case_id,
        learning_input_manifest_sha256=manifest_sha256,
        learning_analysis_decision_sha256=decision_sha256,
        created_at=decision.provenance.decided_at,
        items=items,
        requires_review=bool(items),
    )
    case_dir = next(
        (parent for parent in manifest_path.parents if parent.name == manifest.case_id),
        None,
    )
    output_root = case_dir / "work" if case_dir is not None else manifest_path.parent
    output_dir = output_root / "learning_review" / decision_sha256
    review_path = output_dir / "learning_review_manifest.json"
    html_path = output_dir / "learning_review.html"
    reused = review_path.exists() and html_path.exists()
    if review_path.exists():
        persisted = _read_model_file(
            review_path, LearningReviewManifest, "learning review manifest"
        )
        if serialize_model(persisted) != serialize_model(review):
            raise CaseValidationError("conflicting learning review packet")
    else:
        atomic_write_model(review_path, review, model_type=LearningReviewManifest)
    html = _render_learning_review(review, manifest, decision).encode("utf-8")
    if html_path.exists() and html_path.read_bytes() != html:
        raise CaseValidationError("conflicting learning review HTML")
    if not html_path.exists():
        def validate_html(path: Path) -> None:
            if path.read_bytes() != html:
                raise CaseValidationError("invalid review HTML")

        atomic_write_bytes(
            html_path,
            html,
            validator=validate_html,
        )
    return LearningReviewPreparationResult(review_path, html_path, review, reused)


def prepare_learning_context(
    case_dir: Path,
    config: AppConfig,
    *,
    metadata_path: Path,
    catalog_path: Path | None = None,
    catalog_dir: Path | None = None,
    interrupt_hook: InterruptHook | None = None,
) -> LearningContextPreparationResult:
    """Freeze reviewed scoring, exam metadata, and the current catalog for the LLM."""

    case_dir = case_dir.resolve(strict=True)
    _, state = verify_case(case_dir)
    grading_path, grading = _active_reviewed_grading(case_dir, state)
    metadata_path = metadata_path.resolve(strict=True)
    metadata = load_exam_metadata(metadata_path)
    if metadata.case_id != grading.case_id:
        raise CaseValidationError("exam metadata belongs to another case")
    if grading.phase != GradingPhase.REVIEWED or grading.requires_review:
        raise InvalidTransitionError("phase 7 requires completed grading review")
    if grading.final_score is None:
        raise CaseValidationError("reviewed grading must contain a final score")

    metadata_sha256, _ = artifact_digest(metadata_path)
    if catalog_path is None and catalog_dir is not None:
        catalog_path = _latest_catalog_path(catalog_dir, metadata.subject)
    if catalog_path is not None:
        resolved_catalog = catalog_path.resolve(strict=True)
        catalog = load_knowledge_catalog(resolved_catalog)
        catalog_sha256, _ = artifact_digest(resolved_catalog)
    else:
        catalog = _empty_catalog(metadata)
        catalog_sha256 = _model_sha256(catalog)
    if catalog.subject.casefold() != metadata.subject.casefold():
        raise CaseValidationError("knowledge catalog and exam metadata subjects differ")

    reviewed_sha256, _ = artifact_digest(grading_path)
    result_by_id = {item.target_id: item for item in grading.targets}
    targets = [
        LearningTargetInput(target=item, result=result_by_id[item.target_id])
        for item in grading.input_manifest.targets
    ]
    fingerprint = digest_value(
        {
            "case_id": grading.case_id,
            "reviewed_grading_sha256": reviewed_sha256,
            "exam_metadata_sha256": metadata_sha256,
            "knowledge_catalog_sha256": catalog_sha256,
            "config": config.learning_context_fingerprint_payload(),
        }
    )
    output_path = (
        case_dir
        / "work"
        / "learning_context"
        / fingerprint
        / "learning_input_manifest.json"
    )
    candidate = LearningInputManifest(
        case_id=grading.case_id,
        exam_master_sha256=grading.exam_master_sha256,
        submission_sha256=grading.submission_sha256,
        reviewed_grading_sha256=reviewed_sha256,
        exam_metadata_sha256=metadata_sha256,
        knowledge_catalog_sha256=catalog_sha256,
        manifest_fingerprint=fingerprint,
        created_at=datetime.now(UTC),
        metadata=metadata,
        catalog=catalog,
        targets=targets,
    )
    if output_path.exists():
        persisted = load_learning_input_manifest(output_path)
        _verify_reusable_manifest(persisted, candidate)
        return LearningContextPreparationResult(case_dir, output_path, persisted, True)
    atomic_write_model(
        output_path,
        candidate,
        model_type=LearningInputManifest,
        interrupt_hook=interrupt_hook,
    )
    return LearningContextPreparationResult(case_dir, output_path, candidate, False)


def compile_learning_analysis(
    manifest_path: Path,
    decision_path: Path,
    config: AppConfig,
    *,
    db_path: Path,
) -> LearningCompilationResult:
    """Validate LLM semantics and deterministically compile evidence and history."""

    manifest_path = manifest_path.resolve(strict=True)
    decision_path = decision_path.resolve(strict=True)
    manifest = load_learning_input_manifest(manifest_path)
    decision = load_learning_decisions(decision_path)
    manifest_sha256, _ = artifact_digest(manifest_path)
    decision_sha256, _ = artifact_digest(decision_path)
    if decision.case_id != manifest.case_id:
        raise CaseValidationError("learning decisions belong to another case")
    if decision.learning_input_manifest_sha256 != manifest_sha256:
        raise CaseValidationError("learning decisions reference another input manifest")
    metadata = manifest.metadata
    if (
        (metadata.confidence < LOW_MAPPING_CONFIDENCE or metadata.source.value == "inferred_content")
        and not metadata.human_confirmed
    ):
        raise CaseValidationError("exam metadata requires completed human review")

    catalog = _materialize_catalog(manifest, decision)
    catalog_sha256 = _model_sha256(catalog)
    mappings, unmapped = _validate_mappings(manifest, decision, catalog)
    evidence = _build_current_evidence(manifest, mappings)
    available_history = [
        item
        for item in load_active_history(
            db_path,
            subject=catalog.subject,
            catalog_sha256=catalog_sha256,
        )
        if item.case_id != manifest.case_id
    ]
    history = _history_before_current(manifest, available_history)
    history_fingerprint = digest_value(
        [
            {
                "analysis_id": item.analysis_id,
                "evidence_id": item.evidence_id,
                "outcome": item.outcome,
                "effective_weight": item.effective_weight,
            }
            for item in history
        ]
    )
    analysis_id = "analysis-" + digest_value(
        {
            "case_id": manifest.case_id,
            "manifest_sha256": manifest_sha256,
            "decision_sha256": decision_sha256,
            "catalog_sha256": catalog_sha256,
            "history_fingerprint": history_fingerprint,
            "mapping": config.knowledge_mapping_version,
            "mastery": config.mastery_algorithm_version,
            "report": config.report_policy_version,
        }
    )[:20]
    snapshots = _build_snapshots(
        analysis_id,
        manifest,
        evidence,
        history,
    )
    analysis = LearningAnalysis(
        analysis_id=analysis_id,
        case_id=manifest.case_id,
        created_at=decision.provenance.decided_at,
        learning_input_manifest_sha256=manifest_sha256,
        learning_analysis_decision_sha256=decision_sha256,
        knowledge_catalog_sha256=catalog_sha256,
        mapping_policy_version=config.knowledge_mapping_version,
        mastery_algorithm_version=config.mastery_algorithm_version,
        report_policy_version=config.report_policy_version,
        input_manifest=manifest,
        decision=decision,
        catalog=catalog,
        evidence=evidence,
        historical_evidence=[
            HistoricalKnowledgeEvidence(
                analysis_id=item.analysis_id,
                evidence_id=item.evidence_id,
                case_id=item.case_id,
                question_id=item.question_id,
                target_id=item.target_id,
                rubric_ref=item.rubric_ref,
                printed_label=item.printed_label,
                point_id=item.point_id,
                outcome=item.outcome,
                effective_weight=item.effective_weight,
                occurred_at=item.occurred_at,
                occurred_at_precision=item.occurred_at_precision,
                source_relative_paths=list(item.source_relative_paths),
            )
            for item in history
        ],
        snapshots=snapshots,
        recommendations=decision.recommendations,
        unmapped_rubric_refs=unmapped,
        warnings=[*decision.warnings],
    )
    return LearningCompilationResult(analysis, catalog_sha256, len(history))


def _empty_catalog(metadata: ExamMetadataDecision) -> KnowledgeCatalog:
    return KnowledgeCatalog(
        catalog_id="catalog-" + digest_value(metadata.subject.strip().casefold())[:20],
        subject=metadata.subject,
        version=1,
        created_at=metadata.provenance.decided_at,
        provenance=metadata.provenance,
    )


def _latest_catalog_path(catalog_dir: Path, subject: str) -> Path | None:
    catalog_id = "catalog-" + digest_value(subject.strip().casefold())[:20]
    directory = catalog_dir / catalog_id
    candidates: list[tuple[int, Path]] = []
    for path in directory.glob("*.json"):
        catalog = load_knowledge_catalog(path)
        if catalog.catalog_id != catalog_id or catalog.subject.casefold() != subject.casefold():
            raise CaseValidationError(f"misfiled knowledge catalog: {path}")
        candidates.append((catalog.version, path))
    if not candidates:
        return None
    versions = [version for version, _ in candidates]
    if len(versions) != len(set(versions)):
        raise CaseValidationError("knowledge catalog directory contains duplicate versions")
    return max(candidates, key=lambda item: item[0])[1]


def _materialize_catalog(
    manifest: LearningInputManifest,
    decision: LearningAnalysisDecision,
) -> KnowledgeCatalog:
    base = manifest.catalog
    if _model_sha256(base) != manifest.knowledge_catalog_sha256:
        raise CaseValidationError("embedded knowledge catalog hash does not match manifest")
    if not decision.proposed_points and not decision.knowledge_changes:
        return base

    points = {item.point_id: item for item in base.points}
    target_ids = {item.target.target_id for item in manifest.targets}
    for proposal in decision.proposed_points:
        if proposal.requires_review or not proposal.human_confirmed:
            raise CaseValidationError("proposed points with unresolved review cannot be used")
        if not set(proposal.evidence_target_ids) <= target_ids:
            raise CaseValidationError("knowledge point proposal references unknown targets")
        if proposal.point_id in points:
            raise CaseValidationError(f"duplicate proposed point {proposal.point_id}")
        if proposal.parent_point_id is not None and proposal.parent_point_id not in points:
            raise CaseValidationError("proposed knowledge point references unknown parent")
        points[proposal.point_id] = KnowledgePoint(
            point_id=proposal.point_id,
            name=proposal.name,
            description=proposal.description,
            aliases=proposal.aliases,
            parent_point_id=proposal.parent_point_id,
        )

    changes: list[KnowledgeChangeRecord] = []
    for change in decision.knowledge_changes:
        if change.requires_review or not change.human_confirmed:
            raise CaseValidationError("knowledge changes require completed human review")
        if not set(change.evidence_target_ids) <= target_ids:
            raise CaseValidationError("knowledge change references unknown evidence targets")
        if any(point_id not in points for point_id in change.source_point_ids):
            raise CaseValidationError("knowledge change references unknown source points")
        if any(point_id not in points for point_id in change.target_point_ids):
            raise CaseValidationError("knowledge change references unknown target points")
        if change.change_type in {
            KnowledgeChangeType.RENAME,
            KnowledgeChangeType.ALIAS,
            KnowledgeChangeType.MOVE,
        } and len(change.source_point_ids) != 1:
            raise CaseValidationError("rename, alias, and move require one source point")
        if change.change_type == KnowledgeChangeType.MOVE:
            if change.proposed_parent_point_id not in points:
                raise CaseValidationError("knowledge move references unknown parent")
        for point_id in change.source_point_ids:
            point = points[point_id]
            payload = point.model_dump(mode="python")
            if change.change_type == KnowledgeChangeType.RENAME:
                payload["name"] = change.proposed_name
            elif change.change_type == KnowledgeChangeType.ALIAS:
                payload["aliases"] = [*point.aliases, change.proposed_alias]
            elif change.change_type == KnowledgeChangeType.MOVE:
                payload["parent_point_id"] = change.proposed_parent_point_id
            elif change.change_type in {
                KnowledgeChangeType.RETIRE,
                KnowledgeChangeType.MERGE,
                KnowledgeChangeType.SPLIT,
            }:
                payload["status"] = KnowledgePointStatus.RETIRED
                payload["replacement_point_ids"] = change.target_point_ids
            points[point_id] = KnowledgePoint.model_validate(payload)
        changes.append(
            KnowledgeChangeRecord(
                change_id=change.change_id,
                change_type=change.change_type,
                source_point_ids=change.source_point_ids,
                target_point_ids=change.target_point_ids,
                rationale=change.rationale,
                human_confirmed=change.human_confirmed,
            )
        )
    return KnowledgeCatalog(
        catalog_id=base.catalog_id,
        subject=base.subject,
        version=base.version + 1,
        parent_catalog_sha256=manifest.knowledge_catalog_sha256,
        created_at=decision.provenance.decided_at,
        provenance=decision.provenance,
        points=sorted(points.values(), key=lambda item: item.point_id),
        relations=base.relations,
        changes=[*base.changes, *changes],
    )


def _validate_mappings(
    manifest: LearningInputManifest,
    decision: LearningAnalysisDecision,
    catalog: KnowledgeCatalog,
) -> tuple[list[KnowledgeMappingDecision], list[str]]:
    targets = {item.target.target_id: item for item in manifest.targets}
    expected = {
        (item.target.target_id, rubric.ref)
        for item in manifest.targets
        for rubric in item.target.rubric
    }
    grouped: dict[tuple[str, str], list[KnowledgeMappingDecision]] = defaultdict(list)
    active_points = {
        item.point_id for item in catalog.points if item.status == KnowledgePointStatus.ACTIVE
    }
    mappings_by_id = {item.mapping_id: item for item in decision.mappings}
    mapping_ids = set(mappings_by_id)
    for mapping in decision.mappings:
        key = (mapping.target_id, mapping.rubric_ref)
        if key not in expected:
            raise CaseValidationError("knowledge mapping references an unknown target rubric")
        grouped[key].append(mapping)
        if mapping.requires_review:
            raise CaseValidationError("knowledge mapping still requires human review")
        if (
            mapping.confidence < LOW_MAPPING_CONFIDENCE
            or mapping.status == MappingStatus.UNMAPPED
        ) and not mapping.human_confirmed:
            raise CaseValidationError(
                "low-confidence and unmapped decisions require human confirmation"
            )
        if mapping.status == MappingStatus.MAPPED and mapping.point_id not in active_points:
            raise CaseValidationError("knowledge mapping references a missing or retired point")
        if AttributionScope.DIAGNOSTIC in mapping.attribution_scopes:
            diagnosed = {
                diagnosis.error_type
                for diagnosis in targets[mapping.target_id].result.error_diagnoses
                if mapping.rubric_ref in diagnosis.rubric_refs
            }
            if not set(mapping.source_error_types) <= diagnosed:
                raise CaseValidationError(
                    "diagnostic mapping is not supported by phase 6 error evidence"
                )
            if set(mapping.source_error_types) & NON_DIAGNOSTIC_ERRORS:
                raise CaseValidationError(
                    "blank or incorrect-objective errors cannot diagnose a knowledge point"
                )
    if set(grouped) != expected:
        missing = sorted(expected - set(grouped))
        raise CaseValidationError(f"learning decisions do not cover every rubric: {missing}")
    unmapped: list[str] = []
    for key in sorted(expected):
        items = grouped[key]
        statuses = {item.status for item in items}
        if statuses == {MappingStatus.UNMAPPED}:
            if len(items) != 1:
                raise CaseValidationError("unmapped rubrics require exactly one decision")
            unmapped.append(key[1])
        elif statuses == {MappingStatus.MAPPED}:
            if abs(sum(item.weight for item in items) - 1.0) > 1e-9:
                raise CaseValidationError("mapped rubric weights must sum to one")
        else:
            raise CaseValidationError("a rubric cannot be both mapped and unmapped")
    for recommendation in decision.recommendations:
        if not set(recommendation.point_ids) <= active_points:
            raise CaseValidationError("recommendation references an unknown knowledge point")
        if not set(recommendation.mapping_ids) <= mapping_ids:
            raise CaseValidationError("recommendation references an unknown mapping")
        if any(
            mappings_by_id[mapping_id].status != MappingStatus.MAPPED
            or mappings_by_id[mapping_id].point_id not in recommendation.point_ids
            for mapping_id in recommendation.mapping_ids
        ):
            raise CaseValidationError(
                "recommendation mappings must provide evidence for its knowledge points"
            )
    return decision.mappings, unmapped


def _build_current_evidence(
    manifest: LearningInputManifest,
    mappings: list[KnowledgeMappingDecision],
) -> list[KnowledgeEvidence]:
    targets = {item.target.target_id: item for item in manifest.targets}
    built: list[KnowledgeEvidence] = []
    for mapping in mappings:
        if mapping.status == MappingStatus.UNMAPPED:
            continue
        item = targets[mapping.target_id]
        rubric = next(value for value in item.target.rubric if value.ref == mapping.rubric_ref)
        evaluation = next(
            value
            for value in item.result.rubric_evaluations
            if value.rubric_ref == mapping.rubric_ref
        )
        if rubric.points is None or rubric.points <= 0 or evaluation.awarded_points is None:
            raise CaseValidationError("mapped rubric evidence must be positive and fully scored")
        if evaluation.awarded_points > rubric.points:
            raise CaseValidationError("rubric award exceeds its maximum")
        error_types = (
            mapping.source_error_types
            if AttributionScope.DIAGNOSTIC in mapping.attribution_scopes
            else []
        )
        built.append(
            KnowledgeEvidence(
                evidence_id="evidence-" + digest_value(
                    {
                        "case_id": manifest.case_id,
                        "mapping_id": mapping.mapping_id,
                        "reviewed_grading_sha256": manifest.reviewed_grading_sha256,
                    }
                )[:20],
                mapping_id=mapping.mapping_id,
                case_id=manifest.case_id,
                question_id=item.target.question_id,
                target_id=item.target.target_id,
                rubric_ref=mapping.rubric_ref,
                printed_label=item.target.printed_label,
                point_id=mapping.point_id,
                attribution_scopes=mapping.attribution_scopes,
                outcome=evaluation.awarded_points / rubric.points,
                awarded_points=evaluation.awarded_points,
                max_points=rubric.points,
                allocated_points=rubric.points * mapping.weight,
                effective_weight=rubric.points * mapping.weight * mapping.confidence,
                mapping_confidence=mapping.confidence,
                error_types=error_types,
                submission_item_ids=item.result.submission_item_ids,
            )
        )
    return built


def _build_snapshots(
    analysis_id: str,
    manifest: LearningInputManifest,
    current: list[KnowledgeEvidence],
    history: list[HistoryEvidence],
) -> list[MasterySnapshot]:
    by_point: dict[str, list[tuple[str, str, str, str, float, float]]] = defaultdict(list)
    for item in history:
        by_point[item.point_id].append(
            (
                item.evidence_id,
                item.case_id,
                item.question_id,
                item.target_id,
                item.outcome,
                item.effective_weight,
            )
        )
    for item in current:
        by_point[item.point_id].append(
            (
                item.evidence_id,
                manifest.case_id,
                item.question_id,
                item.target_id,
                item.outcome,
                item.effective_weight,
            )
        )
    snapshots: list[MasterySnapshot] = []
    current_point_ids = {item.point_id for item in current}
    for point_id, values in sorted(by_point.items()):
        if point_id not in current_point_ids:
            continue
        weight = sum(item[5] for item in values)
        performance = sum(item[4] * item[5] for item in values) / weight
        target_count = len({(item[1], item[3]) for item in values})
        question_count = len({(item[1], item[2]) for item in values})
        exam_count = len({item[1] for item in values})
        if target_count < 2:
            state = MasteryState.INSUFFICIENT_EVIDENCE
            strength = ClaimStrength.ITEM_ONLY
        elif exam_count == 1:
            state = MasteryState.SINGLE_EXAM_SIGNAL
            strength = ClaimStrength.SINGLE_EXAM
        else:
            strength = ClaimStrength.LONGITUDINAL
            if performance < 0.6:
                state = MasteryState.NEEDS_PRACTICE
            elif performance >= 0.8:
                state = MasteryState.CONSISTENT
            else:
                state = MasteryState.MIXED
        snapshots.append(
            MasterySnapshot(
                snapshot_id="snapshot-" + digest_value(
                    {"analysis_id": analysis_id, "point_id": point_id}
                )[:20],
                point_id=point_id,
                performance_index=performance,
                state=state,
                trend=_trend_for_point(point_id, manifest, current, history),
                allowed_claim_strength=strength,
                evidence_count=len(values),
                independent_target_count=target_count,
                independent_question_count=question_count,
                independent_exam_count=exam_count,
                effective_weight=weight,
                evidence_ids=[item[0] for item in values],
            )
        )
    return snapshots


def _trend_for_point(
    point_id: str,
    manifest: LearningInputManifest,
    current: list[KnowledgeEvidence],
    history: list[HistoryEvidence],
) -> TrendStatus:
    metadata = manifest.metadata
    if metadata.occurred_at_precision not in {
        ExamDatePrecision.DAY,
        ExamDatePrecision.DATETIME,
    }:
        return TrendStatus.NOT_COMPARABLE
    current_values = [item for item in current if item.point_id == point_id]
    if not current_values:
        return TrendStatus.NOT_COMPARABLE
    prior_by_exam: dict[tuple[str, str], list[HistoryEvidence]] = defaultdict(list)
    for item in history:
        if (
            item.point_id == point_id
            and item.occurred_at is not None
            and item.occurred_at_precision in {"day", "datetime"}
        ):
            prior_by_exam[(item.case_id, item.occurred_at)].append(item)
    if not prior_by_exam:
        return TrendStatus.NOT_COMPARABLE
    latest_key = max(
        prior_by_exam,
        key=lambda item: _occurred_order(
            item[1], prior_by_exam[item][0].occurred_at_precision
        ),
    )
    if (
        metadata.occurred_at is None
        or _occurred_order(metadata.occurred_at, metadata.occurred_at_precision.value)
        <= _occurred_order(latest_key[1], prior_by_exam[latest_key][0].occurred_at_precision)
    ):
        return TrendStatus.NOT_COMPARABLE
    prior = prior_by_exam[latest_key]
    prior_weight = sum(item.effective_weight for item in prior)
    prior_score = sum(item.outcome * item.effective_weight for item in prior) / prior_weight
    current_weight = sum(item.effective_weight for item in current_values)
    current_score = (
        sum(item.outcome * item.effective_weight for item in current_values) / current_weight
    )
    delta = current_score - prior_score
    if delta >= 0.1:
        return TrendStatus.IMPROVING
    if delta <= -0.1:
        return TrendStatus.DECLINING
    return TrendStatus.STABLE


def _occurred_order(value: str, precision: str) -> float:
    if precision == ExamDatePrecision.DAY.value:
        parsed = datetime.fromisoformat(value).replace(tzinfo=UTC)
    else:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.timestamp()


def _history_before_current(
    manifest: LearningInputManifest,
    history: list[HistoryEvidence],
) -> list[HistoryEvidence]:
    metadata = manifest.metadata
    if metadata.occurred_at is None:
        return []
    current_start, _ = _occurred_interval(
        metadata.occurred_at, metadata.occurred_at_precision.value
    )
    included: list[HistoryEvidence] = []
    for item in history:
        if item.occurred_at is None or item.occurred_at_precision == "unknown":
            continue
        prior_start, prior_end = _occurred_interval(
            item.occurred_at, item.occurred_at_precision
        )
        is_before = (
            prior_start < current_start
            if item.occurred_at_precision == ExamDatePrecision.DATETIME.value
            else prior_end <= current_start
        )
        if is_before:
            included.append(item)
    return included


def _occurred_interval(value: str, precision: str) -> tuple[float, float]:
    if precision == ExamDatePrecision.YEAR.value:
        start = datetime(int(value), 1, 1, tzinfo=UTC)
        end = datetime(int(value) + 1, 1, 1, tzinfo=UTC)
    elif precision == ExamDatePrecision.MONTH.value:
        year, month = (int(item) for item in value.split("-"))
        start = datetime(year, month, 1, tzinfo=UTC)
        end = (
            datetime(year + 1, 1, 1, tzinfo=UTC)
            if month == 12
            else datetime(year, month + 1, 1, tzinfo=UTC)
        )
    elif precision == ExamDatePrecision.DAY.value:
        start = datetime.fromisoformat(value).replace(tzinfo=UTC)
        end = start + timedelta(days=1)
    else:
        start = datetime.fromisoformat(value.replace("Z", "+00:00"))
        end = start
    return start.timestamp(), end.timestamp()


def _active_reviewed_grading(
    case_dir: Path, state: PipelineState
) -> tuple[Path, Grading]:
    completion = next(
        (item for item in state.completed_stages if item.stage == PipelineStage.REVIEWED),
        None,
    )
    if completion is None:
        raise InvalidTransitionError("phase 7 requires an active reviewed grading artifact")
    references = [
        item for item in completion.artifacts if item.schema_id == "grading.schema.json"
    ]
    if len(references) != 1:
        raise CaseValidationError("reviewed stage must have one active grading artifact")
    path = case_dir / references[0].relative_path
    return path, _read_model_file(path, Grading, "reviewed grading")


def _learning_review_items(
    manifest: LearningInputManifest,
    decision: LearningAnalysisDecision,
) -> list[LearningReviewItem]:
    pending: list[tuple[LearningReviewItemKind, str, list[str]]] = []
    metadata_reasons: list[str] = []
    if manifest.metadata.confidence < LOW_MAPPING_CONFIDENCE:
        metadata_reasons.append("考试元数据置信度低于正式报告门槛")
    if manifest.metadata.source.value == "inferred_content":
        metadata_reasons.append("考试元数据来自内容推断")
    if metadata_reasons and not manifest.metadata.human_confirmed:
        pending.append(
            (LearningReviewItemKind.EXAM_METADATA, manifest.case_id, metadata_reasons)
        )
    for proposal in decision.proposed_points:
        reasons = []
        if proposal.requires_review:
            reasons.append("模型将新增知识点标记为需要复核")
        if not proposal.human_confirmed:
            reasons.append("新增知识点尚未人工确认")
        if reasons:
            pending.append(
                (
                    LearningReviewItemKind.KNOWLEDGE_POINT_PROPOSAL,
                    proposal.point_id,
                    reasons,
                )
            )
    for change in decision.knowledge_changes:
        reasons = []
        if change.requires_review:
            reasons.append("知识目录变更被标记为需要复核")
        if not change.human_confirmed:
            reasons.append("对现有知识点的变更尚未人工确认")
        if reasons:
            pending.append(
                (LearningReviewItemKind.KNOWLEDGE_CHANGE, change.change_id, reasons)
            )
    for mapping in decision.mappings:
        reasons = []
        if mapping.requires_review:
            reasons.append("知识映射被标记为需要复核")
        if mapping.confidence < LOW_MAPPING_CONFIDENCE and not mapping.human_confirmed:
            reasons.append("知识映射置信度低于正式报告门槛")
        if mapping.status == MappingStatus.UNMAPPED and not mapping.human_confirmed:
            reasons.append("显式未映射项需要人工确认")
        if reasons:
            pending.append(
                (LearningReviewItemKind.KNOWLEDGE_MAPPING, mapping.mapping_id, reasons)
            )
    return [
        LearningReviewItem(
            review_item_id="review-" + digest_value(
                {"kind": kind.value, "entity_id": entity_id}
            )[:20],
            kind=kind,
            entity_id=entity_id,
            reasons=reasons,
        )
        for kind, entity_id, reasons in pending
    ]


def _render_learning_review(
    review: LearningReviewManifest,
    manifest: LearningInputManifest,
    decision: LearningAnalysisDecision,
) -> str:
    details: dict[str, str] = {manifest.case_id: manifest.metadata.subject}
    details.update(
        (item.point_id, f"{item.name}：{item.description}")
        for item in decision.proposed_points
    )
    details.update(
        (item.change_id, item.rationale) for item in decision.knowledge_changes
    )
    details.update((item.mapping_id, item.rationale) for item in decision.mappings)
    rows = "".join(
        "<tr><td>"
        + escape(item.kind.value)
        + "</td><td><code>"
        + escape(item.entity_id)
        + "</code></td><td>"
        + escape(details.get(item.entity_id, ""))
        + "</td><td>"
        + escape("；".join(item.reasons))
        + "</td></tr>"
        for item in review.items
    )
    if not rows:
        rows = '<tr><td colspan="4">没有待复核项。</td></tr>'
    return (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<title>阶段 7 学习分析复核</title><style>body{max-width:1100px;margin:2rem auto;'
        'font:15px/1.6 system-ui,sans-serif}table{border-collapse:collapse;width:100%}'
        'th,td{border:1px solid #ccc;padding:.5rem;text-align:left;vertical-align:top}'
        '</style></head><body><h1>阶段 7 学习分析复核</h1>'
        f'<p>Case：<code>{escape(review.case_id)}</code></p>'
        '<p>人工确认后，请更新原决定 JSON 中对应实体的 '
        '<code>human_confirmed</code>/<code>requires_review</code>，并保留审阅说明。</p>'
        '<table><thead><tr><th>类型</th><th>实体</th><th>内容</th><th>原因</th>'
        f'</tr></thead><tbody>{rows}</tbody></table></body></html>'
    )


def _verify_reusable_manifest(
    persisted: LearningInputManifest, candidate: LearningInputManifest
) -> None:
    expected = candidate.model_dump(mode="json")
    actual = persisted.model_dump(mode="json")
    expected.pop("created_at", None)
    actual.pop("created_at", None)
    if actual != expected:
        raise CaseValidationError("conflicting learning context at the same fingerprint")


def _model_sha256(model: BaseModel) -> str:
    return sha256(serialize_model(model)).hexdigest()


def _read_model_file(path: Path, model_type: type[ModelT], label: str) -> ModelT:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise CaseValidationError(f"cannot read {label} {path}: {exc}") from exc
    return validate_json(model_type, raw)
