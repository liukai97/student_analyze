"""Phase 7 contracts for knowledge mapping, history, and reports."""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
import re
from typing import Literal

from pydantic import Field, model_validator

from student_analyze.document_models import DocumentDecisionProvenance
from student_analyze.grading_models import (
    AcademicErrorType,
    GradingTargetInput,
    GradingTargetResult,
)
from student_analyze.models import Sha256, StableId, StrictModel
from student_analyze.page_models import Confidence


LEARNING_SCHEMA_VERSION = "1.0.0"


class ExamDatePrecision(str, Enum):
    UNKNOWN = "unknown"
    YEAR = "year"
    MONTH = "month"
    DAY = "day"
    DATETIME = "datetime"


class MetadataSource(str, Enum):
    USER_PROVIDED = "user_provided"
    PRINTED_EVIDENCE = "printed_evidence"
    INFERRED_CONTENT = "inferred_content"
    HUMAN_REVIEW = "human_review"


class KnowledgePointStatus(str, Enum):
    ACTIVE = "active"
    RETIRED = "retired"


class KnowledgeRelationType(str, Enum):
    PARENT_OF = "parent_of"
    PREREQUISITE_OF = "prerequisite_of"
    RELATED_TO = "related_to"


class KnowledgeChangeType(str, Enum):
    RENAME = "rename"
    MOVE = "move"
    ALIAS = "alias"
    MERGE = "merge"
    SPLIT = "split"
    RETIRE = "retire"


class MappingStatus(str, Enum):
    MAPPED = "mapped"
    UNMAPPED = "unmapped"


class MappingRole(str, Enum):
    PRIMARY = "primary"
    SUPPORTING = "supporting"


class AttributionScope(str, Enum):
    ASSESSED = "assessed"
    DIAGNOSTIC = "diagnostic"


class PracticeType(str, Enum):
    CONCEPT_REVIEW = "concept_review"
    TARGETED_PRACTICE = "targeted_practice"
    ERROR_CORRECTION = "error_correction"
    RETRIEVAL_PRACTICE = "retrieval_practice"
    MIXED_REVIEW = "mixed_review"


class PracticeDifficulty(str, Enum):
    FOUNDATIONAL = "foundational"
    STANDARD = "standard"
    CHALLENGE = "challenge"


class MasteryState(str, Enum):
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    SINGLE_EXAM_SIGNAL = "single_exam_signal"
    NEEDS_PRACTICE = "needs_practice"
    MIXED = "mixed"
    CONSISTENT = "consistent"


class TrendStatus(str, Enum):
    NOT_COMPARABLE = "not_comparable"
    IMPROVING = "improving"
    DECLINING = "declining"
    STABLE = "stable"


class ClaimStrength(str, Enum):
    ITEM_ONLY = "item_only"
    SINGLE_EXAM = "single_exam"
    LONGITUDINAL = "longitudinal"


class ClaimKind(str, Enum):
    INSUFFICIENT = "insufficient"
    OBSERVED_STRENGTH = "observed_strength"
    NEEDS_PRACTICE = "needs_practice"
    MIXED = "mixed"
    TREND = "trend"


class LearningReviewItemKind(str, Enum):
    EXAM_METADATA = "exam_metadata"
    KNOWLEDGE_POINT_PROPOSAL = "knowledge_point_proposal"
    KNOWLEDGE_CHANGE = "knowledge_change"
    KNOWLEDGE_MAPPING = "knowledge_mapping"


class StudentProfile(StrictModel):
    schema_version: Literal[LEARNING_SCHEMA_VERSION] = LEARNING_SCHEMA_VERSION
    student_profile_id: StableId = "student-default"
    display_name: str | None = Field(default=None, min_length=1)
    created_at: datetime


class ExamMetadataDecision(StrictModel):
    schema_version: Literal[LEARNING_SCHEMA_VERSION] = LEARNING_SCHEMA_VERSION
    case_id: StableId
    subject: str = Field(min_length=1)
    title: str | None = Field(default=None, min_length=1)
    grade_level: str | None = Field(default=None, min_length=1)
    term: str | None = Field(default=None, min_length=1)
    occurred_at: str | None = None
    occurred_at_precision: ExamDatePrecision = ExamDatePrecision.UNKNOWN
    source: MetadataSource
    confidence: Confidence
    evidence: list[str] = Field(min_length=1)
    human_confirmed: bool = False
    human_review_note: str | None = Field(default=None, min_length=1)
    provenance: DocumentDecisionProvenance

    @model_validator(mode="after")
    def validate_occurred_at(self) -> ExamMetadataDecision:
        if self.human_confirmed != (self.human_review_note is not None):
            raise ValueError("confirmed exam metadata requires exactly one review note")
        value = self.occurred_at
        precision = self.occurred_at_precision
        if precision == ExamDatePrecision.UNKNOWN:
            if value is not None:
                raise ValueError("unknown exam dates must omit occurred_at")
            return self
        if value is None:
            raise ValueError("known exam date precision requires occurred_at")
        try:
            if precision == ExamDatePrecision.YEAR:
                if re.fullmatch(r"\d{4}", value) is None:
                    raise ValueError
                date(int(value), 1, 1)
            elif precision == ExamDatePrecision.MONTH:
                if re.fullmatch(r"\d{4}-\d{2}", value) is None:
                    raise ValueError
                date.fromisoformat(f"{value}-01")
            elif precision == ExamDatePrecision.DAY:
                if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None:
                    raise ValueError
                date.fromisoformat(value)
            else:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    raise ValueError
        except ValueError as exc:
            raise ValueError("occurred_at does not match its declared precision") from exc
        return self


class KnowledgePoint(StrictModel):
    point_id: StableId
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)
    parent_point_id: StableId | None = None
    status: KnowledgePointStatus = KnowledgePointStatus.ACTIVE
    replacement_point_ids: list[StableId] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_point(self) -> KnowledgePoint:
        normalized = [item.strip().casefold() for item in self.aliases]
        if any(not item for item in normalized) or len(normalized) != len(set(normalized)):
            raise ValueError("knowledge point aliases must be non-empty and unique")
        if self.parent_point_id == self.point_id:
            raise ValueError("knowledge point cannot be its own parent")
        if self.point_id in self.replacement_point_ids:
            raise ValueError("knowledge point cannot replace itself")
        if self.status == KnowledgePointStatus.ACTIVE and self.replacement_point_ids:
            raise ValueError("only retired knowledge points may have replacements")
        return self


class KnowledgeRelation(StrictModel):
    relation_id: StableId
    source_point_id: StableId
    target_point_id: StableId
    relation_type: KnowledgeRelationType

    @model_validator(mode="after")
    def validate_relation(self) -> KnowledgeRelation:
        if self.source_point_id == self.target_point_id:
            raise ValueError("knowledge relations cannot be self-referential")
        return self


class KnowledgeChangeRecord(StrictModel):
    change_id: StableId
    change_type: KnowledgeChangeType
    source_point_ids: list[StableId] = Field(default_factory=list)
    target_point_ids: list[StableId] = Field(default_factory=list)
    rationale: str = Field(min_length=1)
    human_confirmed: bool


class KnowledgeCatalog(StrictModel):
    schema_version: Literal[LEARNING_SCHEMA_VERSION] = LEARNING_SCHEMA_VERSION
    catalog_id: StableId
    subject: str = Field(min_length=1)
    version: int = Field(ge=1)
    parent_catalog_sha256: Sha256 | None = None
    created_at: datetime
    provenance: DocumentDecisionProvenance
    points: list[KnowledgePoint] = Field(default_factory=list)
    relations: list[KnowledgeRelation] = Field(default_factory=list)
    changes: list[KnowledgeChangeRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_catalog(self) -> KnowledgeCatalog:
        point_ids = [item.point_id for item in self.points]
        if len(point_ids) != len(set(point_ids)):
            raise ValueError("knowledge catalog point IDs must be unique")
        points = {item.point_id: item for item in self.points}
        names: set[str] = set()
        for point in self.points:
            terms = [point.name, *point.aliases]
            for term in terms:
                normalized = term.strip().casefold()
                if normalized in names:
                    raise ValueError("knowledge point names and aliases must be unique")
                names.add(normalized)
            if point.parent_point_id is not None and point.parent_point_id not in points:
                raise ValueError("knowledge point references an unknown parent")
            if any(item not in points for item in point.replacement_point_ids):
                raise ValueError("knowledge point references an unknown replacement")
        relation_ids = [item.relation_id for item in self.relations]
        if len(relation_ids) != len(set(relation_ids)):
            raise ValueError("knowledge relation IDs must be unique")
        if any(
            item.source_point_id not in points or item.target_point_id not in points
            for item in self.relations
        ):
            raise ValueError("knowledge relation references an unknown point")
        if any(
            point_id not in points
            for change in self.changes
            for point_id in [*change.source_point_ids, *change.target_point_ids]
        ):
            raise ValueError("knowledge change history references an unknown point")
        _reject_parent_cycles(points)
        return self


class KnowledgePointProposal(StrictModel):
    point_id: StableId
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)
    parent_point_id: StableId | None = None
    confidence: Confidence
    evidence_target_ids: list[StableId] = Field(min_length=1)
    human_confirmed: bool = False
    human_review_note: str | None = Field(default=None, min_length=1)
    requires_review: bool = False

    @model_validator(mode="after")
    def validate_review(self) -> KnowledgePointProposal:
        if self.human_confirmed != (self.human_review_note is not None):
            raise ValueError("confirmed point proposals require exactly one review note")
        if self.human_confirmed and self.requires_review:
            raise ValueError("confirmed point proposals cannot still require review")
        return self


class KnowledgeChangeDecision(StrictModel):
    change_id: StableId
    change_type: KnowledgeChangeType
    source_point_ids: list[StableId] = Field(default_factory=list)
    target_point_ids: list[StableId] = Field(default_factory=list)
    proposed_name: str | None = Field(default=None, min_length=1)
    proposed_alias: str | None = Field(default=None, min_length=1)
    proposed_parent_point_id: StableId | None = None
    confidence: Confidence
    rationale: str = Field(min_length=1)
    evidence_target_ids: list[StableId] = Field(min_length=1)
    human_confirmed: bool = False
    human_review_note: str | None = Field(default=None, min_length=1)
    requires_review: bool = True

    @model_validator(mode="after")
    def validate_change(self) -> KnowledgeChangeDecision:
        if self.human_confirmed != (self.human_review_note is not None):
            raise ValueError("confirmed knowledge changes require exactly one review note")
        if self.human_confirmed and self.requires_review:
            raise ValueError("confirmed knowledge changes cannot still require review")
        if not self.source_point_ids:
            raise ValueError("knowledge changes require source points")
        if self.change_type == KnowledgeChangeType.RENAME and self.proposed_name is None:
            raise ValueError("rename requires proposed_name")
        if self.change_type == KnowledgeChangeType.ALIAS and self.proposed_alias is None:
            raise ValueError("alias requires proposed_alias")
        if self.change_type == KnowledgeChangeType.MOVE and self.proposed_parent_point_id is None:
            raise ValueError("move requires proposed_parent_point_id")
        if self.change_type in {KnowledgeChangeType.MERGE, KnowledgeChangeType.SPLIT}:
            if not self.target_point_ids:
                raise ValueError("merge and split require target points")
        return self


class LearningTargetInput(StrictModel):
    target: GradingTargetInput
    result: GradingTargetResult

    @model_validator(mode="after")
    def validate_target(self) -> LearningTargetInput:
        if self.target.target_id != self.result.target_id:
            raise ValueError("learning target input and result use different target IDs")
        if self.target.question_id != self.result.question_id:
            raise ValueError("learning target input and result use different question IDs")
        if self.result.final_score is None:
            raise ValueError("phase 7 requires final target scores")
        input_rubrics = {item.ref for item in self.target.rubric}
        result_rubrics = {item.rubric_ref for item in self.result.rubric_evaluations}
        if input_rubrics != result_rubrics:
            raise ValueError("learning target result must evaluate every input rubric")
        return self


class LearningInputManifest(StrictModel):
    schema_version: Literal[LEARNING_SCHEMA_VERSION] = LEARNING_SCHEMA_VERSION
    case_id: StableId
    exam_master_sha256: Sha256
    submission_sha256: Sha256
    reviewed_grading_sha256: Sha256
    exam_metadata_sha256: Sha256
    knowledge_catalog_sha256: Sha256
    manifest_fingerprint: Sha256
    created_at: datetime
    metadata: ExamMetadataDecision
    catalog: KnowledgeCatalog
    targets: list[LearningTargetInput] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_manifest(self) -> LearningInputManifest:
        if self.metadata.case_id != self.case_id:
            raise ValueError("exam metadata belongs to another case")
        if self.metadata.subject.casefold() != self.catalog.subject.casefold():
            raise ValueError("exam metadata and knowledge catalog use different subjects")
        target_ids = [item.target.target_id for item in self.targets]
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("learning input target IDs must be unique")
        return self


class KnowledgeMappingDecision(StrictModel):
    mapping_id: StableId
    target_id: StableId
    rubric_ref: StableId
    status: MappingStatus
    point_id: StableId | None = None
    role: MappingRole = MappingRole.PRIMARY
    attribution_scopes: list[AttributionScope] = Field(default_factory=list)
    weight: float = Field(default=0, ge=0, le=1)
    confidence: Confidence
    rationale: str = Field(min_length=1)
    source_error_types: list[AcademicErrorType] = Field(default_factory=list)
    unmapped_reason: str | None = Field(default=None, min_length=1)
    human_confirmed: bool = False
    human_review_note: str | None = Field(default=None, min_length=1)
    requires_review: bool = False

    @model_validator(mode="after")
    def validate_mapping(self) -> KnowledgeMappingDecision:
        if self.human_confirmed != (self.human_review_note is not None):
            raise ValueError("confirmed mappings require exactly one review note")
        if self.human_confirmed and self.requires_review:
            raise ValueError("confirmed mappings cannot still require review")
        if len(self.attribution_scopes) != len(set(self.attribution_scopes)):
            raise ValueError("mapping attribution scopes must be unique")
        if len(self.source_error_types) != len(set(self.source_error_types)):
            raise ValueError("mapping source error types must be unique")
        if self.status == MappingStatus.MAPPED:
            if (
                self.point_id is None
                or self.weight <= 0
                or AttributionScope.ASSESSED not in self.attribution_scopes
                or self.unmapped_reason is not None
            ):
                raise ValueError("mapped rubric decisions require a point, weight, and assessed scope")
        else:
            if (
                self.point_id is not None
                or self.weight != 0
                or self.attribution_scopes
                or self.source_error_types
                or self.unmapped_reason is None
            ):
                raise ValueError("unmapped rubric decisions must contain only an explicit reason")
        if AttributionScope.DIAGNOSTIC in self.attribution_scopes and not self.source_error_types:
            raise ValueError("diagnostic mappings require source error types")
        return self


class LearningRecommendation(StrictModel):
    recommendation_id: StableId
    point_ids: list[StableId] = Field(min_length=1)
    practice_type: PracticeType
    difficulty: PracticeDifficulty
    priority: int = Field(ge=1, le=3)
    action: str = Field(min_length=1)
    mapping_ids: list[StableId] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_references(self) -> LearningRecommendation:
        if len(self.point_ids) != len(set(self.point_ids)):
            raise ValueError("recommendation point IDs must be unique")
        if len(self.mapping_ids) != len(set(self.mapping_ids)):
            raise ValueError("recommendation mapping IDs must be unique")
        return self


class LearningAnalysisDecision(StrictModel):
    schema_version: Literal[LEARNING_SCHEMA_VERSION] = LEARNING_SCHEMA_VERSION
    case_id: StableId
    learning_input_manifest_sha256: Sha256
    provenance: DocumentDecisionProvenance
    proposed_points: list[KnowledgePointProposal] = Field(default_factory=list)
    knowledge_changes: list[KnowledgeChangeDecision] = Field(default_factory=list)
    mappings: list[KnowledgeMappingDecision] = Field(min_length=1)
    recommendations: list[LearningRecommendation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_decision(self) -> LearningAnalysisDecision:
        for items, attribute, label in (
            (self.proposed_points, "point_id", "proposed point"),
            (self.knowledge_changes, "change_id", "knowledge change"),
            (self.mappings, "mapping_id", "mapping"),
            (self.recommendations, "recommendation_id", "recommendation"),
        ):
            values = [getattr(item, attribute) for item in items]
            if len(values) != len(set(values)):
                raise ValueError(f"{label} IDs must be unique")
        return self


class LearningReviewItem(StrictModel):
    review_item_id: StableId
    kind: LearningReviewItemKind
    entity_id: StableId
    reasons: list[str] = Field(min_length=1)


class LearningReviewManifest(StrictModel):
    schema_version: Literal[LEARNING_SCHEMA_VERSION] = LEARNING_SCHEMA_VERSION
    case_id: StableId
    learning_input_manifest_sha256: Sha256
    learning_analysis_decision_sha256: Sha256
    created_at: datetime
    items: list[LearningReviewItem] = Field(default_factory=list)
    requires_review: bool

    @model_validator(mode="after")
    def validate_review_items(self) -> LearningReviewManifest:
        item_ids = [item.review_item_id for item in self.items]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("learning review item IDs must be unique")
        if self.requires_review != bool(self.items):
            raise ValueError("requires_review must reflect learning review items")
        return self


class KnowledgeEvidence(StrictModel):
    evidence_id: StableId
    mapping_id: StableId
    case_id: StableId
    question_id: StableId
    target_id: StableId
    rubric_ref: StableId
    printed_label: str = Field(min_length=1)
    point_id: StableId
    attribution_scopes: list[AttributionScope] = Field(min_length=1)
    outcome: float = Field(ge=0, le=1)
    awarded_points: float = Field(ge=0)
    max_points: float = Field(gt=0)
    allocated_points: float = Field(gt=0)
    effective_weight: float = Field(gt=0)
    mapping_confidence: Confidence
    error_types: list[AcademicErrorType] = Field(default_factory=list)
    submission_item_ids: list[StableId] = Field(min_length=1)


class HistoricalKnowledgeEvidence(StrictModel):
    analysis_id: StableId
    evidence_id: StableId
    case_id: StableId
    question_id: StableId
    target_id: StableId
    rubric_ref: StableId
    printed_label: str = Field(min_length=1)
    point_id: StableId
    outcome: float = Field(ge=0, le=1)
    effective_weight: float = Field(gt=0)
    occurred_at: str | None = None
    occurred_at_precision: ExamDatePrecision
    source_relative_paths: list[str] = Field(min_length=1)


class MasterySnapshot(StrictModel):
    snapshot_id: StableId
    point_id: StableId
    performance_index: float = Field(ge=0, le=1)
    state: MasteryState
    trend: TrendStatus
    allowed_claim_strength: ClaimStrength
    evidence_count: int = Field(ge=1)
    independent_target_count: int = Field(ge=1)
    independent_question_count: int = Field(ge=1)
    independent_exam_count: int = Field(ge=1)
    effective_weight: float = Field(gt=0)
    evidence_ids: list[StableId] = Field(min_length=1)


class LearningAnalysis(StrictModel):
    schema_version: Literal[LEARNING_SCHEMA_VERSION] = LEARNING_SCHEMA_VERSION
    analysis_id: StableId
    case_id: StableId
    created_at: datetime
    learning_input_manifest_sha256: Sha256
    learning_analysis_decision_sha256: Sha256
    knowledge_catalog_sha256: Sha256
    mapping_policy_version: str = Field(min_length=1)
    mastery_algorithm_version: str = Field(min_length=1)
    report_policy_version: str = Field(min_length=1)
    input_manifest: LearningInputManifest
    decision: LearningAnalysisDecision
    catalog: KnowledgeCatalog
    evidence: list[KnowledgeEvidence] = Field(default_factory=list)
    historical_evidence: list[HistoricalKnowledgeEvidence] = Field(default_factory=list)
    snapshots: list[MasterySnapshot] = Field(default_factory=list)
    recommendations: list[LearningRecommendation] = Field(default_factory=list)
    unmapped_rubric_refs: list[StableId] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_analysis(self) -> LearningAnalysis:
        if self.case_id != self.input_manifest.case_id or self.case_id != self.decision.case_id:
            raise ValueError("learning analysis inputs belong to another case")
        evidence_ids = [item.evidence_id for item in self.evidence]
        historical_ids = [item.evidence_id for item in self.historical_evidence]
        if len(evidence_ids) != len(set(evidence_ids)) or len(historical_ids) != len(
            set(historical_ids)
        ):
            raise ValueError("knowledge evidence IDs must be unique")
        if set(evidence_ids) & set(historical_ids):
            raise ValueError("current and historical evidence IDs must be disjoint")
        snapshot_points = [item.point_id for item in self.snapshots]
        if len(snapshot_points) != len(set(snapshot_points)):
            raise ValueError("learning analysis snapshots must be unique per point")
        available_evidence = set(evidence_ids) | set(historical_ids)
        if any(
            not set(snapshot.evidence_ids) <= available_evidence
            for snapshot in self.snapshots
        ):
            raise ValueError("mastery snapshot references unknown evidence")
        return self


class ReportClaim(StrictModel):
    claim_id: StableId
    point_id: StableId
    kind: ClaimKind
    strength: ClaimStrength
    text: str = Field(min_length=1)
    evidence_ids: list[StableId] = Field(min_length=1)


class ReportAsset(StrictModel):
    name: str = Field(min_length=1)
    relative_path: str = Field(min_length=1)
    sha256: Sha256
    size_bytes: int = Field(ge=0)
    media_type: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_path(self) -> ReportAsset:
        normalized = self.relative_path.replace("\\", "/")
        parts = normalized.split("/")
        if normalized.startswith("/") or any(item in {"", ".", ".."} for item in parts):
            raise ValueError("report assets must stay below the case directory")
        if normalized != self.relative_path:
            raise ValueError("report asset paths must use forward slashes")
        return self


class ReportManifest(StrictModel):
    schema_version: Literal[LEARNING_SCHEMA_VERSION] = LEARNING_SCHEMA_VERSION
    report_id: StableId
    case_id: StableId
    analysis_id: StableId
    stage_fingerprint: Sha256
    created_at: datetime
    final_score: float = Field(ge=0)
    max_score: float = Field(gt=0)
    longitudinal_summary: str = Field(min_length=1)
    analysis: LearningAnalysis
    claims: list[ReportClaim] = Field(default_factory=list)
    assets: list[ReportAsset] = Field(min_length=1)
    disclaimers: list[str] = Field(min_length=1)
    requires_review: bool = False

    @model_validator(mode="after")
    def validate_report(self) -> ReportManifest:
        if self.case_id != self.analysis.case_id or self.analysis_id != self.analysis.analysis_id:
            raise ValueError("report manifest references another learning analysis")
        if self.requires_review:
            raise ValueError("formal reports cannot retain review items")
        asset_names = [item.name for item in self.assets]
        asset_paths = [item.relative_path for item in self.assets]
        if len(asset_names) != len(set(asset_names)) or len(asset_paths) != len(set(asset_paths)):
            raise ValueError("report asset names and paths must be unique")
        snapshots = {item.point_id: item for item in self.analysis.snapshots}
        for claim in self.claims:
            snapshot = snapshots.get(claim.point_id)
            if snapshot is None:
                raise ValueError("report claim references a point without a snapshot")
            if not set(claim.evidence_ids) <= set(snapshot.evidence_ids):
                raise ValueError("report claim references unknown evidence")
            if claim.strength != snapshot.allowed_claim_strength:
                raise ValueError("report claim exceeds its allowed strength")
        return self


def _reject_parent_cycles(points: dict[str, KnowledgePoint]) -> None:
    for start in points:
        seen: set[str] = set()
        current: str | None = start
        while current is not None:
            if current in seen:
                raise ValueError("knowledge point parent graph contains a cycle")
            seen.add(current)
            point = points[current]
            current = point.parent_point_id
