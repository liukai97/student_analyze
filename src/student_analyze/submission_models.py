"""Phase 5 contracts for response mapping and faithful transcription."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import Field, model_validator

from student_analyze.document_models import (
    DocumentDecisionProvenance,
    DocumentRole,
    RegionKind,
)
from student_analyze.exam_master_models import (
    AnnotationActor,
    EvidenceDecisionStatus,
    QuestionType,
)
from student_analyze.models import Sha256, StableId, StrictModel
from student_analyze.page_models import Confidence, ImageSize, PixelBox


SUBMISSION_SCHEMA_VERSION = "1.0.0"
LOW_TRANSCRIPTION_CONFIDENCE = 0.8


class SubmissionSourceRole(str, Enum):
    ANSWER_SHEET = "answer_sheet"
    QUESTION_BOOKLET_SCRATCH = "question_booklet_scratch"
    SCRATCH_SHEET = "scratch_sheet"
    UNCERTAIN = "uncertain"


class VisibleContentRole(str, Enum):
    PRINTED = "printed"
    STUDENT_HANDWRITING = "student_handwriting"
    TEACHER_ANNOTATION = "teacher_annotation"
    UNCERTAIN = "uncertain"


class SubmissionQuestionPart(StrictModel):
    part_id: StableId
    printed_label: str = Field(min_length=1)
    order: int = Field(ge=1)


class SubmissionQuestionTarget(StrictModel):
    question_id: StableId
    version_id: StableId
    printed_label: str = Field(min_length=1)
    question_type: QuestionType
    option_labels: list[str] = Field(default_factory=list)
    parts: list[SubmissionQuestionPart] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_target(self) -> SubmissionQuestionTarget:
        if len(self.option_labels) != len(set(self.option_labels)):
            raise ValueError("submission target option labels must be unique")
        part_ids = [part.part_id for part in self.parts]
        part_orders = [part.order for part in self.parts]
        if len(part_ids) != len(set(part_ids)):
            raise ValueError("submission target part IDs must be unique")
        if len(part_orders) != len(set(part_orders)):
            raise ValueError("submission target part orders must be unique")
        if self.question_type in {
            QuestionType.OBJECTIVE_SINGLE,
            QuestionType.OBJECTIVE_MULTIPLE,
        } and not self.option_labels:
            raise ValueError("objective submission targets require option labels")
        return self


class SubmissionNavigationPage(StrictModel):
    page_id: StableId
    document_role: DocumentRole
    relative_path: str = Field(min_length=1)
    sha256: Sha256
    size_bytes: int = Field(gt=0)
    image_size: ImageSize

    @model_validator(mode="after")
    def validate_relative_path(self) -> SubmissionNavigationPage:
        _validate_relative_path(self.relative_path, "navigation page")
        return self


class SubmissionSourceRegion(StrictModel):
    region_id: StableId
    page_id: StableId
    kind: RegionKind
    bbox: PixelBox
    question_ids: list[StableId] = Field(default_factory=list)
    version_ids: list[StableId] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_targets(self) -> SubmissionSourceRegion:
        if len(self.question_ids) != len(set(self.question_ids)):
            raise ValueError("source-region question IDs must be unique")
        if len(self.version_ids) != len(set(self.version_ids)):
            raise ValueError("source-region version IDs must be unique")
        if len(self.question_ids) != len(self.version_ids):
            raise ValueError("source-region question and version targets must align")
        return self


class KnownAnnotationRegion(StrictModel):
    evidence_ref: StableId
    page_id: StableId
    bbox: PixelBox
    actor: AnnotationActor
    status: EvidenceDecisionStatus
    human_confirmed: bool
    requires_review: bool


class SubmissionStructureManifest(StrictModel):
    schema_version: Literal[SUBMISSION_SCHEMA_VERSION] = SUBMISSION_SCHEMA_VERSION
    case_id: StableId
    document_graph_sha256: Sha256
    page_manifest_sha256: Sha256
    exam_master_sha256: Sha256
    manifest_fingerprint: Sha256
    created_at: datetime
    questions: list[SubmissionQuestionTarget] = Field(min_length=1)
    pages: list[SubmissionNavigationPage] = Field(min_length=1)
    source_regions: list[SubmissionSourceRegion] = Field(min_length=1)
    known_annotations: list[KnownAnnotationRegion] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_manifest(self) -> SubmissionStructureManifest:
        _assert_unique(self.questions, "question_id", "structure questions")
        _assert_unique(self.pages, "page_id", "structure pages")
        _assert_unique(self.source_regions, "region_id", "structure source regions")
        _assert_unique(
            self.known_annotations, "evidence_ref", "known annotation regions"
        )
        page_ids = {page.page_id for page in self.pages}
        question_ids = {question.question_id for question in self.questions}
        version_ids = {question.version_id for question in self.questions}
        if any(region.page_id not in page_ids for region in self.source_regions):
            raise ValueError("source regions must reference navigation pages")
        if any(
            question_id not in question_ids
            for region in self.source_regions
            for question_id in region.question_ids
        ):
            raise ValueError("source regions reference unknown questions")
        if any(
            version_id not in version_ids
            for region in self.source_regions
            for version_id in region.version_ids
        ):
            raise ValueError("source regions reference unknown versions")
        if any(item.page_id not in page_ids for item in self.known_annotations):
            raise ValueError("known annotations must reference navigation pages")
        return self


class SubmissionMappingDecision(StrictModel):
    ref: StableId
    question_id: StableId
    version_id: StableId
    part_id: StableId | None = None
    slot_label: str = Field(min_length=1)
    slot_order: int = Field(ge=1)
    source_role: SubmissionSourceRole
    source_region_id: StableId | None = None
    page_id: StableId
    bbox: PixelBox
    visible_roles: list[VisibleContentRole] = Field(min_length=1)
    excluded_annotation_refs: list[StableId] = Field(default_factory=list)
    confidence: Confidence
    evidence: list[str] = Field(min_length=1)
    requires_review: bool = False
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_mapping(self) -> SubmissionMappingDecision:
        if len(self.visible_roles) != len(set(self.visible_roles)):
            raise ValueError("visible content roles must be unique")
        if len(self.excluded_annotation_refs) != len(
            set(self.excluded_annotation_refs)
        ):
            raise ValueError("excluded annotation refs must be unique")
        if (
            self.source_role == SubmissionSourceRole.UNCERTAIN
            or VisibleContentRole.UNCERTAIN in self.visible_roles
        ) and not self.requires_review:
            raise ValueError("uncertain response mappings must require review")
        if (
            VisibleContentRole.TEACHER_ANNOTATION in self.visible_roles
            and not self.excluded_annotation_refs
            and not self.requires_review
        ):
            raise ValueError(
                "new or unlinked teacher annotations must require review"
            )
        return self


class SubmissionMappingDecisionSet(StrictModel):
    schema_version: Literal[SUBMISSION_SCHEMA_VERSION] = SUBMISSION_SCHEMA_VERSION
    case_id: StableId
    structure_manifest_sha256: Sha256
    provenance: DocumentDecisionProvenance
    items: list[SubmissionMappingDecision] = Field(min_length=1)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_items(self) -> SubmissionMappingDecisionSet:
        _assert_unique(self.items, "ref", "response mapping refs")
        formal_slots = [
            (item.question_id, item.part_id, item.slot_label)
            for item in self.items
            if item.source_role == SubmissionSourceRole.ANSWER_SHEET
        ]
        if len(formal_slots) != len(set(formal_slots)):
            raise ValueError("formal response slot labels must be unique per target")
        return self


class SubmissionCropAsset(StrictModel):
    asset_id: StableId
    mapping_ref: StableId
    page_id: StableId
    page_bbox: PixelBox
    raw_bbox: PixelBox
    source_asset_id: StableId
    source_relative_path: str = Field(min_length=1)
    source_sha256: Sha256
    relative_path: str = Field(min_length=1)
    sha256: Sha256
    size_bytes: int = Field(gt=0)
    media_type: Literal["image/jpeg"] = "image/jpeg"
    image_size: ImageSize

    @model_validator(mode="after")
    def validate_paths(self) -> SubmissionCropAsset:
        _validate_relative_path(self.source_relative_path, "source asset")
        _validate_relative_path(self.relative_path, "submission crop")
        return self


class SubmissionInputItem(StrictModel):
    mapping: SubmissionMappingDecision
    crop: SubmissionCropAsset

    @model_validator(mode="after")
    def validate_item(self) -> SubmissionInputItem:
        if self.crop.mapping_ref != self.mapping.ref:
            raise ValueError("submission crop must reference its mapping decision")
        if self.crop.page_id != self.mapping.page_id:
            raise ValueError("submission crop and mapping must use the same page")
        if self.crop.page_bbox != self.mapping.bbox:
            raise ValueError("submission crop bbox must equal the mapping bbox")
        return self


class SubmissionInputManifest(StrictModel):
    schema_version: Literal[SUBMISSION_SCHEMA_VERSION] = SUBMISSION_SCHEMA_VERSION
    case_id: StableId
    structure_manifest_sha256: Sha256
    mapping_decision_sha256: Sha256
    manifest_fingerprint: Sha256
    created_at: datetime
    structure_manifest: SubmissionStructureManifest
    mapping_decisions: SubmissionMappingDecisionSet
    items: list[SubmissionInputItem] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_manifest(self) -> SubmissionInputManifest:
        if self.structure_manifest.case_id != self.case_id:
            raise ValueError("embedded submission structure belongs to another case")
        if self.mapping_decisions.case_id != self.case_id:
            raise ValueError("embedded response mappings belong to another case")
        mapping_refs = [item.mapping.ref for item in self.items]
        crop_ids = [item.crop.asset_id for item in self.items]
        crop_paths = [item.crop.relative_path for item in self.items]
        if len(mapping_refs) != len(set(mapping_refs)):
            raise ValueError("submission input mapping refs must be unique")
        if len(crop_ids) != len(set(crop_ids)):
            raise ValueError("submission input crop IDs must be unique")
        if len(crop_paths) != len(set(crop_paths)):
            raise ValueError("submission input crop paths must be unique")
        expected_refs = {item.ref for item in self.mapping_decisions.items}
        if set(mapping_refs) != expected_refs:
            raise ValueError("submission inputs must cover every mapping exactly once")
        return self


class AlternativeReading(StrictModel):
    observed_content: str = Field(min_length=1)
    normalized_answer: str | None = Field(default=None, min_length=1)
    confidence: Confidence


class SubmissionTranscriptionDecision(StrictModel):
    mapping_ref: StableId
    observed_content: str | None = Field(default=None, min_length=1)
    normalized_answer: str | None = Field(default=None, min_length=1)
    alternatives: list[AlternativeReading] = Field(default_factory=list)
    is_blank: bool
    has_erasure: bool
    confidence: Confidence
    evidence: list[str] = Field(min_length=1)
    uncertainty_notes: list[str] = Field(default_factory=list)
    requires_review: bool = False
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_transcription(self) -> SubmissionTranscriptionDecision:
        if self.is_blank:
            if self.observed_content is not None or self.normalized_answer is not None:
                raise ValueError("blank responses cannot contain transcribed answers")
            if self.alternatives:
                raise ValueError("blank responses cannot contain alternative readings")
            if self.has_erasure:
                raise ValueError("erased writing must not be recorded as a blank response")
        if self.normalized_answer is not None and self.observed_content is None:
            raise ValueError("normalized answers require observed content")
        if self.observed_content is None and not self.is_blank:
            if not self.has_erasure or not self.requires_review:
                raise ValueError(
                    "non-blank illegible responses require erasure and review"
                )
        alternative_contents = [item.observed_content for item in self.alternatives]
        if len(alternative_contents) != len(set(alternative_contents)):
            raise ValueError("alternative readings must be unique")
        if self.alternatives and not self.requires_review:
            raise ValueError("alternative readings must require review")
        if self.uncertainty_notes and not self.requires_review:
            raise ValueError("transcription uncertainty must require review")
        if self.confidence < LOW_TRANSCRIPTION_CONFIDENCE and not self.requires_review:
            raise ValueError("low-confidence transcriptions must require review")
        return self


class SubmissionTranscriptionDecisionSet(StrictModel):
    schema_version: Literal[SUBMISSION_SCHEMA_VERSION] = SUBMISSION_SCHEMA_VERSION
    case_id: StableId
    submission_input_manifest_sha256: Sha256
    provenance: DocumentDecisionProvenance
    decisions: list[SubmissionTranscriptionDecision] = Field(min_length=1)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_decisions(self) -> SubmissionTranscriptionDecisionSet:
        _assert_unique(self.decisions, "mapping_ref", "transcription mapping refs")
        return self


class SubmissionItem(StrictModel):
    item_id: StableId
    mapping_ref: StableId
    question_id: StableId
    version_id: StableId
    part_id: StableId | None = None
    slot_label: str = Field(min_length=1)
    slot_order: int = Field(ge=1)
    source_role: SubmissionSourceRole
    source_region_id: StableId | None = None
    visible_roles: list[VisibleContentRole] = Field(min_length=1)
    excluded_annotation_refs: list[StableId] = Field(default_factory=list)
    observed_content: str | None = Field(default=None, min_length=1)
    normalized_answer: str | None = Field(default=None, min_length=1)
    alternatives: list[AlternativeReading] = Field(default_factory=list)
    is_blank: bool
    has_erasure: bool
    mapping_confidence: Confidence
    transcription_confidence: Confidence
    confidence: Confidence
    mapping_evidence: list[str] = Field(min_length=1)
    transcription_evidence: list[str] = Field(min_length=1)
    uncertainty_notes: list[str] = Field(default_factory=list)
    crop: SubmissionCropAsset
    requires_review: bool
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_item(self) -> SubmissionItem:
        if self.crop.mapping_ref != self.mapping_ref:
            raise ValueError("final submission crop must reference its mapping")
        if self.confidence != min(
            self.mapping_confidence, self.transcription_confidence
        ):
            raise ValueError("final submission confidence must be conservative")
        if len(self.visible_roles) != len(set(self.visible_roles)):
            raise ValueError("final visible roles must be unique")
        if len(self.excluded_annotation_refs) != len(
            set(self.excluded_annotation_refs)
        ):
            raise ValueError("final excluded annotation refs must be unique")
        return self


class SubmissionReviewItem(StrictModel):
    item_id: StableId
    reasons: list[str] = Field(min_length=1)


class Submission(StrictModel):
    schema_version: Literal[SUBMISSION_SCHEMA_VERSION] = SUBMISSION_SCHEMA_VERSION
    case_id: StableId
    document_graph_sha256: Sha256
    page_manifest_sha256: Sha256
    exam_master_sha256: Sha256
    structure_manifest_sha256: Sha256
    mapping_decision_sha256: Sha256
    submission_input_manifest_sha256: Sha256
    transcription_decision_sha256: Sha256
    stage_fingerprint: Sha256
    created_at: datetime
    mapping_provenance: DocumentDecisionProvenance
    transcription_provenance: DocumentDecisionProvenance
    items: list[SubmissionItem] = Field(min_length=1)
    review_items: list[SubmissionReviewItem] = Field(default_factory=list)
    requires_review: bool = False
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_submission(self) -> Submission:
        _assert_unique(self.items, "item_id", "submission item IDs")
        _assert_unique(self.items, "mapping_ref", "submission mapping refs")
        _assert_unique(self.review_items, "item_id", "submission review item IDs")
        formal_slots = [
            (item.question_id, item.part_id, item.slot_label)
            for item in self.items
            if item.source_role == SubmissionSourceRole.ANSWER_SHEET
        ]
        if len(formal_slots) != len(set(formal_slots)):
            raise ValueError("final formal response slots must be unique")
        flagged = {item.item_id for item in self.items if item.requires_review}
        review_ids = {item.item_id for item in self.review_items}
        if review_ids != flagged:
            raise ValueError("submission review items must cover every flagged item")
        if self.requires_review != bool(self.review_items):
            raise ValueError("submission requires_review must reflect review items")
        return self


def _assert_unique(items: list[object], attribute: str, label: str) -> None:
    values = [getattr(item, attribute) for item in items]
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")


def _validate_relative_path(value: str, label: str) -> None:
    normalized = value.replace("\\", "/")
    parts = normalized.split("/")
    if normalized.startswith("/") or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"{label} path must stay below the case directory")
    if normalized != value:
        raise ValueError(f"{label} path must use forward slashes")
