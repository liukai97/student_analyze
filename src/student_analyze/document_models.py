"""Phase 3 contracts for semantic document decisions and resolved graphs."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import Field, model_validator

from student_analyze.models import Sha256, StableId, StrictModel
from student_analyze.page_models import Confidence, PixelBox


DOCUMENT_SCHEMA_VERSION = "1.0.0"


class DecisionMethod(str, Enum):
    CODEX_STANDARD_IMAGE_INPUT = "codex_standard_image_input"
    HUMAN_REVIEW = "human_review"
    DETERMINISTIC_TEST_FIXTURE = "deterministic_test_fixture"


class DocumentRole(str, Enum):
    QUESTION_BOOKLET = "question_booklet"
    ANSWER_SHEET = "answer_sheet"
    REPLACEMENT_OR_ERRATA = "replacement_or_errata"
    SUPPLEMENTAL_MATERIAL = "supplemental_material"
    SCRATCH = "scratch"
    OTHER = "other"
    UNKNOWN = "unknown"


class RegionKind(str, Enum):
    PRINTED_QUESTION = "printed_question"
    ANSWER_AREA = "answer_area"
    SCRATCH = "scratch"
    REPLACEMENT_NOTICE = "replacement_notice"
    OTHER = "other"


class RelationType(str, Enum):
    CONTAINS_PAGE = "contains_page"
    CONTINUED_BY = "continued_by"
    ANSWERS = "answers"
    SUPERSEDES = "supersedes"
    SCRATCH_EVIDENCE_FOR = "scratch_evidence_for"
    DERIVED_FROM = "derived_from"


class RelationDecisionStatus(str, Enum):
    ACCEPTED = "accepted"
    CANDIDATE = "candidate"


class VersionResolutionStatus(str, Enum):
    EFFECTIVE = "effective"
    SUPERSEDED = "superseded"
    UNRESOLVED = "unresolved"


class ReviewItemKind(str, Enum):
    DOCUMENT = "document"
    PAGE = "page"
    QUESTION = "question"
    QUESTION_VERSION = "question_version"
    REGION = "region"
    RELATION = "relation"
    VERSION_RESOLUTION = "version_resolution"


class DocumentDecisionProvenance(StrictModel):
    method: DecisionMethod
    model_identifier: str | None = None
    model_identifier_unavailable_reason: str | None = None
    prompt_version: str = Field(min_length=1)
    skill_version: str = Field(min_length=1)
    decided_at: datetime

    @model_validator(mode="after")
    def validate_model_identifier(self) -> DocumentDecisionProvenance:
        if self.model_identifier is None and not self.model_identifier_unavailable_reason:
            raise ValueError(
                "an unavailable reason is required when model_identifier is absent"
            )
        if self.model_identifier is not None and self.model_identifier_unavailable_reason:
            raise ValueError(
                "model_identifier_unavailable_reason must be absent when an identifier exists"
            )
        return self


class DocumentDecision(StrictModel):
    ref: StableId
    role: DocumentRole
    title: str | None = None
    confidence: Confidence
    evidence: list[str] = Field(min_length=1)
    requires_review: bool = False
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unknown_role(self) -> DocumentDecision:
        if self.role == DocumentRole.UNKNOWN and not self.requires_review:
            raise ValueError("unknown document roles must require review")
        return self


class PageClassificationDecision(StrictModel):
    page_id: StableId
    document_ref: StableId
    order: int = Field(ge=1)
    printed_page_number: str | None = None
    confidence: Confidence
    evidence: list[str] = Field(min_length=1)
    requires_review: bool = False
    warnings: list[str] = Field(default_factory=list)


class QuestionDecision(StrictModel):
    ref: StableId
    printed_label: str = Field(min_length=1)
    parent_ref: StableId | None = None
    order: int = Field(ge=1)
    confidence: Confidence
    evidence: list[str] = Field(min_length=1)
    requires_review: bool = False
    warnings: list[str] = Field(default_factory=list)


class RegionDecision(StrictModel):
    ref: StableId
    page_id: StableId
    kind: RegionKind
    bbox: PixelBox
    order: int = Field(ge=1)
    confidence: Confidence
    evidence: list[str] = Field(min_length=1)
    requires_review: bool = False
    warnings: list[str] = Field(default_factory=list)


class QuestionVersionDecision(StrictModel):
    ref: StableId
    question_ref: StableId
    label: str = Field(min_length=1)
    region_refs: list[StableId] = Field(min_length=1)
    confidence: Confidence
    evidence: list[str] = Field(min_length=1)
    requires_review: bool = False
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_region_refs(self) -> QuestionVersionDecision:
        if len(self.region_refs) != len(set(self.region_refs)):
            raise ValueError("question version region_refs must be unique")
        return self


class RelationDecision(StrictModel):
    type: RelationType
    from_ref: StableId
    to_ref: StableId
    status: RelationDecisionStatus
    confidence: Confidence
    evidence: list[str] = Field(min_length=1)
    requires_review: bool

    @model_validator(mode="after")
    def validate_status(self) -> RelationDecision:
        if self.from_ref == self.to_ref:
            raise ValueError("relationship endpoints must differ")
        if self.status == RelationDecisionStatus.ACCEPTED and self.requires_review:
            raise ValueError("accepted relationships cannot require review")
        if self.status == RelationDecisionStatus.CANDIDATE and not self.requires_review:
            raise ValueError("candidate relationships must require review")
        if self.type in {RelationType.CONTAINS_PAGE, RelationType.DERIVED_FROM}:
            raise ValueError(
                "contains_page and derived_from are deterministic and cannot be decisions"
            )
        return self


class DocumentGraphDecisionSet(StrictModel):
    schema_version: Literal[DOCUMENT_SCHEMA_VERSION] = DOCUMENT_SCHEMA_VERSION
    case_id: StableId
    page_manifest_sha256: Sha256
    provenance: DocumentDecisionProvenance
    documents: list[DocumentDecision] = Field(min_length=1)
    pages: list[PageClassificationDecision] = Field(min_length=1)
    questions: list[QuestionDecision] = Field(min_length=1)
    question_versions: list[QuestionVersionDecision] = Field(min_length=1)
    regions: list[RegionDecision] = Field(min_length=1)
    relations: list[RelationDecision] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_graph_decisions(self) -> DocumentGraphDecisionSet:
        documents = _unique_by_ref(self.documents, "document")
        questions = _unique_by_ref(self.questions, "question")
        versions = _unique_by_ref(self.question_versions, "question version")
        regions = _unique_by_ref(self.regions, "region")

        page_ids = [page.page_id for page in self.pages]
        if len(page_ids) != len(set(page_ids)):
            raise ValueError("page classifications must contain unique page_id values")
        page_orders = [(page.document_ref, page.order) for page in self.pages]
        if len(page_orders) != len(set(page_orders)):
            raise ValueError("page order must be unique within each document")
        for page in self.pages:
            if page.document_ref not in documents:
                raise ValueError(
                    f"page {page.page_id} references unknown document {page.document_ref}"
                )

        all_refs = [*documents, *questions, *versions, *regions]
        if len(all_refs) != len(set(all_refs)):
            raise ValueError("document, question, version, and region refs must not overlap")

        for question in self.questions:
            if question.parent_ref is not None and question.parent_ref not in questions:
                raise ValueError(
                    f"question {question.ref} references unknown parent {question.parent_ref}"
                )
        _assert_acyclic(
            {question.ref: [question.parent_ref] if question.parent_ref else [] for question in self.questions},
            "question hierarchy",
        )
        question_orders = [
            (question.parent_ref, question.order) for question in self.questions
        ]
        if len(question_orders) != len(set(question_orders)):
            raise ValueError("question order must be unique among siblings")

        page_id_set = set(page_ids)
        for region in self.regions:
            if region.page_id not in page_id_set:
                raise ValueError(
                    f"region {region.ref} references unknown page {region.page_id}"
                )

        versions_by_question: dict[str, list[str]] = {}
        region_to_version: dict[str, str] = {}
        for version in self.question_versions:
            if version.question_ref not in questions:
                raise ValueError(
                    f"version {version.ref} references unknown question {version.question_ref}"
                )
            versions_by_question.setdefault(version.question_ref, []).append(version.ref)
            for region_ref in version.region_refs:
                region = regions.get(region_ref)
                if region is None:
                    raise ValueError(
                        f"version {version.ref} references unknown region {region_ref}"
                    )
                if region.kind != RegionKind.PRINTED_QUESTION:
                    raise ValueError(
                        f"version {version.ref} must use printed_question regions"
                    )
                if region_ref in region_to_version:
                    raise ValueError(
                        f"printed question region {region_ref} belongs to multiple versions"
                    )
                region_to_version[region_ref] = version.ref
        questions_without_versions = sorted(set(questions) - set(versions_by_question))
        if questions_without_versions:
            raise ValueError(
                f"every question requires a version; missing={questions_without_versions}"
            )

        relationship_keys: list[tuple[RelationType, str, str]] = []
        accepted_supersedes: dict[str, list[str]] = {ref: [] for ref in versions}
        for relation in self.relations:
            relationship_keys.append((relation.type, relation.from_ref, relation.to_ref))
            _validate_relation_endpoints(
                relation,
                questions=questions,
                versions=versions,
                regions=regions,
                region_to_version=region_to_version,
            )
            if (
                relation.type == RelationType.SUPERSEDES
                and relation.status == RelationDecisionStatus.ACCEPTED
            ):
                accepted_supersedes[relation.from_ref].append(relation.to_ref)
        if len(relationship_keys) != len(set(relationship_keys)):
            raise ValueError("relationship decisions must be unique by type and endpoints")
        _assert_acyclic(accepted_supersedes, "accepted supersedes graph")
        return self


class DocumentNode(StrictModel):
    document_id: StableId
    decision_ref: StableId
    role: DocumentRole
    title: str | None = None
    page_ids: list[StableId] = Field(min_length=1)
    confidence: Confidence
    evidence: list[str] = Field(min_length=1)
    requires_review: bool = False
    warnings: list[str] = Field(default_factory=list)


class PageNode(StrictModel):
    page_id: StableId
    document_id: StableId
    order: int = Field(ge=1)
    printed_page_number: str | None = None
    confidence: Confidence
    evidence: list[str] = Field(min_length=1)
    requires_review: bool = False
    warnings: list[str] = Field(default_factory=list)


class QuestionNode(StrictModel):
    question_id: StableId
    decision_ref: StableId
    printed_label: str = Field(min_length=1)
    parent_question_id: StableId | None = None
    order: int = Field(ge=1)
    effective_version_id: StableId | None = None
    confidence: Confidence
    evidence: list[str] = Field(min_length=1)
    requires_review: bool = False
    warnings: list[str] = Field(default_factory=list)


class QuestionVersionNode(StrictModel):
    version_id: StableId
    decision_ref: StableId
    question_id: StableId
    label: str = Field(min_length=1)
    region_ids: list[StableId] = Field(min_length=1)
    resolution_status: VersionResolutionStatus
    confidence: Confidence
    evidence: list[str] = Field(min_length=1)
    requires_review: bool = False
    warnings: list[str] = Field(default_factory=list)


class EvidenceRegion(StrictModel):
    region_id: StableId
    decision_ref: StableId
    page_id: StableId
    kind: RegionKind
    bbox: PixelBox
    order: int = Field(ge=1)
    confidence: Confidence
    evidence: list[str] = Field(min_length=1)
    requires_review: bool = False
    warnings: list[str] = Field(default_factory=list)


class GraphRelation(StrictModel):
    type: RelationType
    from_id: StableId
    to_id: StableId
    confidence: Confidence
    evidence: list[str] = Field(min_length=1)


class UnresolvedRelation(StrictModel):
    type: RelationType
    from_id: StableId
    to_id: StableId
    confidence: Confidence
    evidence: list[str] = Field(min_length=1)
    reason: str = Field(min_length=1)


class ReviewItem(StrictModel):
    kind: ReviewItemKind
    ref: StableId
    reason: str = Field(min_length=1)
    blocks_master_ready: bool
    blocks_submission_ready: bool


class DocumentGraph(StrictModel):
    schema_version: Literal[DOCUMENT_SCHEMA_VERSION] = DOCUMENT_SCHEMA_VERSION
    case_id: StableId
    page_manifest_sha256: Sha256
    stage_fingerprint: Sha256
    decision_sha256: Sha256
    provenance: DocumentDecisionProvenance
    created_at: datetime
    requires_review: bool = False
    documents: list[DocumentNode] = Field(min_length=1)
    pages: list[PageNode] = Field(min_length=1)
    questions: list[QuestionNode] = Field(min_length=1)
    question_versions: list[QuestionVersionNode] = Field(min_length=1)
    regions: list[EvidenceRegion] = Field(min_length=1)
    relationships: list[GraphRelation] = Field(default_factory=list)
    unresolved_relations: list[UnresolvedRelation] = Field(default_factory=list)
    review_items: list[ReviewItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_resolved_graph(self) -> DocumentGraph:
        documents = _unique_by_id(self.documents, "document_id", "document")
        pages = _unique_by_id(self.pages, "page_id", "page")
        questions = _unique_by_id(self.questions, "question_id", "question")
        versions = _unique_by_id(
            self.question_versions, "version_id", "question version"
        )
        regions = _unique_by_id(self.regions, "region_id", "region")

        node_ids = [*documents, *pages, *questions, *versions, *regions]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("resolved graph node IDs must not overlap")

        for document in self.documents:
            if len(document.page_ids) != len(set(document.page_ids)):
                raise ValueError(f"document {document.document_id} has duplicate pages")
            if any(page_id not in pages for page_id in document.page_ids):
                raise ValueError(f"document {document.document_id} references unknown page")
        for page in self.pages:
            if page.document_id not in documents:
                raise ValueError(f"page {page.page_id} references unknown document")
        for question in self.questions:
            if (
                question.parent_question_id is not None
                and question.parent_question_id not in questions
            ):
                raise ValueError(f"question {question.question_id} has unknown parent")
            if (
                question.effective_version_id is not None
                and question.effective_version_id not in versions
            ):
                raise ValueError(f"question {question.question_id} has unknown effective version")
        for version in self.question_versions:
            if version.question_id not in questions:
                raise ValueError(f"version {version.version_id} references unknown question")
            if any(region_id not in regions for region_id in version.region_ids):
                raise ValueError(f"version {version.version_id} references unknown region")
        for region in self.regions:
            if region.page_id not in pages:
                raise ValueError(f"region {region.region_id} references unknown page")

        versions_by_question: dict[str, list[QuestionVersionNode]] = {}
        for version in self.question_versions:
            versions_by_question.setdefault(version.question_id, []).append(version)
        for question in self.questions:
            members = versions_by_question.get(question.question_id, [])
            effective = [
                version
                for version in members
                if version.resolution_status == VersionResolutionStatus.EFFECTIVE
            ]
            if question.effective_version_id is None:
                if effective:
                    raise ValueError(
                        f"question {question.question_id} omits its effective version"
                    )
                if not any(
                    version.resolution_status == VersionResolutionStatus.UNRESOLVED
                    for version in members
                ):
                    raise ValueError(
                        f"question {question.question_id} without an effective version must be unresolved"
                    )
            elif len(effective) != 1 or effective[0].version_id != question.effective_version_id:
                raise ValueError(
                    f"question {question.question_id} effective version is inconsistent"
                )

        region_to_version: dict[str, str] = {}
        for version in self.question_versions:
            for region_id in version.region_ids:
                if region_id in region_to_version:
                    raise ValueError(
                        f"printed question region {region_id} belongs to multiple versions"
                    )
                region_to_version[region_id] = version.version_id

        relationship_keys = [
            (relation.type, relation.from_id, relation.to_id)
            for relation in self.relationships
        ]
        if len(relationship_keys) != len(set(relationship_keys)):
            raise ValueError("resolved relationships must be unique")
        accepted_supersedes: dict[str, list[str]] = {
            version_id: [] for version_id in versions
        }
        for relation in self.relationships:
            if relation.from_id not in node_ids or relation.to_id not in node_ids:
                raise ValueError("resolved relationship has an unknown endpoint")
            _validate_resolved_relation_endpoints(
                relation,
                documents=documents,
                pages=pages,
                versions=versions,
                regions=regions,
                region_to_version=region_to_version,
            )
            if relation.type == RelationType.SUPERSEDES:
                accepted_supersedes[relation.from_id].append(relation.to_id)
        _assert_acyclic(accepted_supersedes, "resolved supersedes graph")

        unresolved_keys = [
            (relation.type, relation.from_id, relation.to_id)
            for relation in self.unresolved_relations
        ]
        if len(unresolved_keys) != len(set(unresolved_keys)):
            raise ValueError("unresolved relationships must be unique")
        for relation in self.unresolved_relations:
            if relation.from_id not in node_ids or relation.to_id not in node_ids:
                raise ValueError("unresolved relationship has an unknown endpoint")
            _validate_resolved_relation_endpoints(
                relation,
                documents=documents,
                pages=pages,
                versions=versions,
                regions=regions,
                region_to_version=region_to_version,
            )

        unresolved_supersedes_endpoints = {
            endpoint
            for relation in self.unresolved_relations
            if relation.type == RelationType.SUPERSEDES
            for endpoint in (relation.from_id, relation.to_id)
        }
        for question in self.questions:
            members = versions_by_question[question.question_id]
            member_ids = {version.version_id for version in members}
            targets = {
                target
                for source in member_ids
                for target in accepted_supersedes.get(source, [])
            }
            roots = member_ids - targets
            has_candidate = bool(member_ids & unresolved_supersedes_endpoints)
            expected_effective = (
                next(iter(member_ids))
                if len(member_ids) == 1 and not has_candidate
                else next(iter(roots))
                if len(roots) == 1 and not has_candidate
                else None
            )
            if question.effective_version_id != expected_effective:
                raise ValueError(
                    f"question {question.question_id} resolution differs from supersedes graph"
                )
            for version in members:
                expected_status = (
                    VersionResolutionStatus.EFFECTIVE
                    if version.version_id == expected_effective
                    else VersionResolutionStatus.SUPERSEDED
                    if version.version_id in targets
                    else VersionResolutionStatus.UNRESOLVED
                )
                if version.resolution_status != expected_status:
                    raise ValueError(
                        f"version {version.version_id} status differs from supersedes graph"
                    )

        if self.requires_review != bool(self.review_items):
            raise ValueError("requires_review must reflect review_items")
        if self.unresolved_relations and not self.requires_review:
            raise ValueError("unresolved relationships must require review")
        return self


def _unique_by_ref(items, label: str) -> dict[str, object]:
    refs = [item.ref for item in items]
    if len(refs) != len(set(refs)):
        raise ValueError(f"{label} refs must be unique")
    return {item.ref: item for item in items}


def _unique_by_id(items, field: str, label: str) -> dict[str, object]:
    ids = [getattr(item, field) for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label} IDs must be unique")
    return {getattr(item, field): item for item in items}


def _validate_relation_endpoints(
    relation: RelationDecision,
    *,
    questions: dict[str, QuestionDecision],
    versions: dict[str, QuestionVersionDecision],
    regions: dict[str, RegionDecision],
    region_to_version: dict[str, str],
) -> None:
    if relation.type == RelationType.SUPERSEDES:
        source = versions.get(relation.from_ref)
        target = versions.get(relation.to_ref)
        if source is None or target is None:
            raise ValueError("supersedes endpoints must be question versions")
        if source.question_ref != target.question_ref:
            raise ValueError("supersedes endpoints must belong to the same question")
        return
    if relation.type == RelationType.CONTINUED_BY:
        source = regions.get(relation.from_ref)
        target = regions.get(relation.to_ref)
        if (
            source is None
            or target is None
            or source.kind != RegionKind.PRINTED_QUESTION
            or target.kind != RegionKind.PRINTED_QUESTION
        ):
            raise ValueError("continued_by endpoints must be printed question regions")
        if region_to_version.get(source.ref) != region_to_version.get(target.ref):
            raise ValueError("continued_by regions must belong to the same question version")
        return
    if relation.type == RelationType.ANSWERS:
        source = regions.get(relation.from_ref)
        if source is None or source.kind != RegionKind.ANSWER_AREA:
            raise ValueError("answers must start from an answer_area region")
        if relation.to_ref not in versions:
            raise ValueError("answers must target a concrete question version")
        return
    if relation.type == RelationType.SCRATCH_EVIDENCE_FOR:
        source = regions.get(relation.from_ref)
        if source is None or source.kind != RegionKind.SCRATCH:
            raise ValueError("scratch_evidence_for must start from a scratch region")
        if relation.to_ref not in versions:
            raise ValueError("scratch_evidence_for must target a question version")
        return
    raise ValueError(f"unsupported semantic relationship type: {relation.type.value}")


def _validate_resolved_relation_endpoints(
    relation: GraphRelation | UnresolvedRelation,
    *,
    documents: dict[str, DocumentNode],
    pages: dict[str, PageNode],
    versions: dict[str, QuestionVersionNode],
    regions: dict[str, EvidenceRegion],
    region_to_version: dict[str, str],
) -> None:
    if relation.type == RelationType.CONTAINS_PAGE:
        if relation.from_id not in documents or relation.to_id not in pages:
            raise ValueError("contains_page must connect a document to a page")
        if relation.to_id not in documents[relation.from_id].page_ids:
            raise ValueError("contains_page differs from document page_ids")
        return
    if relation.type == RelationType.DERIVED_FROM:
        source = regions.get(relation.from_id)
        if source is None or relation.to_id not in pages:
            raise ValueError("derived_from must connect a region to a page")
        if source.page_id != relation.to_id:
            raise ValueError("derived_from differs from region page_id")
        return
    if relation.type == RelationType.SUPERSEDES:
        source = versions.get(relation.from_id)
        target = versions.get(relation.to_id)
        if source is None or target is None:
            raise ValueError("supersedes must connect question versions")
        if source.question_id != target.question_id:
            raise ValueError("supersedes versions must belong to the same question")
        return
    if relation.type == RelationType.CONTINUED_BY:
        source = regions.get(relation.from_id)
        target = regions.get(relation.to_id)
        if (
            source is None
            or target is None
            or source.kind != RegionKind.PRINTED_QUESTION
            or target.kind != RegionKind.PRINTED_QUESTION
        ):
            raise ValueError("continued_by must connect printed question regions")
        if region_to_version.get(source.region_id) != region_to_version.get(target.region_id):
            raise ValueError("continued_by regions must belong to the same version")
        return
    if relation.type == RelationType.ANSWERS:
        source = regions.get(relation.from_id)
        if source is None or source.kind != RegionKind.ANSWER_AREA:
            raise ValueError("answers must start from an answer area")
        if relation.to_id not in versions:
            raise ValueError("answers must target a question version")
        return
    if relation.type == RelationType.SCRATCH_EVIDENCE_FOR:
        source = regions.get(relation.from_id)
        if source is None or source.kind != RegionKind.SCRATCH:
            raise ValueError("scratch_evidence_for must start from scratch")
        if relation.to_id not in versions:
            raise ValueError("scratch_evidence_for must target a question version")
        return
    raise ValueError(f"unsupported relationship type: {relation.type.value}")


def _assert_acyclic(adjacency: dict[str, list[str]], label: str) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise ValueError(f"{label} must be acyclic")
        if node in visited:
            return
        visiting.add(node)
        for target in adjacency.get(node, []):
            visit(target)
        visiting.remove(node)
        visited.add(node)

    for node in adjacency:
        visit(node)
