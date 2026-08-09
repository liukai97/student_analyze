"""Compile reviewed phase 3 semantic decisions into a validated document graph."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Iterable

from student_analyze import __version__
from student_analyze.assets import artifact_digest
from student_analyze.config import AppConfig
from student_analyze.document_models import (
    DocumentGraph,
    DocumentGraphDecisionSet,
    DocumentNode,
    EvidenceRegion,
    GraphRelation,
    PageNode,
    QuestionNode,
    QuestionVersionNode,
    RegionKind,
    RelationDecisionStatus,
    RelationType,
    ReviewItem,
    ReviewItemKind,
    UnresolvedRelation,
    VersionResolutionStatus,
)
from student_analyze.errors import CaseValidationError, InvalidTransitionError
from student_analyze.fingerprint import digest_value
from student_analyze.models import (
    SCHEMA_VERSION,
    ImplementationVersions,
    PipelineStage,
    PipelineState,
)
from student_analyze.page_models import PageManifest
from student_analyze.page_verification import read_page_manifest
from student_analyze.pipeline import (
    build_stage_fingerprint,
    commit_stage_artifact,
    verify_case,
)
from student_analyze.validation import validate_json


InterruptHook = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class DocumentMappingResult:
    case_dir: Path
    graph: DocumentGraph
    state: PipelineState
    reused: bool


def load_document_decisions(path: Path) -> DocumentGraphDecisionSet:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise CaseValidationError(f"cannot read document decision file {path}: {exc}") from exc
    return validate_json(DocumentGraphDecisionSet, raw)


def read_document_graph(path: Path) -> DocumentGraph:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise CaseValidationError(f"cannot read document graph {path}: {exc}") from exc
    return validate_json(DocumentGraph, raw)


def map_documents(
    case_dir: Path,
    decisions: DocumentGraphDecisionSet,
    config: AppConfig,
    *,
    force: bool = False,
    interrupt_hook: InterruptHook | None = None,
) -> DocumentMappingResult:
    case_dir = case_dir.resolve(strict=True)
    case_manifest, state = verify_case(case_dir)
    page_manifest_path, page_manifest = _active_page_manifest(case_dir, state)
    page_manifest_sha256, _ = artifact_digest(page_manifest_path)
    _validate_decisions(decisions, page_manifest, page_manifest_sha256)

    versions = ImplementationVersions(
        code=__version__,
        base_schema=SCHEMA_VERSION,
        config=config.config_version,
        model=decisions.provenance.model_identifier,
        prompt=decisions.provenance.prompt_version,
        skill=decisions.provenance.skill_version,
        tools={},
    )
    decision_sha256 = digest_value(decisions.model_dump(mode="json"))
    stage_inputs = [
        {
            "page_manifest_sha256": page_manifest_sha256,
            "page_stage_fingerprint": page_manifest.stage_fingerprint,
            "decision_sha256": decision_sha256,
        }
    ]
    stage_config = config.mapping_fingerprint_payload()
    stage_fingerprint, _ = build_stage_fingerprint(
        stage=PipelineStage.MAPPED,
        model_type=DocumentGraph,
        schema_id="document_graph.schema.json",
        config=stage_config,
        versions=versions,
        inputs=stage_inputs,
    )

    existing_completion = next(
        (
            completion
            for completion in state.completed_stages
            if completion.stage == PipelineStage.MAPPED
        ),
        None,
    )
    if existing_completion is not None:
        if existing_completion.stage_fingerprint == stage_fingerprint and not force:
            reference = next(
                reference
                for reference in existing_completion.artifacts
                if reference.schema_id == "document_graph.schema.json"
            )
            graph = read_document_graph(case_dir / reference.relative_path)
            verify_document_graph(
                graph,
                page_manifest,
                page_manifest_sha256=page_manifest_sha256,
            )
            return DocumentMappingResult(case_dir, graph, state, reused=True)
        if not force:
            raise InvalidTransitionError(
                "mapped is already complete with a different fingerprint; "
                "use force to create a preserved version"
            )
    elif state.current_stage != PipelineStage.PAGES_READY:
        raise InvalidTransitionError(
            f"cannot map documents from current stage {state.current_stage}"
        )

    graph = _compile_graph(
        decisions,
        page_manifest_sha256=page_manifest_sha256,
        stage_fingerprint=stage_fingerprint,
        decision_sha256=decision_sha256,
    )
    orphan_path = (
        case_dir
        / "artifacts"
        / PipelineStage.MAPPED.value
        / stage_fingerprint
        / "document_graph.json"
    )
    if orphan_path.exists() and not force:
        orphan = read_document_graph(orphan_path)
        if (
            orphan.case_id != graph.case_id
            or orphan.stage_fingerprint != graph.stage_fingerprint
            or orphan.decision_sha256 != graph.decision_sha256
            or orphan.page_manifest_sha256 != graph.page_manifest_sha256
        ):
            raise CaseValidationError(
                f"conflicting uncommitted document graph at {orphan_path}"
            )
        graph = orphan

    verify_document_graph(
        graph,
        page_manifest,
        page_manifest_sha256=page_manifest_sha256,
    )
    committed = commit_stage_artifact(
        case_dir,
        stage=PipelineStage.MAPPED,
        artifact_name="document_graph.json",
        payload=graph,
        model_type=DocumentGraph,
        schema_id="document_graph.schema.json",
        config=stage_config,
        versions=versions,
        inputs=stage_inputs,
        force=force,
        human_confirmed=(
            decisions.provenance.method.value == "human_review"
        ),
        interrupt_hook=interrupt_hook,
    )
    active_reference = committed.artifacts[0]
    active_graph = read_document_graph(case_dir / active_reference.relative_path)
    return DocumentMappingResult(
        case_dir,
        active_graph,
        committed.state,
        reused=committed.reused,
    )


def verify_document_graph(
    graph: DocumentGraph,
    page_manifest: PageManifest,
    *,
    page_manifest_sha256: str,
) -> None:
    if graph.case_id != page_manifest.case_id:
        raise CaseValidationError("document graph and page manifest use different case_id values")
    if graph.page_manifest_sha256 != page_manifest_sha256:
        raise CaseValidationError("document graph references a different page manifest")

    manifest_pages = {page.page_id: page for page in page_manifest.pages}
    graph_pages = {page.page_id: page for page in graph.pages}
    if set(graph_pages) != set(manifest_pages):
        missing = sorted(set(manifest_pages) - set(graph_pages))
        unexpected = sorted(set(graph_pages) - set(manifest_pages))
        raise CaseValidationError(
            f"document graph page coverage mismatch; missing={missing}, unexpected={unexpected}"
        )

    for region in graph.regions:
        page = manifest_pages[region.page_id]
        if (
            region.bbox.right > page.page_image_size.width
            or region.bbox.bottom > page.page_image_size.height
        ):
            raise CaseValidationError(
                f"region {region.region_id} exceeds logical page {region.page_id}"
            )

    expected_contains = {
        (document.document_id, page_id)
        for document in graph.documents
        for page_id in document.page_ids
    }
    actual_contains = {
        (relation.from_id, relation.to_id)
        for relation in graph.relationships
        if relation.type == RelationType.CONTAINS_PAGE
    }
    if actual_contains != expected_contains:
        raise CaseValidationError("contains_page relationships differ from document membership")

    expected_derived = {
        (region.region_id, region.page_id) for region in graph.regions
    }
    actual_derived = {
        (relation.from_id, relation.to_id)
        for relation in graph.relationships
        if relation.type == RelationType.DERIVED_FROM
    }
    if actual_derived != expected_derived:
        raise CaseValidationError("derived_from relationships differ from region provenance")


def _active_page_manifest(
    case_dir: Path, state: PipelineState
) -> tuple[Path, PageManifest]:
    completion = next(
        (
            item
            for item in state.completed_stages
            if item.stage == PipelineStage.PAGES_READY
        ),
        None,
    )
    if completion is None:
        raise InvalidTransitionError("cannot map documents before pages_ready")
    references = [
        reference
        for reference in completion.artifacts
        if reference.schema_id == "page_manifest.schema.json"
    ]
    if len(references) != 1:
        raise CaseValidationError("pages_ready must have exactly one active page manifest")
    path = case_dir / references[0].relative_path
    return path, read_page_manifest(path)


def _validate_decisions(
    decisions: DocumentGraphDecisionSet,
    page_manifest: PageManifest,
    page_manifest_sha256: str,
) -> None:
    if decisions.case_id != page_manifest.case_id:
        raise CaseValidationError("document decisions belong to a different case")
    if decisions.page_manifest_sha256 != page_manifest_sha256:
        raise CaseValidationError("document decisions reference a different page manifest")

    manifest_pages = {page.page_id: page for page in page_manifest.pages}
    decision_page_ids = {page.page_id for page in decisions.pages}
    if decision_page_ids != set(manifest_pages):
        missing = sorted(set(manifest_pages) - decision_page_ids)
        unexpected = sorted(decision_page_ids - set(manifest_pages))
        raise CaseValidationError(
            f"document decisions must classify every logical page exactly once; "
            f"missing={missing}, unexpected={unexpected}"
        )
    document_refs = {document.ref for document in decisions.documents}
    referenced_documents = {page.document_ref for page in decisions.pages}
    unused_documents = sorted(document_refs - referenced_documents)
    if unused_documents:
        raise CaseValidationError(
            f"document decisions contain documents without pages: {unused_documents}"
        )
    for region in decisions.regions:
        page = manifest_pages[region.page_id]
        if (
            region.bbox.right > page.page_image_size.width
            or region.bbox.bottom > page.page_image_size.height
        ):
            raise CaseValidationError(
                f"region {region.ref} exceeds logical page {region.page_id}"
            )


def _compile_graph(
    decisions: DocumentGraphDecisionSet,
    *,
    page_manifest_sha256: str,
    stage_fingerprint: str,
    decision_sha256: str,
) -> DocumentGraph:
    document_ids = {
        item.ref: _stable_graph_id("document", decisions.case_id, item.ref)
        for item in decisions.documents
    }
    question_ids = {
        item.ref: _stable_graph_id("question", decisions.case_id, item.ref)
        for item in decisions.questions
    }
    version_ids = {
        item.ref: _stable_graph_id("version", decisions.case_id, item.ref)
        for item in decisions.question_versions
    }
    region_ids = {
        item.ref: _stable_graph_id("region", decisions.case_id, item.ref)
        for item in decisions.regions
    }
    all_ids = {**document_ids, **question_ids, **version_ids, **region_ids}

    pages_by_document: dict[str, list] = {}
    for page in decisions.pages:
        pages_by_document.setdefault(page.document_ref, []).append(page)

    documents = [
        DocumentNode(
            document_id=document_ids[item.ref],
            decision_ref=item.ref,
            role=item.role,
            title=item.title,
            page_ids=[
                page.page_id
                for page in sorted(pages_by_document[item.ref], key=lambda page: page.order)
            ],
            confidence=item.confidence,
            evidence=item.evidence,
            requires_review=item.requires_review,
            warnings=item.warnings,
        )
        for item in decisions.documents
    ]
    pages = [
        PageNode(
            page_id=item.page_id,
            document_id=document_ids[item.document_ref],
            order=item.order,
            printed_page_number=item.printed_page_number,
            confidence=item.confidence,
            evidence=item.evidence,
            requires_review=item.requires_review,
            warnings=item.warnings,
        )
        for item in decisions.pages
    ]
    regions = [
        EvidenceRegion(
            region_id=region_ids[item.ref],
            decision_ref=item.ref,
            page_id=item.page_id,
            kind=item.kind,
            bbox=item.bbox,
            order=item.order,
            confidence=item.confidence,
            evidence=item.evidence,
            requires_review=item.requires_review,
            warnings=item.warnings,
        )
        for item in decisions.regions
    ]

    accepted_supersedes = [
        relation
        for relation in decisions.relations
        if relation.type == RelationType.SUPERSEDES
        and relation.status == RelationDecisionStatus.ACCEPTED
    ]
    candidate_supersedes_refs = {
        endpoint
        for relation in decisions.relations
        if relation.type == RelationType.SUPERSEDES
        and relation.status == RelationDecisionStatus.CANDIDATE
        for endpoint in (relation.from_ref, relation.to_ref)
    }
    versions_by_question: dict[str, list] = {}
    for version in decisions.question_versions:
        versions_by_question.setdefault(version.question_ref, []).append(version)

    effective_by_question: dict[str, str | None] = {}
    version_status: dict[str, VersionResolutionStatus] = {}
    for question_ref, versions_for_question in versions_by_question.items():
        refs = {version.ref for version in versions_for_question}
        targets = {
            relation.to_ref
            for relation in accepted_supersedes
            if relation.from_ref in refs
        }
        roots = refs - targets
        has_candidate = bool(refs & candidate_supersedes_refs)
        if len(refs) == 1 and not has_candidate:
            effective = next(iter(refs))
        elif len(roots) == 1 and not has_candidate:
            effective = next(iter(roots))
        else:
            effective = None
        effective_by_question[question_ref] = effective
        for version_ref in refs:
            if version_ref == effective:
                version_status[version_ref] = VersionResolutionStatus.EFFECTIVE
            elif version_ref in targets:
                version_status[version_ref] = VersionResolutionStatus.SUPERSEDED
            else:
                version_status[version_ref] = VersionResolutionStatus.UNRESOLVED

    questions = [
        QuestionNode(
            question_id=question_ids[item.ref],
            decision_ref=item.ref,
            printed_label=item.printed_label,
            parent_question_id=(
                question_ids[item.parent_ref] if item.parent_ref is not None else None
            ),
            order=item.order,
            effective_version_id=(
                version_ids[effective_by_question[item.ref]]
                if effective_by_question[item.ref] is not None
                else None
            ),
            confidence=item.confidence,
            evidence=item.evidence,
            requires_review=(
                item.requires_review or effective_by_question[item.ref] is None
            ),
            warnings=item.warnings,
        )
        for item in decisions.questions
    ]
    question_versions = [
        QuestionVersionNode(
            version_id=version_ids[item.ref],
            decision_ref=item.ref,
            question_id=question_ids[item.question_ref],
            label=item.label,
            region_ids=[region_ids[ref] for ref in item.region_refs],
            resolution_status=version_status[item.ref],
            confidence=item.confidence,
            evidence=item.evidence,
            requires_review=(
                item.requires_review
                or version_status[item.ref] == VersionResolutionStatus.UNRESOLVED
            ),
            warnings=item.warnings,
        )
        for item in decisions.question_versions
    ]

    relationships: list[GraphRelation] = []
    for document in documents:
        relationships.extend(
            GraphRelation(
                type=RelationType.CONTAINS_PAGE,
                from_id=document.document_id,
                to_id=page_id,
                confidence=1.0,
                evidence=["Derived from the validated page-to-document assignment."],
            )
            for page_id in document.page_ids
        )
    relationships.extend(
        GraphRelation(
            type=RelationType.DERIVED_FROM,
            from_id=region.region_id,
            to_id=region.page_id,
            confidence=1.0,
            evidence=["Derived from the validated region page_id and bbox."],
        )
        for region in regions
    )
    relationships.extend(
        GraphRelation(
            type=relation.type,
            from_id=all_ids[relation.from_ref],
            to_id=all_ids[relation.to_ref],
            confidence=relation.confidence,
            evidence=relation.evidence,
        )
        for relation in decisions.relations
        if relation.status == RelationDecisionStatus.ACCEPTED
    )
    unresolved_relations = [
        UnresolvedRelation(
            type=relation.type,
            from_id=all_ids[relation.from_ref],
            to_id=all_ids[relation.to_ref],
            confidence=relation.confidence,
            evidence=relation.evidence,
            reason="Semantic relationship requires review before acceptance.",
        )
        for relation in decisions.relations
        if relation.status == RelationDecisionStatus.CANDIDATE
    ]

    review_items = _build_review_items(
        decisions,
        documents=documents,
        pages=pages,
        questions=questions,
        versions=question_versions,
        regions=regions,
        all_ids=all_ids,
        effective_by_question=effective_by_question,
    )
    warnings = _unique(
        [
            *decisions.warnings,
            *(warning for item in decisions.documents for warning in item.warnings),
            *(warning for item in decisions.pages for warning in item.warnings),
            *(warning for item in decisions.questions for warning in item.warnings),
            *(warning for item in decisions.question_versions for warning in item.warnings),
            *(warning for item in decisions.regions for warning in item.warnings),
        ]
    )
    return DocumentGraph(
        case_id=decisions.case_id,
        page_manifest_sha256=page_manifest_sha256,
        stage_fingerprint=stage_fingerprint,
        decision_sha256=decision_sha256,
        provenance=decisions.provenance,
        created_at=datetime.now(UTC),
        requires_review=bool(review_items),
        documents=documents,
        pages=pages,
        questions=questions,
        question_versions=question_versions,
        regions=regions,
        relationships=relationships,
        unresolved_relations=unresolved_relations,
        review_items=review_items,
        warnings=warnings,
    )


def _build_review_items(
    decisions: DocumentGraphDecisionSet,
    *,
    documents: list[DocumentNode],
    pages: list[PageNode],
    questions: list[QuestionNode],
    versions: list[QuestionVersionNode],
    regions: list[EvidenceRegion],
    all_ids: dict[str, str],
    effective_by_question: dict[str, str | None],
) -> list[ReviewItem]:
    review_items: list[ReviewItem] = []
    document_by_ref = {item.decision_ref: item for item in documents}
    page_by_id = {item.page_id: item for item in pages}
    question_by_ref = {item.decision_ref: item for item in questions}
    version_by_ref = {item.decision_ref: item for item in versions}
    region_by_ref = {item.decision_ref: item for item in regions}

    for decision in decisions.documents:
        if decision.requires_review:
            master = decision.role.value != "answer_sheet"
            review_items.append(
                ReviewItem(
                    kind=ReviewItemKind.DOCUMENT,
                    ref=document_by_ref[decision.ref].document_id,
                    reason="Document role requires review.",
                    blocks_master_ready=master,
                    blocks_submission_ready=True,
                )
            )
    document_role_by_id = {item.document_id: item.role for item in documents}
    for decision in decisions.pages:
        if decision.requires_review:
            role = document_role_by_id[page_by_id[decision.page_id].document_id]
            review_items.append(
                ReviewItem(
                    kind=ReviewItemKind.PAGE,
                    ref=decision.page_id,
                    reason="Page classification or semantic order requires review.",
                    blocks_master_ready=role.value != "answer_sheet",
                    blocks_submission_ready=True,
                )
            )
    for decision in decisions.questions:
        if decision.requires_review:
            review_items.append(
                ReviewItem(
                    kind=ReviewItemKind.QUESTION,
                    ref=question_by_ref[decision.ref].question_id,
                    reason="Question identity or hierarchy requires review.",
                    blocks_master_ready=True,
                    blocks_submission_ready=True,
                )
            )
    for decision in decisions.question_versions:
        if decision.requires_review:
            review_items.append(
                ReviewItem(
                    kind=ReviewItemKind.QUESTION_VERSION,
                    ref=version_by_ref[decision.ref].version_id,
                    reason="Question version evidence requires review.",
                    blocks_master_ready=True,
                    blocks_submission_ready=True,
                )
            )
    for decision in decisions.regions:
        if decision.requires_review:
            master = decision.kind != RegionKind.ANSWER_AREA
            review_items.append(
                ReviewItem(
                    kind=ReviewItemKind.REGION,
                    ref=region_by_ref[decision.ref].region_id,
                    reason="Region boundary or semantic role requires review.",
                    blocks_master_ready=master,
                    blocks_submission_ready=True,
                )
            )
    for relation in decisions.relations:
        if relation.status == RelationDecisionStatus.CANDIDATE:
            blocks_master = relation.type in {
                RelationType.SUPERSEDES,
                RelationType.CONTINUED_BY,
            }
            blocks_submission = relation.type != RelationType.SCRATCH_EVIDENCE_FOR
            relation_ref = _stable_graph_id(
                "relation",
                decisions.case_id,
                f"{relation.type.value}-{relation.from_ref}-{relation.to_ref}",
            )
            review_items.append(
                ReviewItem(
                    kind=ReviewItemKind.RELATION,
                    ref=relation_ref,
                    reason="Candidate semantic relationship requires review.",
                    blocks_master_ready=blocks_master,
                    blocks_submission_ready=blocks_submission,
                )
            )
    for question_ref, effective in effective_by_question.items():
        if effective is None:
            review_items.append(
                ReviewItem(
                    kind=ReviewItemKind.VERSION_RESOLUTION,
                    ref=question_by_ref[question_ref].question_id,
                    reason="Question has multiple possible effective versions.",
                    blocks_master_ready=True,
                    blocks_submission_ready=True,
                )
            )
    return _unique_review_items(review_items)


def _stable_graph_id(prefix: str, case_id: str, decision_ref: str) -> str:
    identity = digest_value(
        {"case_id": case_id, "kind": prefix, "decision_ref": decision_ref}
    )
    return f"{prefix}-{identity[:20]}"


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _unique_review_items(items: list[ReviewItem]) -> list[ReviewItem]:
    unique: dict[tuple[ReviewItemKind, str, str], ReviewItem] = {}
    for item in items:
        unique.setdefault((item.kind, item.ref, item.reason), item)
    return list(unique.values())
