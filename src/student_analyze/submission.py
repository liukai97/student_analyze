"""Phase 5 response mapping, crop preparation, and submission compilation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
from pathlib import Path
import re
from typing import Iterable

from PIL import Image, UnidentifiedImageError

from student_analyze import __version__
from student_analyze.assets import artifact_digest
from student_analyze.atomic import (
    InterruptHook,
    atomic_write_bytes,
    atomic_write_model,
    serialize_model,
)
from student_analyze.config import AppConfig
from student_analyze.document_mapper import read_document_graph
from student_analyze.document_models import (
    DecisionMethod,
    DocumentGraph,
    DocumentRole,
    RegionKind,
    RelationType,
)
from student_analyze.errors import (
    CaseValidationError,
    InvalidTransitionError,
    ReviewRequiredError,
)
from student_analyze.exam_master import read_exam_master, verify_exam_master
from student_analyze.exam_master_models import (
    AnnotationActor,
    EvidenceSourceKind,
    ExamMaster,
    QuestionType,
)
from student_analyze.fingerprint import digest_value
from student_analyze.models import (
    ImplementationVersions,
    PipelineStage,
    PipelineState,
    SCHEMA_VERSION,
)
from student_analyze.page_geometry import apply_matrix
from student_analyze.page_models import ImageSize, PageManifest, PixelBox
from student_analyze.page_verification import read_page_manifest
from student_analyze.pipeline import (
    build_stage_fingerprint,
    commit_stage_artifact,
    verify_case,
)
from student_analyze.submission_models import (
    AlternativeReading,
    KnownAnnotationRegion,
    Submission,
    SubmissionCropAsset,
    SubmissionInputItem,
    SubmissionInputManifest,
    SubmissionItem,
    SubmissionMappingDecision,
    SubmissionMappingDecisionSet,
    SubmissionNavigationPage,
    SubmissionQuestionPart,
    SubmissionQuestionTarget,
    SubmissionReviewItem,
    SubmissionSourceRegion,
    SubmissionSourceRole,
    SubmissionStructureManifest,
    SubmissionTranscriptionDecision,
    SubmissionTranscriptionDecisionSet,
    VisibleContentRole,
)
from student_analyze.validation import validate_json


_NAVIGATION_ROLES = {
    DocumentRole.QUESTION_BOOKLET,
    DocumentRole.ANSWER_SHEET,
    DocumentRole.SCRATCH,
}


@dataclass(frozen=True, slots=True)
class SubmissionStructurePreparationResult:
    case_dir: Path
    manifest_path: Path
    manifest: SubmissionStructureManifest
    reused: bool


@dataclass(frozen=True, slots=True)
class SubmissionInputPreparationResult:
    case_dir: Path
    manifest_path: Path
    manifest: SubmissionInputManifest
    reused: bool


@dataclass(frozen=True, slots=True)
class SubmissionBuildResult:
    case_dir: Path
    submission: Submission
    state: PipelineState
    reused: bool


def load_submission_structure_manifest(path: Path) -> SubmissionStructureManifest:
    return _read_model_file(path, SubmissionStructureManifest, "submission structure")


def load_submission_mapping_decisions(path: Path) -> SubmissionMappingDecisionSet:
    return _read_model_file(path, SubmissionMappingDecisionSet, "submission mappings")


def read_submission_input_manifest(path: Path) -> SubmissionInputManifest:
    return _read_model_file(path, SubmissionInputManifest, "submission input manifest")


def load_submission_transcription_decisions(
    path: Path,
) -> SubmissionTranscriptionDecisionSet:
    return _read_model_file(
        path, SubmissionTranscriptionDecisionSet, "submission transcriptions"
    )


def read_submission(path: Path) -> Submission:
    return _read_model_file(path, Submission, "submission artifact")


def prepare_submission_structure(
    case_dir: Path,
    config: AppConfig,
    *,
    interrupt_hook: InterruptHook | None = None,
) -> SubmissionStructurePreparationResult:
    """Create an answer-key-redacted manifest for the visual mapping pass."""

    case_dir = case_dir.resolve(strict=True)
    _, state = verify_case(case_dir)
    graph_path, graph = _active_document_graph(case_dir, state)
    page_path, page_manifest = _active_page_manifest(case_dir, state)
    master_path, master = _active_exam_master(case_dir, state)
    graph_sha256, _ = artifact_digest(graph_path)
    page_sha256, _ = artifact_digest(page_path)
    master_sha256, _ = artifact_digest(master_path)
    _reject_submission_blockers(graph)
    verify_exam_master(case_dir, master, graph, graph_sha256=graph_sha256)

    fingerprint = digest_value(
        {
            "document_graph_sha256": graph_sha256,
            "page_manifest_sha256": page_sha256,
            "exam_master_sha256": master_sha256,
            "config": config.submission_context_fingerprint_payload(),
        }
    )
    output_dir = case_dir / "work" / "submission_context" / fingerprint
    manifest_path = output_dir / "submission_structure_manifest.json"

    questions = [
        SubmissionQuestionTarget(
            question_id=question.question_id,
            version_id=question.version_id,
            printed_label=question.printed_label,
            question_type=question.question_type,
            option_labels=[option.label for option in question.options],
            parts=[
                SubmissionQuestionPart(
                    part_id=part.part_id,
                    printed_label=part.printed_label,
                    order=order,
                )
                for order, part in enumerate(question.subparts, start=1)
            ],
        )
        for question in master.questions
    ]

    document_roles = {item.document_id: item.role for item in graph.documents}
    page_roles = {
        item.page_id: document_roles[item.document_id] for item in graph.pages
    }
    pages = [
        SubmissionNavigationPage(
            page_id=page.page_id,
            document_role=page_roles[page.page_id],
            relative_path=page.derived.relative_path,
            sha256=page.derived.sha256,
            size_bytes=page.derived.size_bytes,
            image_size=page.page_image_size,
        )
        for page in page_manifest.pages
        if page_roles.get(page.page_id) in _NAVIGATION_ROLES
    ]
    selected_page_ids = {page.page_id for page in pages}

    version_to_question = {
        version.version_id: version.question_id for version in graph.question_versions
    }
    targets_by_region: dict[str, list[tuple[str, str]]] = {}
    for relation in graph.relationships:
        if relation.type not in {
            RelationType.ANSWERS,
            RelationType.SCRATCH_EVIDENCE_FOR,
        }:
            continue
        question_id = version_to_question.get(relation.to_id)
        if question_id is not None:
            targets_by_region.setdefault(relation.from_id, []).append(
                (question_id, relation.to_id)
            )

    source_regions: list[SubmissionSourceRegion] = []
    question_order = {
        question.question_id: index for index, question in enumerate(master.questions)
    }
    for region in graph.regions:
        if region.page_id not in selected_page_ids or region.kind not in {
            RegionKind.ANSWER_AREA,
            RegionKind.SCRATCH,
        }:
            continue
        targets = sorted(
            set(targets_by_region.get(region.region_id, [])),
            key=lambda item: (question_order.get(item[0], len(question_order)), item),
        )
        source_regions.append(
            SubmissionSourceRegion(
                region_id=region.region_id,
                page_id=region.page_id,
                kind=region.kind,
                bbox=region.bbox,
                question_ids=[item[0] for item in targets],
                version_ids=[item[1] for item in targets],
            )
        )
    if not any(region.kind == RegionKind.ANSWER_AREA for region in source_regions):
        raise ReviewRequiredError(
            "submission mapping requires at least one accepted answer-area region"
        )

    known_annotations = [
        KnownAnnotationRegion(
            evidence_ref=item.ref,
            page_id=item.page_id,
            bbox=item.bbox,
            actor=item.actor,
            status=item.status,
            human_confirmed=item.human_confirmed,
            requires_review=item.requires_review,
        )
        for item in master.answer_evidence
        if item.source_kind == EvidenceSourceKind.TEACHER_ANNOTATION
        and item.page_id in selected_page_ids
    ]
    warnings = _unique_strings(
        [
            "Reference answers, rubrics, solution summaries, and answer-source content "
            "are intentionally omitted from this transcription context.",
            *(
                [
                    "No mapped scratch regions are available; answer-sheet transcription "
                    "can proceed, but question-booklet scratch evidence must be proposed "
                    "with explicit page bboxes and review when needed."
                ]
                if not any(region.kind == RegionKind.SCRATCH for region in source_regions)
                else []
            ),
        ]
    )
    manifest = SubmissionStructureManifest(
        case_id=graph.case_id,
        document_graph_sha256=graph_sha256,
        page_manifest_sha256=page_sha256,
        exam_master_sha256=master_sha256,
        manifest_fingerprint=fingerprint,
        created_at=datetime.now(UTC),
        questions=questions,
        pages=pages,
        source_regions=source_regions,
        known_annotations=known_annotations,
        warnings=warnings,
    )

    if manifest_path.exists():
        existing = load_submission_structure_manifest(manifest_path)
        verify_submission_structure_manifest(
            case_dir,
            existing,
            graph,
            page_manifest,
            master,
            graph_sha256=graph_sha256,
            page_manifest_sha256=page_sha256,
            exam_master_sha256=master_sha256,
        )
        if _structure_signature(existing) != _structure_signature(manifest):
            raise CaseValidationError(
                f"conflicting submission structure manifest at {manifest_path}"
            )
        return SubmissionStructurePreparationResult(
            case_dir, manifest_path, existing, reused=True
        )

    atomic_write_model(
        manifest_path,
        manifest,
        model_type=SubmissionStructureManifest,
        interrupt_hook=interrupt_hook,
    )
    verify_submission_structure_manifest(
        case_dir,
        manifest,
        graph,
        page_manifest,
        master,
        graph_sha256=graph_sha256,
        page_manifest_sha256=page_sha256,
        exam_master_sha256=master_sha256,
    )
    return SubmissionStructurePreparationResult(
        case_dir, manifest_path, manifest, reused=False
    )


def prepare_submission_inputs(
    case_dir: Path,
    structure: SubmissionStructureManifest,
    mappings: SubmissionMappingDecisionSet,
    config: AppConfig,
    *,
    interrupt_hook: InterruptHook | None = None,
) -> SubmissionInputPreparationResult:
    """Validate response mappings and materialize high-detail evidence crops."""

    case_dir = case_dir.resolve(strict=True)
    _, state = verify_case(case_dir)
    graph_path, graph = _active_document_graph(case_dir, state)
    page_path, page_manifest = _active_page_manifest(case_dir, state)
    master_path, master = _active_exam_master(case_dir, state)
    graph_sha256, _ = artifact_digest(graph_path)
    page_sha256, _ = artifact_digest(page_path)
    master_sha256, _ = artifact_digest(master_path)
    verify_submission_structure_manifest(
        case_dir,
        structure,
        graph,
        page_manifest,
        master,
        graph_sha256=graph_sha256,
        page_manifest_sha256=page_sha256,
        exam_master_sha256=master_sha256,
    )
    structure_sha256 = digest_value(structure.model_dump(mode="json"))
    _validate_mapping_decisions(mappings, structure, structure_sha256=structure_sha256)
    mapping_sha256 = digest_value(mappings.model_dump(mode="json"))
    fingerprint = digest_value(
        {
            "structure_manifest_sha256": structure_sha256,
            "mapping_decision_sha256": mapping_sha256,
            "config": config.submission_input_fingerprint_payload(),
        }
    )
    output_dir = case_dir / "work" / "submission_inputs" / fingerprint
    manifest_path = output_dir / "submission_input_manifest.json"
    if manifest_path.exists():
        existing = read_submission_input_manifest(manifest_path)
        verify_submission_input_manifest(
            case_dir,
            existing,
            graph,
            page_manifest,
            master,
            graph_sha256=graph_sha256,
            page_manifest_sha256=page_sha256,
            exam_master_sha256=master_sha256,
        )
        return SubmissionInputPreparationResult(
            case_dir, manifest_path, existing, reused=True
        )

    page_by_id = {page.page_id: page for page in page_manifest.pages}
    items: list[SubmissionInputItem] = []
    for mapping in _sorted_mappings(mappings.items, structure):
        source_page = page_by_id[mapping.page_id]
        relative_path = (
            Path("work")
            / "submission_inputs"
            / fingerprint
            / "crops"
            / f"{mapping.ref}.jpg"
        )
        crop_path = case_dir / relative_path
        content, image_size = _render_crop(
            case_dir / source_page.derived.relative_path,
            mapping.bbox,
            jpeg_quality=config.jpeg_quality,
        )
        expected_sha256 = sha256(content).hexdigest()
        if crop_path.exists():
            existing_sha256, existing_size = artifact_digest(crop_path)
            if existing_sha256 != expected_sha256 or existing_size != len(content):
                raise CaseValidationError(f"conflicting submission crop at {crop_path}")
        else:
            atomic_write_bytes(
                crop_path,
                content,
                validator=_validate_jpeg,
                interrupt_hook=interrupt_hook,
            )
        crop_sha256, crop_size = artifact_digest(crop_path)
        crop = SubmissionCropAsset(
            asset_id=_stable_id("submission-crop", graph.case_id, mapping.ref),
            mapping_ref=mapping.ref,
            page_id=mapping.page_id,
            page_bbox=mapping.bbox,
            raw_bbox=_page_bbox_to_raw(source_page, mapping.bbox),
            source_asset_id=source_page.source_asset_id,
            source_relative_path=source_page.source_relative_path,
            source_sha256=source_page.source_sha256,
            relative_path=relative_path.as_posix(),
            sha256=crop_sha256,
            size_bytes=crop_size,
            image_size=image_size,
        )
        items.append(SubmissionInputItem(mapping=mapping, crop=crop))

    manifest = SubmissionInputManifest(
        case_id=graph.case_id,
        structure_manifest_sha256=structure_sha256,
        mapping_decision_sha256=mapping_sha256,
        manifest_fingerprint=fingerprint,
        created_at=datetime.now(UTC),
        structure_manifest=structure,
        mapping_decisions=mappings,
        items=items,
    )
    atomic_write_model(
        manifest_path,
        manifest,
        model_type=SubmissionInputManifest,
        interrupt_hook=interrupt_hook,
    )
    verify_submission_input_manifest(
        case_dir,
        manifest,
        graph,
        page_manifest,
        master,
        graph_sha256=graph_sha256,
        page_manifest_sha256=page_sha256,
        exam_master_sha256=master_sha256,
    )
    return SubmissionInputPreparationResult(
        case_dir, manifest_path, manifest, reused=False
    )


def build_submission(
    case_dir: Path,
    input_manifest: SubmissionInputManifest,
    transcriptions: SubmissionTranscriptionDecisionSet,
    config: AppConfig,
    *,
    force: bool = False,
    interrupt_hook: InterruptHook | None = None,
) -> SubmissionBuildResult:
    """Compile reviewed model decisions into the immutable Submission artifact."""

    case_dir = case_dir.resolve(strict=True)
    _, state = verify_case(case_dir)
    graph_path, graph = _active_document_graph(case_dir, state)
    page_path, page_manifest = _active_page_manifest(case_dir, state)
    master_path, master = _active_exam_master(case_dir, state)
    graph_sha256, _ = artifact_digest(graph_path)
    page_sha256, _ = artifact_digest(page_path)
    master_sha256, _ = artifact_digest(master_path)
    verify_submission_input_manifest(
        case_dir,
        input_manifest,
        graph,
        page_manifest,
        master,
        graph_sha256=graph_sha256,
        page_manifest_sha256=page_sha256,
        exam_master_sha256=master_sha256,
    )
    input_manifest_sha256 = sha256(serialize_model(input_manifest)).hexdigest()
    _validate_transcription_decisions(
        transcriptions,
        input_manifest,
        input_manifest_sha256=input_manifest_sha256,
    )
    transcription_sha256 = digest_value(transcriptions.model_dump(mode="json"))

    versions = ImplementationVersions(
        code=__version__,
        base_schema=SCHEMA_VERSION,
        config=config.config_version,
        model=transcriptions.provenance.model_identifier,
        prompt=transcriptions.provenance.prompt_version,
        skill=transcriptions.provenance.skill_version,
        tools={"pillow": version("Pillow")},
    )
    stage_inputs = [
        {
            "document_graph_sha256": graph_sha256,
            "page_manifest_sha256": page_sha256,
            "exam_master_sha256": master_sha256,
            "structure_manifest_sha256": input_manifest.structure_manifest_sha256,
            "mapping_decision_sha256": input_manifest.mapping_decision_sha256,
            "submission_input_manifest_sha256": input_manifest_sha256,
            "transcription_decision_sha256": transcription_sha256,
        }
    ]
    stage_config = config.submission_fingerprint_payload()
    stage_fingerprint, _ = build_stage_fingerprint(
        stage=PipelineStage.SUBMISSION_READY,
        model_type=Submission,
        schema_id="submission.schema.json",
        config=stage_config,
        versions=versions,
        inputs=stage_inputs,
    )

    existing_completion = next(
        (
            item
            for item in state.completed_stages
            if item.stage == PipelineStage.SUBMISSION_READY
        ),
        None,
    )
    if existing_completion is not None:
        if existing_completion.stage_fingerprint == stage_fingerprint and not force:
            reference = next(
                item
                for item in existing_completion.artifacts
                if item.schema_id == "submission.schema.json"
            )
            submission = read_submission(case_dir / reference.relative_path)
            verify_submission(
                case_dir,
                submission,
                graph,
                page_manifest,
                master,
                graph_sha256=graph_sha256,
                page_manifest_sha256=page_sha256,
                exam_master_sha256=master_sha256,
            )
            return SubmissionBuildResult(case_dir, submission, state, reused=True)
        if not force:
            raise InvalidTransitionError(
                "submission_ready is already complete with a different fingerprint; "
                "use force to create a preserved version"
            )
    elif state.current_stage != PipelineStage.MASTER_READY:
        raise InvalidTransitionError(
            f"cannot build a Submission from current stage {state.current_stage}"
        )

    submission = _compile_submission(
        input_manifest,
        transcriptions,
        graph_sha256=graph_sha256,
        page_manifest_sha256=page_sha256,
        exam_master_sha256=master_sha256,
        input_manifest_sha256=input_manifest_sha256,
        transcription_sha256=transcription_sha256,
        stage_fingerprint=stage_fingerprint,
    )
    verify_submission(
        case_dir,
        submission,
        graph,
        page_manifest,
        master,
        graph_sha256=graph_sha256,
        page_manifest_sha256=page_sha256,
        exam_master_sha256=master_sha256,
    )
    orphan_path = (
        case_dir
        / "artifacts"
        / PipelineStage.SUBMISSION_READY.value
        / stage_fingerprint
        / "submission.json"
    )
    if orphan_path.exists() and not force:
        orphan = read_submission(orphan_path)
        if (
            orphan.case_id != submission.case_id
            or orphan.stage_fingerprint != submission.stage_fingerprint
            or orphan.transcription_decision_sha256
            != submission.transcription_decision_sha256
        ):
            raise CaseValidationError(
                f"conflicting uncommitted Submission at {orphan_path}"
            )
        submission = orphan

    committed = commit_stage_artifact(
        case_dir,
        stage=PipelineStage.SUBMISSION_READY,
        artifact_name="submission.json",
        payload=submission,
        model_type=Submission,
        schema_id="submission.schema.json",
        config=stage_config,
        versions=versions,
        inputs=stage_inputs,
        force=force,
        human_confirmed=(
            input_manifest.mapping_decisions.provenance.method
            == DecisionMethod.HUMAN_REVIEW
            or transcriptions.provenance.method == DecisionMethod.HUMAN_REVIEW
        ),
        interrupt_hook=interrupt_hook,
    )
    reference = committed.artifacts[0]
    active = read_submission(case_dir / reference.relative_path)
    return SubmissionBuildResult(
        case_dir, active, committed.state, reused=committed.reused
    )


def verify_submission_structure_manifest(
    case_dir: Path,
    manifest: SubmissionStructureManifest,
    graph: DocumentGraph,
    page_manifest: PageManifest,
    master: ExamMaster,
    *,
    graph_sha256: str,
    page_manifest_sha256: str,
    exam_master_sha256: str,
) -> None:
    if manifest.case_id != graph.case_id or manifest.case_id != master.case_id:
        raise CaseValidationError("submission structure belongs to another case")
    if manifest.document_graph_sha256 != graph_sha256:
        raise CaseValidationError("submission structure references another document graph")
    if manifest.page_manifest_sha256 != page_manifest_sha256:
        raise CaseValidationError("submission structure references another page manifest")
    if manifest.exam_master_sha256 != exam_master_sha256:
        raise CaseValidationError("submission structure references another Exam Master")

    master_questions = {item.question_id: item for item in master.questions}
    structure_questions = {item.question_id: item for item in manifest.questions}
    if set(master_questions) != set(structure_questions):
        raise CaseValidationError("submission structure must cover every master question")
    for question_id, target in structure_questions.items():
        master_question = master_questions[question_id]
        if target.version_id != master_question.version_id:
            raise CaseValidationError(
                f"submission target {question_id} uses a non-effective version"
            )
        if target.question_type != master_question.question_type:
            raise CaseValidationError(
                f"submission target {question_id} changes question type"
            )
        if target.option_labels != [item.label for item in master_question.options]:
            raise CaseValidationError(
                f"submission target {question_id} changes option labels"
            )
        expected_parts = [
            (part.part_id, part.printed_label, order)
            for order, part in enumerate(master_question.subparts, start=1)
        ]
        actual_parts = [
            (part.part_id, part.printed_label, part.order) for part in target.parts
        ]
        if actual_parts != expected_parts:
            raise CaseValidationError(
                f"submission target {question_id} changes master subparts"
            )

    page_by_id = {page.page_id: page for page in page_manifest.pages}
    document_roles = {item.document_id: item.role for item in graph.documents}
    expected_page_roles = {
        item.page_id: document_roles[item.document_id]
        for item in graph.pages
        if document_roles[item.document_id] in _NAVIGATION_ROLES
    }
    actual_page_ids = {item.page_id for item in manifest.pages}
    if actual_page_ids != set(expected_page_roles):
        raise CaseValidationError(
            "submission structure navigation pages differ from the document graph"
        )
    for navigation in manifest.pages:
        page = page_by_id.get(navigation.page_id)
        if page is None:
            raise CaseValidationError(
                f"submission navigation page is unknown: {navigation.page_id}"
            )
        if navigation.document_role != expected_page_roles[navigation.page_id]:
            raise CaseValidationError(
                f"submission navigation role changed: {navigation.page_id}"
            )
        if (
            navigation.relative_path != page.derived.relative_path
            or navigation.sha256 != page.derived.sha256
            or navigation.size_bytes != page.derived.size_bytes
            or navigation.image_size != page.page_image_size
        ):
            raise CaseValidationError(
                f"submission navigation page changed: {navigation.page_id}"
            )
        path = case_dir / navigation.relative_path
        digest, size = artifact_digest(path)
        if digest != navigation.sha256 or size != navigation.size_bytes:
            raise CaseValidationError(f"submission navigation asset changed: {path}")

    graph_regions = {region.region_id: region for region in graph.regions}
    version_to_question = {
        version.version_id: version.question_id for version in graph.question_versions
    }
    graph_targets: dict[str, set[tuple[str, str]]] = {}
    for relation in graph.relationships:
        if relation.type not in {
            RelationType.ANSWERS,
            RelationType.SCRATCH_EVIDENCE_FOR,
        }:
            continue
        question_id = version_to_question.get(relation.to_id)
        if question_id is not None:
            graph_targets.setdefault(relation.from_id, set()).add(
                (question_id, relation.to_id)
            )
    expected_region_ids = {
        region.region_id
        for region in graph.regions
        if region.page_id in actual_page_ids
        and region.kind in {RegionKind.ANSWER_AREA, RegionKind.SCRATCH}
    }
    actual_region_ids = {item.region_id for item in manifest.source_regions}
    if actual_region_ids != expected_region_ids:
        raise CaseValidationError(
            "submission source regions differ from the document graph"
        )
    for source in manifest.source_regions:
        region = graph_regions.get(source.region_id)
        if (
            region is None
            or source.page_id != region.page_id
            or source.kind != region.kind
            or source.bbox != region.bbox
        ):
            raise CaseValidationError(
                f"submission source region changed: {source.region_id}"
            )
        if set(zip(source.question_ids, source.version_ids, strict=True)) != graph_targets.get(
            source.region_id, set()
        ):
            raise CaseValidationError(
                f"submission source targets changed: {source.region_id}"
            )

    evidence = {
        item.ref: item
        for item in master.answer_evidence
        if item.source_kind == EvidenceSourceKind.TEACHER_ANNOTATION
        and item.page_id in actual_page_ids
    }
    if {item.evidence_ref for item in manifest.known_annotations} != set(evidence):
        raise CaseValidationError(
            "submission annotation exclusions differ from the Exam Master"
        )
    for annotation in manifest.known_annotations:
        item = evidence.get(annotation.evidence_ref)
        if (
            item is None
            or annotation.page_id != item.page_id
            or annotation.bbox != item.bbox
            or annotation.actor != item.actor
            or annotation.status != item.status
            or annotation.human_confirmed != item.human_confirmed
            or annotation.requires_review != item.requires_review
        ):
            raise CaseValidationError(
                f"known annotation changed: {annotation.evidence_ref}"
            )


def verify_submission_input_manifest(
    case_dir: Path,
    manifest: SubmissionInputManifest,
    graph: DocumentGraph,
    page_manifest: PageManifest,
    master: ExamMaster,
    *,
    graph_sha256: str,
    page_manifest_sha256: str,
    exam_master_sha256: str,
) -> None:
    structure_sha256 = digest_value(
        manifest.structure_manifest.model_dump(mode="json")
    )
    mapping_sha256 = digest_value(manifest.mapping_decisions.model_dump(mode="json"))
    if manifest.structure_manifest_sha256 != structure_sha256:
        raise CaseValidationError("embedded submission structure hash mismatch")
    if manifest.mapping_decision_sha256 != mapping_sha256:
        raise CaseValidationError("embedded response mapping hash mismatch")
    verify_submission_structure_manifest(
        case_dir,
        manifest.structure_manifest,
        graph,
        page_manifest,
        master,
        graph_sha256=graph_sha256,
        page_manifest_sha256=page_manifest_sha256,
        exam_master_sha256=exam_master_sha256,
    )
    _validate_mapping_decisions(
        manifest.mapping_decisions,
        manifest.structure_manifest,
        structure_sha256=structure_sha256,
    )
    page_by_id = {page.page_id: page for page in page_manifest.pages}
    for item in manifest.items:
        page = page_by_id[item.crop.page_id]
        if item.crop.raw_bbox != _page_bbox_to_raw(page, item.crop.page_bbox):
            raise CaseValidationError(
                f"submission crop raw bbox changed: {item.crop.asset_id}"
            )
        if (
            item.crop.source_asset_id != page.source_asset_id
            or item.crop.source_relative_path != page.source_relative_path
            or item.crop.source_sha256 != page.source_sha256
        ):
            raise CaseValidationError(
                f"submission crop source changed: {item.crop.asset_id}"
            )
        path = case_dir / item.crop.relative_path
        digest, size = artifact_digest(path)
        if digest != item.crop.sha256 or size != item.crop.size_bytes:
            raise CaseValidationError(f"submission crop changed: {path}")
        try:
            with Image.open(path) as image:
                if image.size != (
                    item.crop.image_size.width,
                    item.crop.image_size.height,
                ):
                    raise CaseValidationError(
                        f"submission crop dimensions changed: {path}"
                    )
        except (OSError, UnidentifiedImageError) as exc:
            raise CaseValidationError(f"cannot decode submission crop {path}: {exc}") from exc


def verify_submission(
    case_dir: Path,
    submission: Submission,
    graph: DocumentGraph,
    page_manifest: PageManifest,
    master: ExamMaster,
    *,
    graph_sha256: str,
    page_manifest_sha256: str,
    exam_master_sha256: str,
) -> None:
    if submission.case_id != graph.case_id or submission.case_id != master.case_id:
        raise CaseValidationError("Submission belongs to another case")
    if submission.document_graph_sha256 != graph_sha256:
        raise CaseValidationError("Submission references another document graph")
    if submission.page_manifest_sha256 != page_manifest_sha256:
        raise CaseValidationError("Submission references another page manifest")
    if submission.exam_master_sha256 != exam_master_sha256:
        raise CaseValidationError("Submission references another Exam Master")

    questions = {question.question_id: question for question in master.questions}
    page_by_id = {page.page_id: page for page in page_manifest.pages}
    region_by_id = {region.region_id: region for region in graph.regions}
    annotation_refs = {
        item.ref
        for item in master.answer_evidence
        if item.source_kind == EvidenceSourceKind.TEACHER_ANNOTATION
    }
    for item in submission.items:
        question = questions.get(item.question_id)
        if question is None or item.version_id != question.version_id:
            raise CaseValidationError(
                f"submission item targets an unknown question version: {item.item_id}"
            )
        part_ids = {part.part_id for part in question.subparts}
        if item.part_id is not None and item.part_id not in part_ids:
            raise CaseValidationError(
                f"submission item targets an unknown part: {item.item_id}"
            )
        if question.subparts and item.part_id is None:
            raise CaseValidationError(
                f"submission item omits a required part ID: {item.item_id}"
            )
        if not question.subparts and item.part_id is not None:
            raise CaseValidationError(
                f"submission item adds an unexpected part ID: {item.item_id}"
            )
        if any(ref not in annotation_refs for ref in item.excluded_annotation_refs):
            raise CaseValidationError(
                f"submission item excludes an unknown annotation: {item.item_id}"
            )
        if item.source_region_id is not None and item.source_region_id not in region_by_id:
            raise CaseValidationError(
                f"submission item references an unknown region: {item.item_id}"
            )
        page = page_by_id[item.crop.page_id]
        if item.crop.raw_bbox != _page_bbox_to_raw(page, item.crop.page_bbox):
            raise CaseValidationError(
                f"submission item raw bbox changed: {item.item_id}"
            )
        path = case_dir / item.crop.relative_path
        digest, size = artifact_digest(path)
        if digest != item.crop.sha256 or size != item.crop.size_bytes:
            raise CaseValidationError(f"submission evidence crop changed: {path}")
        if item.confidence != min(
            item.mapping_confidence, item.transcription_confidence
        ):
            raise CaseValidationError(
                f"submission item confidence is not conservative: {item.item_id}"
            )
        _validate_transcribed_values(
            question.question_type,
            [option.label for option in question.options],
            observed_content=item.observed_content,
            normalized_answer=item.normalized_answer,
            alternatives=item.alternatives,
            is_blank=item.is_blank,
            has_erasure=item.has_erasure,
            confidence=item.transcription_confidence,
            uncertainty_notes=item.uncertainty_notes,
            requires_review=item.requires_review,
            label=f"submission item {item.item_id}",
        )
    _validate_formal_coverage(submission.items, master)


def _validate_mapping_decisions(
    mappings: SubmissionMappingDecisionSet,
    structure: SubmissionStructureManifest,
    *,
    structure_sha256: str,
) -> None:
    if mappings.case_id != structure.case_id:
        raise CaseValidationError("response mappings belong to another case")
    if mappings.structure_manifest_sha256 != structure_sha256:
        raise CaseValidationError("response mappings reference another structure manifest")
    questions = {item.question_id: item for item in structure.questions}
    pages = {item.page_id: item for item in structure.pages}
    regions = {item.region_id: item for item in structure.source_regions}
    annotations = {item.evidence_ref: item for item in structure.known_annotations}
    formal_order_keys: list[tuple[str, str | None, int]] = []
    for mapping in mappings.items:
        question = questions.get(mapping.question_id)
        if question is None or mapping.version_id != question.version_id:
            raise CaseValidationError(
                f"mapping {mapping.ref} targets an unknown question version"
            )
        part_ids = {part.part_id for part in question.parts}
        if question.parts and mapping.part_id is None:
            raise CaseValidationError(f"mapping {mapping.ref} requires a part ID")
        if not question.parts and mapping.part_id is not None:
            raise CaseValidationError(f"mapping {mapping.ref} has an unexpected part ID")
        if mapping.part_id is not None and mapping.part_id not in part_ids:
            raise CaseValidationError(f"mapping {mapping.ref} targets an unknown part")
        page = pages.get(mapping.page_id)
        if page is None:
            raise CaseValidationError(f"mapping {mapping.ref} uses an unknown page")
        if not _box_within_image(mapping.bbox, page.image_size.width, page.image_size.height):
            raise CaseValidationError(f"mapping {mapping.ref} bbox exceeds its page")

        region = (
            regions.get(mapping.source_region_id)
            if mapping.source_region_id is not None
            else None
        )
        if mapping.source_role == SubmissionSourceRole.ANSWER_SHEET:
            if page.document_role != DocumentRole.ANSWER_SHEET:
                raise CaseValidationError(
                    f"formal mapping {mapping.ref} is not on an answer sheet"
                )
            if (
                region is None
                or region.kind != RegionKind.ANSWER_AREA
                or region.page_id != mapping.page_id
                or not _box_contains(region.bbox, mapping.bbox)
            ):
                raise CaseValidationError(
                    f"formal mapping {mapping.ref} must stay inside its answer area"
                )
            if (mapping.question_id, mapping.version_id) not in set(
                zip(region.question_ids, region.version_ids, strict=True)
            ):
                raise CaseValidationError(
                    f"formal mapping {mapping.ref} is not supported by an answers relation"
                )
            formal_order_keys.append(
                (mapping.question_id, mapping.part_id, mapping.slot_order)
            )
        elif mapping.source_role == SubmissionSourceRole.QUESTION_BOOKLET_SCRATCH:
            if page.document_role != DocumentRole.QUESTION_BOOKLET:
                raise CaseValidationError(
                    f"question-booklet scratch {mapping.ref} uses another document role"
                )
        elif mapping.source_role == SubmissionSourceRole.SCRATCH_SHEET:
            if page.document_role != DocumentRole.SCRATCH:
                raise CaseValidationError(
                    f"scratch-sheet mapping {mapping.ref} uses another document role"
                )

        if mapping.source_role in {
            SubmissionSourceRole.QUESTION_BOOKLET_SCRATCH,
            SubmissionSourceRole.SCRATCH_SHEET,
        } and region is not None:
            if region.kind != RegionKind.SCRATCH or (
                mapping.question_id,
                mapping.version_id,
            ) not in set(zip(region.question_ids, region.version_ids, strict=True)):
                raise CaseValidationError(
                    f"scratch mapping {mapping.ref} lacks scratch_evidence_for support"
                )

        if region is not None:
            if region.page_id != mapping.page_id or not _box_contains(
                region.bbox, mapping.bbox
            ):
                raise CaseValidationError(
                    f"mapping {mapping.ref} falls outside its source region"
                )
        elif mapping.source_region_id is not None:
            raise CaseValidationError(
                f"mapping {mapping.ref} references an unknown source region"
            )

        overlapping_annotations = {
            item.evidence_ref
            for item in structure.known_annotations
            if item.page_id == mapping.page_id
            and _boxes_intersect(item.bbox, mapping.bbox)
        }
        excluded = set(mapping.excluded_annotation_refs)
        if not overlapping_annotations.issubset(excluded):
            missing = sorted(overlapping_annotations - excluded)
            raise CaseValidationError(
                f"mapping {mapping.ref} does not exclude known annotations: {missing}"
            )
        for ref in excluded:
            annotation = annotations.get(ref)
            if (
                annotation is None
                or annotation.page_id != mapping.page_id
                or not _boxes_intersect(annotation.bbox, mapping.bbox)
            ):
                raise CaseValidationError(
                    f"mapping {mapping.ref} excludes an unrelated annotation {ref}"
                )
    if len(formal_order_keys) != len(set(formal_order_keys)):
        raise CaseValidationError("formal slot order must be unique per response target")
    _validate_formal_mapping_coverage(mappings.items, structure)


def _validate_transcription_decisions(
    transcriptions: SubmissionTranscriptionDecisionSet,
    manifest: SubmissionInputManifest,
    *,
    input_manifest_sha256: str,
) -> None:
    if transcriptions.case_id != manifest.case_id:
        raise CaseValidationError("transcriptions belong to another case")
    if transcriptions.submission_input_manifest_sha256 != input_manifest_sha256:
        raise CaseValidationError("transcriptions reference another input manifest")
    expected = {item.mapping.ref for item in manifest.items}
    actual = {item.mapping_ref for item in transcriptions.decisions}
    if actual != expected:
        raise CaseValidationError(
            "transcriptions must cover every response mapping exactly once; "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )
    questions = {
        item.question_id: item for item in manifest.structure_manifest.questions
    }
    mappings = {item.mapping.ref: item.mapping for item in manifest.items}
    for decision in transcriptions.decisions:
        mapping = mappings[decision.mapping_ref]
        question = questions[mapping.question_id]
        _validate_transcribed_values(
            question.question_type,
            question.option_labels,
            observed_content=decision.observed_content,
            normalized_answer=decision.normalized_answer,
            alternatives=decision.alternatives,
            is_blank=decision.is_blank,
            has_erasure=decision.has_erasure,
            confidence=decision.confidence,
            uncertainty_notes=decision.uncertainty_notes,
            requires_review=decision.requires_review,
            label=f"transcription {decision.mapping_ref}",
        )


def _compile_submission(
    manifest: SubmissionInputManifest,
    transcriptions: SubmissionTranscriptionDecisionSet,
    *,
    graph_sha256: str,
    page_manifest_sha256: str,
    exam_master_sha256: str,
    input_manifest_sha256: str,
    transcription_sha256: str,
    stage_fingerprint: str,
) -> Submission:
    decisions = {item.mapping_ref: item for item in transcriptions.decisions}
    annotations = {
        item.evidence_ref: item
        for item in manifest.structure_manifest.known_annotations
    }
    items: list[SubmissionItem] = []
    review_items: list[SubmissionReviewItem] = []
    for input_item in manifest.items:
        mapping = input_item.mapping
        decision = decisions[mapping.ref]
        reasons: list[str] = []
        if mapping.requires_review:
            reasons.append("Response mapping requires review.")
        if decision.requires_review:
            reasons.append("Faithful transcription requires review.")
        if mapping.source_role == SubmissionSourceRole.UNCERTAIN:
            reasons.append("Response source role is uncertain.")
        if VisibleContentRole.UNCERTAIN in mapping.visible_roles:
            reasons.append("The crop contains visually uncertain content roles.")
        if (
            VisibleContentRole.TEACHER_ANNOTATION in mapping.visible_roles
            and not mapping.excluded_annotation_refs
        ):
            reasons.append("A newly suspected teacher annotation is not yet classified.")
        for ref in mapping.excluded_annotation_refs:
            annotation = annotations[ref]
            if (
                annotation.requires_review
                or annotation.actor != AnnotationActor.TEACHER
                or not annotation.human_confirmed
            ):
                reasons.append(
                    f"Excluded annotation {ref} still has unresolved actor or scope."
                )
        reasons = _unique_strings(reasons)
        item_id = _stable_id("submission-item", manifest.case_id, mapping.ref)
        requires_review = bool(reasons)
        item = SubmissionItem(
            item_id=item_id,
            mapping_ref=mapping.ref,
            question_id=mapping.question_id,
            version_id=mapping.version_id,
            part_id=mapping.part_id,
            slot_label=mapping.slot_label,
            slot_order=mapping.slot_order,
            source_role=mapping.source_role,
            source_region_id=mapping.source_region_id,
            visible_roles=mapping.visible_roles,
            excluded_annotation_refs=mapping.excluded_annotation_refs,
            observed_content=decision.observed_content,
            normalized_answer=decision.normalized_answer,
            alternatives=decision.alternatives,
            is_blank=decision.is_blank,
            has_erasure=decision.has_erasure,
            mapping_confidence=mapping.confidence,
            transcription_confidence=decision.confidence,
            confidence=min(mapping.confidence, decision.confidence),
            mapping_evidence=mapping.evidence,
            transcription_evidence=decision.evidence,
            uncertainty_notes=decision.uncertainty_notes,
            crop=input_item.crop,
            requires_review=requires_review,
            warnings=_unique_strings([*mapping.warnings, *decision.warnings]),
        )
        items.append(item)
        if requires_review:
            review_items.append(SubmissionReviewItem(item_id=item_id, reasons=reasons))

    return Submission(
        case_id=manifest.case_id,
        document_graph_sha256=graph_sha256,
        page_manifest_sha256=page_manifest_sha256,
        exam_master_sha256=exam_master_sha256,
        structure_manifest_sha256=manifest.structure_manifest_sha256,
        mapping_decision_sha256=manifest.mapping_decision_sha256,
        submission_input_manifest_sha256=input_manifest_sha256,
        transcription_decision_sha256=transcription_sha256,
        stage_fingerprint=stage_fingerprint,
        created_at=datetime.now(UTC),
        mapping_provenance=manifest.mapping_decisions.provenance,
        transcription_provenance=transcriptions.provenance,
        items=items,
        review_items=review_items,
        requires_review=bool(review_items),
        warnings=_unique_strings(
            [
                *manifest.structure_manifest.warnings,
                *manifest.mapping_decisions.warnings,
                *transcriptions.warnings,
            ]
        ),
    )


def _validate_transcribed_values(
    question_type: QuestionType,
    option_labels: list[str],
    *,
    observed_content: str | None,
    normalized_answer: str | None,
    alternatives: list[AlternativeReading],
    is_blank: bool,
    has_erasure: bool,
    confidence: float,
    uncertainty_notes: list[str],
    requires_review: bool,
    label: str,
) -> None:
    # Reuse the Pydantic invariants for final artifacts as well as decision inputs.
    SubmissionTranscriptionDecision(
        mapping_ref="validation-item",
        observed_content=observed_content,
        normalized_answer=normalized_answer,
        alternatives=alternatives,
        is_blank=is_blank,
        has_erasure=has_erasure,
        confidence=confidence,
        evidence=["validated persisted transcription fields"],
        uncertainty_notes=uncertainty_notes,
        requires_review=requires_review,
    )
    if normalized_answer is not None and not is_blank:
        if question_type == QuestionType.OBJECTIVE_SINGLE:
            if normalized_answer not in option_labels:
                raise CaseValidationError(
                    f"{label} must normalize a single-choice response to an option label"
                )
        elif question_type == QuestionType.OBJECTIVE_MULTIPLE:
            labels = [
                token
                for token in re.split(r"[\s,;，、]+", normalized_answer.strip())
                if token
            ]
            if (
                not labels
                or len(labels) != len(set(labels))
                or any(token not in option_labels for token in labels)
            ):
                raise CaseValidationError(
                    f"{label} must normalize multiple choices to unique option labels"
                )
    for alternative in alternatives:
        if alternative.normalized_answer is None:
            continue
        if question_type == QuestionType.OBJECTIVE_SINGLE:
            if alternative.normalized_answer not in option_labels:
                raise CaseValidationError(
                    f"{label} alternative must normalize to an option label"
                )
        elif question_type == QuestionType.OBJECTIVE_MULTIPLE:
            alternative_labels = [
                token
                for token in re.split(
                    r"[\s,;，、]+", alternative.normalized_answer.strip()
                )
                if token
            ]
            if (
                not alternative_labels
                or len(alternative_labels) != len(set(alternative_labels))
                or any(token not in option_labels for token in alternative_labels)
            ):
                raise CaseValidationError(
                    f"{label} alternatives must normalize to unique option labels"
                )


def _validate_formal_mapping_coverage(
    mappings: Iterable[SubmissionMappingDecision],
    structure: SubmissionStructureManifest,
) -> None:
    expected = {
        (
            question.question_id,
            part.part_id if isinstance(part, SubmissionQuestionPart) else None,
        )
        for question in structure.questions
        for part in (question.parts or [None])
    }
    actual = {
        (item.question_id, item.part_id)
        for item in mappings
        if item.source_role == SubmissionSourceRole.ANSWER_SHEET
    }
    if actual != expected:
        raise CaseValidationError(
            "formal answer-sheet mappings must cover every response target; "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )


def _validate_formal_coverage(items: Iterable[SubmissionItem], master: ExamMaster) -> None:
    expected = {
        (question.question_id, part.part_id if part is not None else None)
        for question in master.questions
        for part in (question.subparts or [None])
    }
    actual = {
        (item.question_id, item.part_id)
        for item in items
        if item.source_role == SubmissionSourceRole.ANSWER_SHEET
    }
    if actual != expected:
        raise CaseValidationError(
            "final Submission must retain every formal response target; "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )


def _sorted_mappings(
    mappings: Iterable[SubmissionMappingDecision],
    structure: SubmissionStructureManifest,
) -> list[SubmissionMappingDecision]:
    question_order = {
        question.question_id: index
        for index, question in enumerate(structure.questions, start=1)
    }
    part_order = {
        (question.question_id, part.part_id): part.order
        for question in structure.questions
        for part in question.parts
    }
    role_order = {
        SubmissionSourceRole.ANSWER_SHEET: 0,
        SubmissionSourceRole.QUESTION_BOOKLET_SCRATCH: 1,
        SubmissionSourceRole.SCRATCH_SHEET: 2,
        SubmissionSourceRole.UNCERTAIN: 3,
    }
    return sorted(
        mappings,
        key=lambda item: (
            question_order[item.question_id],
            part_order.get((item.question_id, item.part_id), 0),
            role_order[item.source_role],
            item.slot_order,
            item.ref,
        ),
    )


def _page_bbox_to_raw(page, bbox: PixelBox) -> PixelBox:
    points = [
        apply_matrix(page.page_to_raw_matrix, point)
        for point in (
            (float(bbox.left), float(bbox.top)),
            (float(bbox.right - 1), float(bbox.top)),
            (float(bbox.left), float(bbox.bottom - 1)),
            (float(bbox.right - 1), float(bbox.bottom - 1)),
        )
    ]
    xs = [round(point[0]) for point in points]
    ys = [round(point[1]) for point in points]
    raw = PixelBox(
        left=min(xs),
        top=min(ys),
        right=max(xs) + 1,
        bottom=max(ys) + 1,
    )
    if not _box_within_image(
        raw, page.source_image_size.width, page.source_image_size.height
    ):
        raise CaseValidationError(f"page bbox maps outside raw source: {bbox}")
    return raw


def _render_crop(
    source_path: Path, bbox: PixelBox, *, jpeg_quality: int
) -> tuple[bytes, ImageSize]:
    try:
        with Image.open(source_path) as source:
            image = source.convert("RGB").crop(
                (bbox.left, bbox.top, bbox.right, bbox.bottom)
            )
    except (OSError, UnidentifiedImageError) as exc:
        raise CaseValidationError(
            f"cannot crop submission input from {source_path}: {exc}"
        ) from exc
    output = BytesIO()
    image.save(
        output,
        format="JPEG",
        quality=jpeg_quality,
        subsampling=0,
        exif=b"",
    )
    return output.getvalue(), ImageSize(width=image.width, height=image.height)


def _validate_jpeg(path: Path) -> None:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            if image.format != "JPEG" or image.width <= 0 or image.height <= 0:
                raise CaseValidationError(f"invalid submission crop at {path}")
    except (OSError, UnidentifiedImageError) as exc:
        raise CaseValidationError(f"cannot decode submission crop {path}: {exc}") from exc


def _active_document_graph(
    case_dir: Path, state: PipelineState
) -> tuple[Path, DocumentGraph]:
    completion = next(
        (item for item in state.completed_stages if item.stage == PipelineStage.MAPPED),
        None,
    )
    if completion is None:
        raise InvalidTransitionError("phase 5 requires an active document graph")
    references = [
        item
        for item in completion.artifacts
        if item.schema_id == "document_graph.schema.json"
    ]
    if len(references) != 1:
        raise CaseValidationError("mapped must have exactly one active document graph")
    path = case_dir / references[0].relative_path
    return path, read_document_graph(path)


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
        raise InvalidTransitionError("phase 5 requires an active page manifest")
    references = [
        item
        for item in completion.artifacts
        if item.schema_id == "page_manifest.schema.json"
    ]
    if len(references) != 1:
        raise CaseValidationError("pages_ready must have exactly one active page manifest")
    path = case_dir / references[0].relative_path
    return path, read_page_manifest(path)


def _active_exam_master(
    case_dir: Path, state: PipelineState
) -> tuple[Path, ExamMaster]:
    completion = next(
        (
            item
            for item in state.completed_stages
            if item.stage == PipelineStage.MASTER_READY
        ),
        None,
    )
    if completion is None:
        raise InvalidTransitionError("phase 5 requires an active Exam Master")
    references = [
        item for item in completion.artifacts if item.schema_id == "exam_master.schema.json"
    ]
    if len(references) != 1:
        raise CaseValidationError("master_ready must have exactly one active Exam Master")
    path = case_dir / references[0].relative_path
    return path, read_exam_master(path)


def _reject_submission_blockers(graph: DocumentGraph) -> None:
    blockers = [item for item in graph.review_items if item.blocks_submission_ready]
    if blockers:
        reasons = "; ".join(item.reason for item in blockers)
        raise ReviewRequiredError(f"document graph blocks submission mapping: {reasons}")


def _read_model_file(path: Path, model_type, label: str):
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise CaseValidationError(f"cannot read {label} file {path}: {exc}") from exc
    return validate_json(model_type, raw)


def _structure_signature(manifest: SubmissionStructureManifest) -> dict[str, object]:
    payload = manifest.model_dump(mode="json")
    payload.pop("created_at", None)
    return payload


def _box_contains(outer: PixelBox, inner: PixelBox) -> bool:
    return (
        inner.left >= outer.left
        and inner.top >= outer.top
        and inner.right <= outer.right
        and inner.bottom <= outer.bottom
    )


def _boxes_intersect(first: PixelBox, second: PixelBox) -> bool:
    return not (
        first.right <= second.left
        or second.right <= first.left
        or first.bottom <= second.top
        or second.bottom <= first.top
    )


def _box_within_image(bbox: PixelBox, width: int, height: int) -> bool:
    return bbox.right <= width and bbox.bottom <= height


def _stable_id(kind: str, *parts: str) -> str:
    digest = digest_value({"kind": kind, "parts": list(parts)})
    return f"{kind}-{digest[:20]}"


def _unique_strings(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
