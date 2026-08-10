"""Prepare blind inputs and compile reviewed phase 4 decisions into an Exam Master."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
from pathlib import Path
from typing import Callable, Iterable

from PIL import Image, UnidentifiedImageError

from student_analyze import __version__
from student_analyze.assets import artifact_digest
from student_analyze.atomic import atomic_write_bytes, atomic_write_model, serialize_model
from student_analyze.config import AppConfig
from student_analyze.document_mapper import read_document_graph
from student_analyze.document_models import (
    DecisionMethod,
    DocumentGraph,
    DocumentRole,
    RegionKind,
)
from student_analyze.errors import (
    CaseValidationError,
    InvalidTransitionError,
    ReviewRequiredError,
)
from student_analyze.exam_master_models import (
    AnswerEntryDecision,
    AnswerEvidenceDecision,
    AnswerEvidenceDecisionSet,
    AnswerProvenance,
    AnswerSource,
    AnswerSupport,
    ApprovalStatus,
    CropReviewStatus,
    EvidenceDecisionStatus,
    EvidenceSourceKind,
    EndorsementKind,
    ExamMaster,
    ExamMasterDecisionSet,
    ExamReviewDecisionSet,
    IndependentReviewStatus,
    MasterQuestion,
    MasterReviewItem,
    MasterSubpart,
    QuestionMasterDecision,
    QuestionReconstructionDecision,
    QuestionReconstructionDecisionSet,
    RubricCriterionDecision,
    RubricSupport,
    SolverCropAsset,
    SolverInputDecisionSet,
    SolverInputManifest,
    SolverQuestionInput,
    SolverRouteReason,
    SolverTask,
    VerificationLevel,
    VerificationMethod,
    VerificationResult,
    VerificationStatus,
)
from student_analyze.exam_verification import run_deterministic_verifications
from student_analyze.fingerprint import digest_value
from student_analyze.models import (
    SCHEMA_VERSION,
    ImplementationVersions,
    PipelineStage,
    PipelineState,
)
from student_analyze.page_models import ImageSize, PageManifest
from student_analyze.page_verification import read_page_manifest
from student_analyze.pipeline import (
    build_stage_fingerprint,
    commit_stage_artifact,
    verify_case,
)
from student_analyze.validation import validate_json


InterruptHook = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class SolverInputPreparationResult:
    case_dir: Path
    manifest_path: Path
    manifest: SolverInputManifest
    reused: bool


@dataclass(frozen=True, slots=True)
class ExamMasterBuildResult:
    case_dir: Path
    master: ExamMaster
    state: PipelineState
    reused: bool


def load_answer_evidence_decisions(path: Path) -> AnswerEvidenceDecisionSet:
    return _read_decision_file(path, AnswerEvidenceDecisionSet, "answer evidence")


def load_solver_input_decisions(path: Path) -> SolverInputDecisionSet:
    return _read_decision_file(path, SolverInputDecisionSet, "solver input")


def load_question_reconstruction_decisions(
    path: Path,
) -> QuestionReconstructionDecisionSet:
    return _read_decision_file(path, QuestionReconstructionDecisionSet, "question reconstruction")


def load_exam_master_decisions(path: Path) -> ExamMasterDecisionSet:
    return _read_decision_file(path, ExamMasterDecisionSet, "Exam Master")


def load_exam_review_decisions(path: Path) -> ExamReviewDecisionSet:
    return _read_decision_file(path, ExamReviewDecisionSet, "exam review")


def read_solver_input_manifest(path: Path) -> SolverInputManifest:
    return _read_decision_file(path, SolverInputManifest, "solver input manifest")


def read_exam_master(path: Path) -> ExamMaster:
    return _read_decision_file(path, ExamMaster, "Exam Master artifact")


def prepare_solver_inputs(
    case_dir: Path,
    evidence_decisions: AnswerEvidenceDecisionSet,
    reconstruction_decisions: QuestionReconstructionDecisionSet,
    crop_decisions: SolverInputDecisionSet,
    config: AppConfig,
    *,
    interrupt_hook: InterruptHook | None = None,
) -> SolverInputPreparationResult:
    case_dir = case_dir.resolve(strict=True)
    _, state = verify_case(case_dir)
    if state.current_stage != PipelineStage.MAPPED:
        raise InvalidTransitionError(
            f"cannot prepare phase 4 solver inputs from current stage {state.current_stage}"
        )
    graph_path, graph = _active_document_graph(case_dir, state)
    graph_sha256, _ = artifact_digest(graph_path)
    page_manifest_path, page_manifest = _active_page_manifest(case_dir, state)

    _validate_answer_evidence(
        evidence_decisions,
        graph,
        page_manifest,
        graph_sha256=graph_sha256,
    )
    _validate_question_reconstructions(
        reconstruction_decisions,
        graph,
        graph_sha256=graph_sha256,
    )
    _validate_crop_decisions(
        crop_decisions,
        reconstruction_decisions,
        graph,
        graph_sha256=graph_sha256,
    )
    evidence_sha256 = digest_value(evidence_decisions.model_dump(mode="json"))
    reconstruction_sha256 = digest_value(
        reconstruction_decisions.model_dump(mode="json")
    )
    crop_decision_sha256 = digest_value(crop_decisions.model_dump(mode="json"))
    manifest_fingerprint = digest_value(
        {
            "document_graph_sha256": graph_sha256,
            "answer_evidence_decision_sha256": evidence_sha256,
            "question_reconstruction_decision_sha256": reconstruction_sha256,
            "crop_decision_sha256": crop_decision_sha256,
            "config": config.master_input_fingerprint_payload(),
        }
    )
    output_dir = case_dir / "work" / "master_inputs" / manifest_fingerprint
    manifest_path = output_dir / "solver_input_manifest.json"
    if manifest_path.exists():
        manifest = read_solver_input_manifest(manifest_path)
        if manifest.manifest_fingerprint != manifest_fingerprint:
            raise CaseValidationError(
                f"conflicting solver input manifest at {manifest_path}"
            )
        verify_solver_input_manifest(
            case_dir,
            manifest,
            graph,
            graph_sha256=graph_sha256,
        )
        return SolverInputPreparationResult(case_dir, manifest_path, manifest, reused=True)

    manifest_questions: list[SolverQuestionInput] = []
    crops_by_question: dict[str, list] = {}
    for crop in crop_decisions.crops:
        crops_by_question.setdefault(crop.question_id, []).append(crop)
    page_by_id = {page.page_id: page for page in page_manifest.pages}
    reconstruction_by_question = {
        question.question_id: question
        for question in reconstruction_decisions.questions
    }

    for question in sorted(graph.questions, key=lambda item: item.order):
        if question.effective_version_id is None:
            raise ReviewRequiredError(
                f"question {question.question_id} has no effective version"
            )
        route_reason = _route_question(question.question_id, evidence_decisions.items)
        task = (
            SolverTask.RECONSTRUCT_ONLY
            if route_reason == SolverRouteReason.RELIABLE_EVIDENCE
            else SolverTask.RECONSTRUCT_AND_SOLVE
        )
        assets: list[SolverCropAsset] = []
        for crop in sorted(
            crops_by_question.get(question.question_id, []),
            key=lambda item: (item.page_id, item.bbox.top, item.bbox.left, item.ref),
        ):
            source_page = page_by_id[crop.page_id]
            relative_path = (
                Path("work")
                / "master_inputs"
                / manifest_fingerprint
                / "crops"
                / f"{crop.ref}.jpg"
            )
            crop_path = case_dir / relative_path
            content, image_size = _render_crop(
                case_dir / source_page.derived.relative_path,
                crop.bbox,
                jpeg_quality=config.jpeg_quality,
            )
            if crop_path.exists():
                existing_sha256, existing_size = artifact_digest(crop_path)
                expected_sha256 = sha256(content).hexdigest()
                if existing_sha256 != expected_sha256 or existing_size != len(content):
                    raise CaseValidationError(f"conflicting solver crop at {crop_path}")
            else:
                atomic_write_bytes(
                    crop_path,
                    content,
                    validator=_validate_jpeg,
                    interrupt_hook=interrupt_hook,
                )
            crop_sha256, crop_size = artifact_digest(crop_path)
            assets.append(
                SolverCropAsset(
                    asset_id=_stable_id("solver-asset", graph.case_id, crop.ref),
                    crop_decision_ref=crop.ref,
                    question_id=crop.question_id,
                    version_id=crop.version_id,
                    source_region_id=crop.source_region_id,
                    page_id=crop.page_id,
                    bbox=crop.bbox,
                    relative_path=relative_path.as_posix(),
                    sha256=crop_sha256,
                    size_bytes=crop_size,
                    image_size=image_size,
                )
            )
        manifest_questions.append(
            SolverQuestionInput(
                question_id=question.question_id,
                version_id=question.effective_version_id,
                printed_label=question.printed_label,
                task=task,
                route_reason=route_reason,
                reconstruction=reconstruction_by_question[question.question_id],
                assets=assets,
            )
        )

    manifest = SolverInputManifest(
        case_id=graph.case_id,
        document_graph_sha256=graph_sha256,
        answer_evidence_decision_sha256=evidence_sha256,
        question_reconstruction_decision_sha256=reconstruction_sha256,
        crop_decision_sha256=crop_decision_sha256,
        manifest_fingerprint=manifest_fingerprint,
        created_at=datetime.now(UTC),
        question_reconstruction_provenance=reconstruction_decisions.provenance,
        questions=manifest_questions,
    )
    atomic_write_model(
        manifest_path,
        manifest,
        model_type=SolverInputManifest,
        interrupt_hook=interrupt_hook,
    )
    verify_solver_input_manifest(
        case_dir,
        manifest,
        graph,
        graph_sha256=graph_sha256,
    )
    return SolverInputPreparationResult(case_dir, manifest_path, manifest, reused=False)


def build_exam_master(
    case_dir: Path,
    evidence_decisions: AnswerEvidenceDecisionSet,
    solver_manifest: SolverInputManifest,
    master_decisions: ExamMasterDecisionSet,
    review_decisions: ExamReviewDecisionSet,
    config: AppConfig,
    *,
    force: bool = False,
    interrupt_hook: InterruptHook | None = None,
) -> ExamMasterBuildResult:
    case_dir = case_dir.resolve(strict=True)
    _, state = verify_case(case_dir)
    graph_path, graph = _active_document_graph(case_dir, state)
    graph_sha256, _ = artifact_digest(graph_path)
    _, page_manifest = _active_page_manifest(case_dir, state)
    _validate_answer_evidence(
        evidence_decisions,
        graph,
        page_manifest,
        graph_sha256=graph_sha256,
    )
    verify_solver_input_manifest(
        case_dir,
        solver_manifest,
        graph,
        graph_sha256=graph_sha256,
    )

    solver_manifest_sha256 = sha256(serialize_model(solver_manifest)).hexdigest()
    evidence_sha256 = digest_value(evidence_decisions.model_dump(mode="json"))
    master_decision_sha256 = digest_value(master_decisions.model_dump(mode="json"))
    review_decision_sha256 = digest_value(review_decisions.model_dump(mode="json"))
    if solver_manifest.answer_evidence_decision_sha256 != evidence_sha256:
        raise CaseValidationError("solver manifest references different answer evidence")
    _validate_master_decisions(
        master_decisions,
        solver_manifest,
        graph,
        graph_sha256=graph_sha256,
        manifest_sha256=solver_manifest_sha256,
    )
    _validate_review_decisions(
        review_decisions,
        master_decisions,
        solver_manifest,
        manifest_sha256=solver_manifest_sha256,
        master_decision_sha256=master_decision_sha256,
    )

    versions = ImplementationVersions(
        code=__version__,
        base_schema=SCHEMA_VERSION,
        config=config.config_version,
        model=master_decisions.provenance.model_identifier,
        prompt=master_decisions.provenance.prompt_version,
        skill=master_decisions.provenance.skill_version,
        tools={"pillow": version("Pillow")},
    )
    stage_inputs = [
        {
            "document_graph_sha256": graph_sha256,
            "answer_evidence_decision_sha256": evidence_sha256,
            "solver_input_manifest_sha256": solver_manifest_sha256,
            "exam_master_decision_sha256": master_decision_sha256,
            "exam_review_decision_sha256": review_decision_sha256,
        }
    ]
    stage_config = config.master_fingerprint_payload()
    stage_fingerprint, _ = build_stage_fingerprint(
        stage=PipelineStage.MASTER_READY,
        model_type=ExamMaster,
        schema_id="exam_master.schema.json",
        config=stage_config,
        versions=versions,
        inputs=stage_inputs,
    )

    existing_completion = next(
        (
            completion
            for completion in state.completed_stages
            if completion.stage == PipelineStage.MASTER_READY
        ),
        None,
    )
    if existing_completion is not None:
        if existing_completion.stage_fingerprint == stage_fingerprint and not force:
            reference = next(
                reference
                for reference in existing_completion.artifacts
                if reference.schema_id == "exam_master.schema.json"
            )
            master = read_exam_master(case_dir / reference.relative_path)
            verify_exam_master(
                case_dir,
                master,
                graph,
                graph_sha256=graph_sha256,
            )
            return ExamMasterBuildResult(case_dir, master, state, reused=True)
        if not force:
            raise InvalidTransitionError(
                "master_ready is already complete with a different fingerprint; "
                "use force to create a preserved version"
            )
    elif state.current_stage != PipelineStage.MAPPED:
        raise InvalidTransitionError(
            f"cannot build Exam Master from current stage {state.current_stage}"
        )

    master = _compile_exam_master(
        graph,
        evidence_decisions,
        solver_manifest,
        master_decisions,
        review_decisions,
        graph_sha256=graph_sha256,
        evidence_sha256=evidence_sha256,
        solver_manifest_sha256=solver_manifest_sha256,
        master_decision_sha256=master_decision_sha256,
        review_decision_sha256=review_decision_sha256,
        stage_fingerprint=stage_fingerprint,
    )
    if master.requires_review:
        reasons = "; ".join(item.reason for item in master.review_items)
        raise ReviewRequiredError(f"Exam Master requires review: {reasons}")
    verify_exam_master(
        case_dir,
        master,
        graph,
        graph_sha256=graph_sha256,
    )

    orphan_path = (
        case_dir
        / "artifacts"
        / PipelineStage.MASTER_READY.value
        / stage_fingerprint
        / "exam_master.json"
    )
    if orphan_path.exists() and not force:
        orphan = read_exam_master(orphan_path)
        if (
            orphan.case_id != master.case_id
            or orphan.stage_fingerprint != master.stage_fingerprint
            or orphan.exam_master_decision_sha256 != master.exam_master_decision_sha256
        ):
            raise CaseValidationError(f"conflicting uncommitted Exam Master at {orphan_path}")
        master = orphan

    committed = commit_stage_artifact(
        case_dir,
        stage=PipelineStage.MASTER_READY,
        artifact_name="exam_master.json",
        payload=master,
        model_type=ExamMaster,
        schema_id="exam_master.schema.json",
        config=stage_config,
        versions=versions,
        inputs=stage_inputs,
        force=force,
        human_confirmed=(
            evidence_decisions.provenance.method == DecisionMethod.HUMAN_REVIEW
            or review_decisions.provenance.method == DecisionMethod.HUMAN_REVIEW
        ),
        interrupt_hook=interrupt_hook,
    )
    reference = committed.artifacts[0]
    active_master = read_exam_master(case_dir / reference.relative_path)
    return ExamMasterBuildResult(
        case_dir,
        active_master,
        committed.state,
        reused=committed.reused,
    )


def verify_solver_input_manifest(
    case_dir: Path,
    manifest: SolverInputManifest,
    graph: DocumentGraph,
    *,
    graph_sha256: str,
) -> None:
    if manifest.case_id != graph.case_id:
        raise CaseValidationError("solver manifest belongs to a different case")
    if manifest.document_graph_sha256 != graph_sha256:
        raise CaseValidationError("solver manifest references a different document graph")
    graph_questions = {question.question_id: question for question in graph.questions}
    manifest_questions = {question.question_id: question for question in manifest.questions}
    if set(manifest_questions) != set(graph_questions):
        raise CaseValidationError("solver manifest must cover every mapped question exactly once")
    regions = {region.region_id: region for region in graph.regions}
    if (
        any(
            question.reconstruction.human_confirmed
            for question in manifest.questions
        )
        and manifest.question_reconstruction_provenance.method
        != DecisionMethod.HUMAN_REVIEW
    ):
        raise CaseValidationError(
            "embedded human-confirmed reconstructions lack human_review provenance"
        )
    for item in manifest.questions:
        graph_question = graph_questions[item.question_id]
        if item.version_id != graph_question.effective_version_id:
            raise CaseValidationError(
                f"solver input {item.question_id} does not use the effective version"
            )
        for asset in item.assets:
            region = regions.get(asset.source_region_id)
            if (
                region is None
                or region.kind != RegionKind.PRINTED_QUESTION
                or region.page_id != asset.page_id
            ):
                raise CaseValidationError(
                    f"solver asset {asset.asset_id} is not derived from printed question evidence"
                )
            path = case_dir / asset.relative_path
            digest, size = artifact_digest(path)
            if digest != asset.sha256 or size != asset.size_bytes:
                raise CaseValidationError(f"solver asset changed: {path}")
            try:
                with Image.open(path) as image:
                    if image.size != (asset.image_size.width, asset.image_size.height):
                        raise CaseValidationError(
                            f"solver asset dimensions changed: {path}"
                        )
            except (OSError, UnidentifiedImageError) as exc:
                raise CaseValidationError(f"cannot decode solver asset {path}: {exc}") from exc


def verify_exam_master(
    case_dir: Path,
    master: ExamMaster,
    graph: DocumentGraph,
    *,
    graph_sha256: str,
) -> None:
    if master.case_id != graph.case_id:
        raise CaseValidationError("Exam Master belongs to a different case")
    if master.document_graph_sha256 != graph_sha256:
        raise CaseValidationError("Exam Master references a different document graph")
    embedded_manifest_sha256 = sha256(
        serialize_model(master.solver_input_manifest)
    ).hexdigest()
    if embedded_manifest_sha256 != master.solver_input_manifest_sha256:
        raise CaseValidationError("embedded solver manifest hash mismatch")
    verify_solver_input_manifest(
        case_dir,
        master.solver_input_manifest,
        graph,
        graph_sha256=graph_sha256,
    )
    graph_questions = {question.question_id: question for question in graph.questions}
    master_questions = {question.question_id: question for question in master.questions}
    if set(master_questions) != set(graph_questions):
        raise CaseValidationError("Exam Master must cover every mapped question exactly once")
    for question_id, question in master_questions.items():
        if question.version_id != graph_questions[question_id].effective_version_id:
            raise CaseValidationError(
                f"Exam Master question {question_id} does not use the effective version"
            )
        if question.approval_status != ApprovalStatus.APPROVED or question.requires_review:
            raise CaseValidationError(
                f"master_ready contains unapproved question {question_id}"
            )
        if any(
            result.status in {VerificationStatus.FAILED, VerificationStatus.REQUIRES_REVIEW}
            for result in question.verification_results
        ):
            raise CaseValidationError(
                f"master_ready contains failed verification for {question_id}"
            )
    if master.requires_review or master.review_items:
        raise CaseValidationError("master_ready cannot contain unresolved review items")


def _validate_answer_evidence(
    decisions: AnswerEvidenceDecisionSet,
    graph: DocumentGraph,
    page_manifest: PageManifest,
    *,
    graph_sha256: str,
) -> None:
    if decisions.case_id != graph.case_id:
        raise CaseValidationError("answer evidence belongs to a different case")
    if decisions.document_graph_sha256 != graph_sha256:
        raise CaseValidationError("answer evidence references a different document graph")
    question_ids = {question.question_id for question in graph.questions}
    graph_page_ids = {page.page_id for page in graph.pages}
    page_by_id = {page.page_id: page for page in page_manifest.pages}
    if (
        any(item.human_confirmed for item in decisions.items)
        and decisions.provenance.method != DecisionMethod.HUMAN_REVIEW
    ):
        raise CaseValidationError(
            "human-confirmed teacher evidence requires human_review provenance"
        )
    for item in decisions.items:
        if not set(item.question_ids) <= question_ids:
            raise CaseValidationError(f"answer evidence {item.ref} targets unknown questions")
        if item.page_id not in graph_page_ids or item.page_id not in page_by_id:
            raise CaseValidationError(f"answer evidence {item.ref} references an unknown page")
        page = page_by_id[item.page_id]
        if item.bbox.right > page.page_image_size.width or item.bbox.bottom > page.page_image_size.height:
            raise CaseValidationError(f"answer evidence {item.ref} exceeds its logical page")


def _validate_crop_decisions(
    decisions: SolverInputDecisionSet,
    reconstructions: QuestionReconstructionDecisionSet,
    graph: DocumentGraph,
    *,
    graph_sha256: str,
) -> None:
    if decisions.case_id != graph.case_id:
        raise CaseValidationError("solver crop decisions belong to a different case")
    if decisions.document_graph_sha256 != graph_sha256:
        raise CaseValidationError("solver crop decisions reference a different document graph")
    questions = {question.question_id: question for question in graph.questions}
    versions = {version.version_id: version for version in graph.question_versions}
    regions = {region.region_id: region for region in graph.regions}
    pages = {page.page_id: page for page in graph.pages}
    documents = {document.document_id: document for document in graph.documents}
    reconstruction_by_question = {
        question.question_id: question for question in reconstructions.questions
    }
    covered_regions: set[str] = set()
    for crop in decisions.crops:
        question = questions.get(crop.question_id)
        if question is None or question.effective_version_id != crop.version_id:
            raise CaseValidationError(
                f"solver crop {crop.ref} does not use an effective question version"
            )
        version_node = versions[crop.version_id]
        if crop.source_region_id not in version_node.region_ids:
            raise CaseValidationError(
                f"solver crop {crop.ref} is outside the effective question version"
            )
        region = regions[crop.source_region_id]
        if region.kind != RegionKind.PRINTED_QUESTION or region.page_id != crop.page_id:
            raise CaseValidationError(
                f"solver crop {crop.ref} must use a printed question region"
            )
        if not _box_contains(region.bbox, crop.bbox):
            raise CaseValidationError(
                f"solver crop {crop.ref} must stay inside its printed question region"
            )
        document = documents[pages[crop.page_id].document_id]
        if document.role not in {
            DocumentRole.QUESTION_BOOKLET,
            DocumentRole.REPLACEMENT_OR_ERRATA,
            DocumentRole.SUPPLEMENTAL_MATERIAL,
        }:
            raise CaseValidationError(
                f"solver crop {crop.ref} uses forbidden document role {document.role.value}"
            )
        if (
            crop.contains_student_handwriting
            or crop.contains_teacher_annotation
            or crop.contains_answer_content
            or crop.requires_review
        ):
            raise ReviewRequiredError(
                f"solver crop {crop.ref} is not a reviewed question-only input"
            )
        if crop.review_status not in {
            CropReviewStatus.VISUALLY_REVIEWED,
            CropReviewStatus.HUMAN_REVIEWED,
            CropReviewStatus.DETERMINISTIC_TEST_FIXTURE,
        }:
            raise ReviewRequiredError(f"solver crop {crop.ref} lacks visual review")
        covered_regions.add(crop.source_region_id)

    required_regions = {
        region_id
        for question in reconstruction_by_question.values()
        for region_id in question.visual_region_ids
    }
    if covered_regions != required_regions:
        missing = sorted(required_regions - covered_regions)
        unexpected = sorted(covered_regions - required_regions)
        raise CaseValidationError(
            f"solver crops must cover every effective printed region; "
            f"missing={missing}, unexpected={unexpected}"
        )


def _validate_question_reconstructions(
    decisions: QuestionReconstructionDecisionSet,
    graph: DocumentGraph,
    *,
    graph_sha256: str,
) -> None:
    if decisions.case_id != graph.case_id:
        raise CaseValidationError("question reconstructions belong to a different case")
    if decisions.document_graph_sha256 != graph_sha256:
        raise CaseValidationError(
            "question reconstructions reference a different document graph"
        )
    graph_questions = {question.question_id: question for question in graph.questions}
    reconstructions = {question.question_id: question for question in decisions.questions}
    if set(reconstructions) != set(graph_questions):
        raise CaseValidationError(
            "question reconstructions must cover every mapped question exactly once"
        )
    versions = {version.version_id: version for version in graph.question_versions}
    if (
        any(question.human_confirmed for question in decisions.questions)
        and decisions.provenance.method != DecisionMethod.HUMAN_REVIEW
    ):
        raise CaseValidationError(
            "human-confirmed question reconstructions require human_review provenance"
        )
    for question_id, reconstruction in reconstructions.items():
        graph_question = graph_questions[question_id]
        if reconstruction.version_id != graph_question.effective_version_id:
            raise CaseValidationError(
                f"question reconstruction {question_id} does not use the effective version"
            )
        version_node = versions[reconstruction.version_id]
        if not set(reconstruction.visual_region_ids) <= set(version_node.region_ids):
            raise CaseValidationError(
                f"question reconstruction {question_id} requests unrelated visual regions"
            )
        contaminated = (
            reconstruction.source_contains_student_content
            or reconstruction.source_contains_teacher_annotation
        )
        if reconstruction.requires_review:
            raise ReviewRequiredError(
                f"question reconstruction {question_id} requires review"
            )
        if contaminated and not reconstruction.human_confirmed:
            raise ReviewRequiredError(
                f"marked-source reconstruction {question_id} lacks human confirmation"
            )


def _validate_master_decisions(
    decisions: ExamMasterDecisionSet,
    manifest: SolverInputManifest,
    graph: DocumentGraph,
    *,
    graph_sha256: str,
    manifest_sha256: str,
) -> None:
    if decisions.case_id != graph.case_id:
        raise CaseValidationError("Exam Master decisions belong to a different case")
    if decisions.document_graph_sha256 != graph_sha256:
        raise CaseValidationError("Exam Master decisions reference a different graph")
    if decisions.solver_input_manifest_sha256 != manifest_sha256:
        raise CaseValidationError("Exam Master decisions reference a different solver manifest")
    manifest_by_question = {question.question_id: question for question in manifest.questions}
    decision_by_question = {question.question_id: question for question in decisions.questions}
    if set(decision_by_question) != set(manifest_by_question):
        raise CaseValidationError("Exam Master decisions must cover every solver input question")
    for question_id, decision in decision_by_question.items():
        manifest_question = manifest_by_question[question_id]
        if decision.version_id != manifest_question.version_id:
            raise CaseValidationError(
                f"Exam Master decision {question_id} uses a different version"
            )
        expected_solver = manifest_question.task == SolverTask.RECONSTRUCT_AND_SOLVE
        if decision.solver_performed != expected_solver:
            raise CaseValidationError(
                f"Exam Master decision {question_id} violates its solver route"
            )
        reconstruction = manifest_question.reconstruction
        if (
            decision.prompt_text != reconstruction.prompt_text
            or decision.question_type != reconstruction.question_type
            or decision.points != reconstruction.points
            or decision.options != reconstruction.options
            or decision.subparts != reconstruction.subparts
            or decision.knowledge_points != reconstruction.knowledge_points
        ):
            raise CaseValidationError(
                f"solver output {question_id} changed the reviewed question reconstruction"
            )


def _validate_review_decisions(
    reviews: ExamReviewDecisionSet,
    decisions: ExamMasterDecisionSet,
    manifest: SolverInputManifest,
    *,
    manifest_sha256: str,
    master_decision_sha256: str,
) -> None:
    if reviews.case_id != decisions.case_id:
        raise CaseValidationError("review decisions belong to a different case")
    if reviews.solver_input_manifest_sha256 != manifest_sha256:
        raise CaseValidationError("review decisions reference a different solver manifest")
    if reviews.exam_master_decision_sha256 != master_decision_sha256:
        raise CaseValidationError("review decisions reference a different solver output")
    if reviews.provenance.prompt_version == decisions.provenance.prompt_version:
        raise CaseValidationError(
            "independent review must use a distinct prompt version from the solver"
        )
    required = {
        question.question_id
        for question in manifest.questions
        if question.task == SolverTask.RECONSTRUCT_AND_SOLVE
    }
    actual = {review.question_id for review in reviews.reviews}
    if actual != required:
        raise CaseValidationError(
            f"independent review must cover exactly the solved questions; "
            f"missing={sorted(required - actual)}, unexpected={sorted(actual - required)}"
        )
    versions = {question.question_id: question.version_id for question in manifest.questions}
    for review in reviews.reviews:
        if review.version_id != versions[review.question_id]:
            raise CaseValidationError(
                f"independent review {review.question_id} uses a different version"
            )


def _compile_exam_master(
    graph: DocumentGraph,
    evidence_decisions: AnswerEvidenceDecisionSet,
    solver_manifest: SolverInputManifest,
    master_decisions: ExamMasterDecisionSet,
    review_decisions: ExamReviewDecisionSet,
    *,
    graph_sha256: str,
    evidence_sha256: str,
    solver_manifest_sha256: str,
    master_decision_sha256: str,
    review_decision_sha256: str,
    stage_fingerprint: str,
) -> ExamMaster:
    graph_questions = {question.question_id: question for question in graph.questions}
    decisions = {question.question_id: question for question in master_decisions.questions}
    reviews = {review.question_id: review for review in review_decisions.reviews}
    questions: list[MasterQuestion] = []
    review_items: list[MasterReviewItem] = []

    for manifest_question in solver_manifest.questions:
        question_id = manifest_question.question_id
        graph_question = graph_questions[question_id]
        decision = decisions[question_id]
        reconstruction = manifest_question.reconstruction
        part_ids = {
            part.ref: _stable_id("part", question_id, part.ref)
            for part in reconstruction.subparts
        }
        subparts = [
            MasterSubpart(
                part_id=part_ids[part.ref],
                decision_ref=part.ref,
                printed_label=part.printed_label,
                prompt_text=part.prompt_text,
                points=part.points,
            )
            for part in reconstruction.subparts
        ]
        reasons: list[str] = []

        if manifest_question.task == SolverTask.RECONSTRUCT_ONLY:
            usable = _usable_evidence(question_id, evidence_decisions.items)
            signatures = {_answer_signature(item, question_id) for item in usable}
            if not usable or len(signatures) != 1:
                reasons.append("Reliable answer evidence is absent or conflicting.")
                answers: list[AnswerEntryDecision] = []
                rubric: list[RubricCriterionDecision] = []
                provenance: list[AnswerProvenance] = []
            else:
                primary = usable[0]
                answers = _answers_from_evidence(
                    primary, question_id, reconstruction, part_ids
                )
                rubric = _rubric_from_evidence(
                    primary, question_id, reconstruction, answers, part_ids
                )
                provenance = [
                    AnswerProvenance(
                        source=_answer_source(item),
                        evidence_refs=[item.ref],
                        source_scope=item.question_ids,
                        verification_level=(
                            VerificationLevel.SOURCE_VALIDATED
                            if item.source_kind == EvidenceSourceKind.OFFICIAL_ANSWER
                            else VerificationLevel.HUMAN_CONFIRMED
                        ),
                    )
                    for item in usable
                ]
            solution_summary = None
            assumptions: list[str] = []
        else:
            answers = _remap_answers(decision.reference_answers, part_ids)
            rubric = _remap_rubric(decision.rubric, question_id, part_ids)
            provenance = [
                AnswerProvenance(
                    source=AnswerSource.INDEPENDENT_SOLUTION,
                    source_scope=[question_id],
                    verification_level=VerificationLevel.INDEPENDENTLY_REVIEWED,
                )
            ]
            solution_summary = decision.solution_summary
            assumptions = decision.assumptions
            review = reviews[question_id]
            if review.status != IndependentReviewStatus.CONFIRMED:
                reasons.append(
                    f"Independent review status is {review.status.value}: "
                    + "; ".join(review.evidence)
                )
            if decision.uncertainty:
                reasons.append("Solver uncertainty remains: " + "; ".join(decision.uncertainty))
            if manifest_question.route_reason == SolverRouteReason.CONFLICTING_EVIDENCE:
                reasons.append("Existing reference-answer evidence conflicts and needs resolution.")

        verification_results = run_deterministic_verifications(decision, answers, rubric)
        failures = [
            result.details
            for result in verification_results
            if result.status in {VerificationStatus.FAILED, VerificationStatus.REQUIRES_REVIEW}
        ]
        reasons.extend(failures)
        requires_review = bool(reasons)
        if requires_review:
            review_items.append(
                MasterReviewItem(question_id=question_id, reason=" ".join(reasons))
            )
        questions.append(
            MasterQuestion(
                question_id=question_id,
                version_id=manifest_question.version_id,
                printed_label=graph_question.printed_label,
                prompt_text=reconstruction.prompt_text,
                question_type=reconstruction.question_type,
                points=reconstruction.points,
                options=reconstruction.options,
                subparts=subparts,
                knowledge_points=reconstruction.knowledge_points,
                solver_required=manifest_question.task == SolverTask.RECONSTRUCT_AND_SOLVE,
                route_reason=manifest_question.route_reason,
                reference_answers=answers or [AnswerEntryDecision(answer="unresolved")],
                rubric=rubric
                or [
                    RubricCriterionDecision(
                        ref=_stable_id("rubric", question_id, "unresolved"),
                        description="Unresolved pending human review.",
                    )
                ],
                solution_summary=solution_summary,
                assumptions=assumptions,
                answer_provenance=provenance
                or [
                    AnswerProvenance(
                        source=AnswerSource.INDEPENDENT_SOLUTION,
                        source_scope=[question_id],
                        verification_level=VerificationLevel.INDEPENDENTLY_REVIEWED,
                    )
                ],
                verification_results=verification_results
                or [
                    VerificationResult(
                        method=VerificationMethod.LOGIC_REVIEW,
                        status=VerificationStatus.REQUIRES_REVIEW,
                        details="No verification result is available.",
                    )
                ],
                approval_status=(
                    ApprovalStatus.REQUIRES_REVIEW
                    if requires_review
                    else ApprovalStatus.APPROVED
                ),
                requires_review=requires_review,
                warnings=[
                    *reconstruction.warnings,
                    *decision.warnings,
                    *decision.uncertainty,
                ],
            )
        )

    return ExamMaster(
        case_id=graph.case_id,
        document_graph_sha256=graph_sha256,
        answer_evidence_decision_sha256=evidence_sha256,
        question_reconstruction_decision_sha256=(
            solver_manifest.question_reconstruction_decision_sha256
        ),
        crop_decision_sha256=solver_manifest.crop_decision_sha256,
        solver_input_manifest_sha256=solver_manifest_sha256,
        exam_master_decision_sha256=master_decision_sha256,
        exam_review_decision_sha256=review_decision_sha256,
        stage_fingerprint=stage_fingerprint,
        created_at=datetime.now(UTC),
        evidence_provenance=evidence_decisions.provenance,
        reconstruction_provenance=(
            solver_manifest.question_reconstruction_provenance
        ),
        solver_provenance=master_decisions.provenance,
        review_provenance=review_decisions.provenance,
        answer_evidence=evidence_decisions.items,
        solver_input_manifest=solver_manifest,
        questions=questions,
        review_items=review_items,
        requires_review=bool(review_items),
        warnings=_unique_strings(
            [
                *evidence_decisions.warnings,
                *master_decisions.warnings,
                *review_decisions.warnings,
            ]
        ),
    )


def _route_question(
    question_id: str,
    evidence_items: Iterable[AnswerEvidenceDecision],
) -> SolverRouteReason:
    relevant = [item for item in evidence_items if question_id in item.question_ids]
    usable = _usable_evidence(question_id, relevant)
    if usable:
        signatures = {_answer_signature(item, question_id) for item in usable}
        if len(signatures) == 1:
            return SolverRouteReason.RELIABLE_EVIDENCE
        return SolverRouteReason.CONFLICTING_EVIDENCE
    if any(
        item.answer_support == AnswerSupport.ACCEPTED_EXAMPLE
        or (
            item.answer_support == AnswerSupport.ANSWER_KEY
            and item.rubric_support
            in {RubricSupport.NONE, RubricSupport.ACCEPTED_EXAMPLE}
        )
        for item in relevant
    ):
        return SolverRouteReason.INCOMPLETE_RUBRIC
    return SolverRouteReason.NO_RELIABLE_EVIDENCE


def _usable_evidence(
    question_id: str,
    evidence_items: Iterable[AnswerEvidenceDecision],
) -> list[AnswerEvidenceDecision]:
    usable: list[AnswerEvidenceDecision] = []
    for item in evidence_items:
        if question_id not in item.question_ids:
            continue
        if item.status != EvidenceDecisionStatus.ACCEPTED or item.requires_review:
            continue
        if item.answer_support != AnswerSupport.ANSWER_KEY:
            continue
        if (
            item.source_kind == EvidenceSourceKind.TEACHER_ANNOTATION
            and item.endorsement_kind
            not in {
                EndorsementKind.CORRECT,
                EndorsementKind.FULL_SCORE,
                EndorsementKind.ALL_CORRECT,
                EndorsementKind.CORRECTED_ANSWER,
            }
        ):
            continue
        if item.rubric_support not in {RubricSupport.EXACT_MATCH, RubricSupport.COMPLETE}:
            continue
        if not any(answer.question_id == question_id for answer in item.answers):
            continue
        if item.source_kind == EvidenceSourceKind.TEACHER_ANNOTATION and not item.human_confirmed:
            continue
        usable.append(item)
    return usable


def _answer_signature(item: AnswerEvidenceDecision, question_id: str) -> tuple:
    answers = tuple(
        sorted(
            (
                answer.printed_part_label or "",
                answer.answer.strip(),
                tuple(sorted(answer.acceptable_alternatives)),
            )
            for answer in item.answers
            if answer.question_id == question_id
        )
    )
    rubric = tuple(
        sorted(
            (
                criterion.printed_part_label or "",
                criterion.description.strip(),
                criterion.points,
            )
            for criterion in item.rubric
            if criterion.question_id == question_id
        )
    )
    return item.rubric_support.value, answers, rubric


def _answers_from_evidence(
    item: AnswerEvidenceDecision,
    question_id: str,
    decision: QuestionReconstructionDecision,
    part_ids: dict[str, str],
) -> list[AnswerEntryDecision]:
    label_to_ref = _part_label_map(decision)
    answers: list[AnswerEntryDecision] = []
    for answer in item.answers:
        if answer.question_id != question_id:
            continue
        part_ref = None
        if answer.printed_part_label is not None:
            decision_ref = label_to_ref.get(answer.printed_part_label)
            if decision_ref is None:
                raise CaseValidationError(
                    f"evidence {item.ref} targets unknown printed part "
                    f"{answer.printed_part_label!r}"
                )
            part_ref = part_ids[decision_ref]
        answers.append(
            AnswerEntryDecision(
                part_ref=part_ref,
                answer=answer.answer,
                acceptable_alternatives=answer.acceptable_alternatives,
            )
        )
    return answers


def _rubric_from_evidence(
    item: AnswerEvidenceDecision,
    question_id: str,
    decision: QuestionReconstructionDecision,
    answers: list[AnswerEntryDecision],
    part_ids: dict[str, str],
) -> list[RubricCriterionDecision]:
    if item.rubric_support == RubricSupport.EXACT_MATCH:
        return [
            RubricCriterionDecision(
                ref=_stable_id("rubric", question_id, str(index)),
                part_ref=answer.part_ref,
                description="Answer exactly matches the approved reference answer.",
                points=(decision.points if len(answers) == 1 else None),
            )
            for index, answer in enumerate(answers, start=1)
        ]
    label_to_ref = _part_label_map(decision)
    rubric: list[RubricCriterionDecision] = []
    for criterion in item.rubric:
        if criterion.question_id != question_id:
            continue
        part_ref = None
        if criterion.printed_part_label is not None:
            decision_ref = label_to_ref.get(criterion.printed_part_label)
            if decision_ref is None:
                raise CaseValidationError(
                    f"evidence rubric {criterion.ref} targets unknown printed part"
                )
            part_ref = part_ids[decision_ref]
        rubric.append(
            RubricCriterionDecision(
                ref=_stable_id("rubric", question_id, criterion.ref),
                part_ref=part_ref,
                description=criterion.description,
                points=criterion.points,
            )
        )
    return rubric


def _remap_answers(
    answers: list[AnswerEntryDecision], part_ids: dict[str, str]
) -> list[AnswerEntryDecision]:
    return [
        AnswerEntryDecision(
            part_ref=part_ids.get(answer.part_ref) if answer.part_ref is not None else None,
            answer=answer.answer,
            acceptable_alternatives=answer.acceptable_alternatives,
        )
        for answer in answers
    ]


def _remap_rubric(
    rubric: list[RubricCriterionDecision],
    question_id: str,
    part_ids: dict[str, str],
) -> list[RubricCriterionDecision]:
    return [
        RubricCriterionDecision(
            ref=_stable_id("rubric", question_id, criterion.ref),
            part_ref=(
                part_ids.get(criterion.part_ref)
                if criterion.part_ref is not None
                else None
            ),
            description=criterion.description,
            points=criterion.points,
        )
        for criterion in rubric
    ]


def _part_label_map(decision: QuestionReconstructionDecision) -> dict[str, str]:
    result: dict[str, str] = {}
    for part in decision.subparts:
        if part.printed_label in result:
            raise CaseValidationError(
                f"question {decision.question_id} has duplicate printed subpart labels"
            )
        result[part.printed_label] = part.ref
    return result


def _answer_source(item: AnswerEvidenceDecision) -> AnswerSource:
    if item.source_kind == EvidenceSourceKind.OFFICIAL_ANSWER:
        return AnswerSource.OFFICIAL_ANSWER
    if item.endorsement_kind == EndorsementKind.CORRECTED_ANSWER:
        return AnswerSource.TEACHER_CORRECTION
    return AnswerSource.TEACHER_ENDORSED_SUBMISSION


def _render_crop(source_path: Path, bbox, *, jpeg_quality: int) -> tuple[bytes, ImageSize]:
    try:
        with Image.open(source_path) as source:
            image = source.convert("RGB").crop(
                (bbox.left, bbox.top, bbox.right, bbox.bottom)
            )
    except (OSError, UnidentifiedImageError) as exc:
        raise CaseValidationError(f"cannot crop solver input from {source_path}: {exc}") from exc
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
                raise CaseValidationError(f"invalid solver crop image at {path}")
    except (OSError, UnidentifiedImageError) as exc:
        raise CaseValidationError(f"cannot decode solver crop {path}: {exc}") from exc


def _active_document_graph(
    case_dir: Path, state: PipelineState
) -> tuple[Path, DocumentGraph]:
    completion = next(
        (item for item in state.completed_stages if item.stage == PipelineStage.MAPPED),
        None,
    )
    if completion is None:
        raise InvalidTransitionError("phase 4 requires an active document graph")
    references = [
        reference
        for reference in completion.artifacts
        if reference.schema_id == "document_graph.schema.json"
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
        raise InvalidTransitionError("phase 4 requires an active page manifest")
    references = [
        reference
        for reference in completion.artifacts
        if reference.schema_id == "page_manifest.schema.json"
    ]
    if len(references) != 1:
        raise CaseValidationError("pages_ready must have exactly one active page manifest")
    path = case_dir / references[0].relative_path
    return path, read_page_manifest(path)


def _read_decision_file(path: Path, model_type, label: str):
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise CaseValidationError(f"cannot read {label} file {path}: {exc}") from exc
    return validate_json(model_type, raw)


def _box_contains(outer, inner) -> bool:
    return (
        inner.left >= outer.left
        and inner.top >= outer.top
        and inner.right <= outer.right
        and inner.bottom <= outer.bottom
    )


def _stable_id(kind: str, *parts: str) -> str:
    digest = digest_value({"kind": kind, "parts": list(parts)})
    return f"{kind}-{digest[:20]}"


def _unique_strings(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
