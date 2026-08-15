"""Case ingestion, state transitions, versioned artifacts, and recovery."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import logging
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TypeVar
from uuid import uuid4

from pydantic import BaseModel

from student_analyze import __version__
from student_analyze.assets import artifact_digest, discover_source_assets, verify_source_assets
from student_analyze.atomic import atomic_write_model, serialize_model
from student_analyze.config import AppConfig
from student_analyze.errors import (
    CaseValidationError,
    ConfigurationError,
    InvalidTransitionError,
    ReviewRequiredError,
    SourceIntegrityError,
)
from student_analyze.fingerprint import (
    compute_input_fingerprint,
    compute_stage_fingerprint,
    digest_value,
)
from student_analyze.models import (
    SCHEMA_VERSION,
    STAGE_ORDER,
    ArtifactReference,
    CaseManifest,
    ImplementationVersions,
    PipelineStage,
    PipelineState,
    StageCompletion,
    StageRunRecord,
)
from student_analyze.validation import validate_json, validate_payload


LOGGER = logging.getLogger(__name__)
PayloadT = TypeVar("PayloadT", bound=BaseModel)
InterruptHook = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class IngestResult:
    case_dir: Path
    manifest: CaseManifest
    state: PipelineState
    reused: bool


@dataclass(frozen=True, slots=True)
class StageResult:
    state: PipelineState
    artifacts: tuple[ArtifactReference, ...]
    reused: bool


def build_stage_fingerprint(
    *,
    stage: PipelineStage,
    model_type: type[BaseModel],
    schema_id: str,
    config: Mapping[str, Any],
    versions: ImplementationVersions,
    inputs: Sequence[Mapping[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """Build the exact fingerprint configuration used by stage commits."""

    fingerprint_config = {
        **config,
        "artifact_schema_id": schema_id,
        "artifact_schema_sha256": digest_value(
            model_type.model_json_schema(mode="validation")
        ),
    }
    fingerprint = compute_stage_fingerprint(
        stage,
        inputs=inputs,
        config=fingerprint_config,
        versions=versions,
    )
    return fingerprint, fingerprint_config


def ingest_case(
    source_root: Path,
    config: AppConfig,
    *,
    force: bool = False,
    interrupt_hook: InterruptHook | None = None,
) -> IngestResult:
    started_at = _now()
    resolved_root = source_root.resolve(strict=True)
    resolved_cases_dir = config.cases_dir.resolve(strict=False)
    if resolved_cases_dir == resolved_root or resolved_root in resolved_cases_dir.parents:
        raise ConfigurationError(
            f"cases_dir must not be inside the read-only source root: {resolved_cases_dir}"
        )
    source_assets = discover_source_assets(resolved_root, config.source_extensions)
    versions = ImplementationVersions(
        code=__version__,
        base_schema=SCHEMA_VERSION,
        config=config.config_version,
    )
    config_fingerprint = digest_value(config.fingerprint_payload())
    input_fingerprint = compute_input_fingerprint(str(resolved_root), source_assets)
    case_id = f"case-{input_fingerprint[:20]}"
    manifest = CaseManifest(
        case_id=case_id,
        source_root=str(resolved_root),
        input_fingerprint=input_fingerprint,
        config_fingerprint=config_fingerprint,
        source_assets=source_assets,
        versions=versions,
        created_at=started_at,
    )

    config.cases_dir.mkdir(parents=True, exist_ok=True)
    _reject_changed_known_source(config.cases_dir, manifest)
    case_dir = config.cases_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = case_dir / "case_manifest.json"
    state_path = case_dir / "pipeline_state.json"

    if manifest_path.exists():
        persisted_manifest = _read_model(manifest_path, CaseManifest)
        _require_same_manifest_identity(persisted_manifest, manifest)
        manifest = persisted_manifest
    else:
        atomic_write_model(manifest_path, manifest, model_type=CaseManifest)
        _interrupt(interrupt_hook, "after_artifact_commit")

    verify_source_assets(manifest.source_assets)
    manifest_digest, manifest_size = artifact_digest(manifest_path)
    stage_fingerprint = compute_stage_fingerprint(
        PipelineStage.INGESTED,
        inputs=[
            {
                "input_fingerprint": manifest.input_fingerprint,
                "manifest_sha256": manifest_digest,
            }
        ],
        config=config.fingerprint_payload(),
        versions=manifest.versions,
    )

    if state_path.exists():
        state = _read_model(state_path, PipelineState)
        if state.case_id != case_id:
            raise CaseValidationError(
                f"pipeline state case_id {state.case_id} does not match {case_id}"
            )
        verify_case(case_dir)
        ingest_completion = state.completed_stages[0]
        if not force and ingest_completion.stage_fingerprint == stage_fingerprint:
            LOGGER.info("reusing complete ingestion for %s", case_id)
            return IngestResult(case_dir, manifest, state, reused=True)
        if not force:
            raise InvalidTransitionError(
                "ingested is already complete with a different configuration or "
                "implementation fingerprint; use --force to append a preserved run"
            )
        if (
            state.current_stage != PipelineStage.INGESTED
            and ingest_completion.stage_fingerprint != stage_fingerprint
        ):
            raise InvalidTransitionError(
                "cannot force a changed ingestion below the current stage; "
                "downstream invalidation is not implemented"
            )
    else:
        state = PipelineState(case_id=case_id, updated_at=started_at)

    run_id = _new_run_id()
    artifact_version = 1 + sum(
        run.stage == PipelineStage.INGESTED for run in state.run_history
    )
    reference = ArtifactReference(
        artifact_id=_artifact_id(
            PipelineStage.INGESTED, "case_manifest.json", manifest_digest
        ),
        stage=PipelineStage.INGESTED,
        run_id=run_id,
        relative_path="case_manifest.json",
        sha256=manifest_digest,
        size_bytes=manifest_size,
        schema_id="case_manifest.schema.json",
        schema_version=manifest.schema_version,
        stage_fingerprint=stage_fingerprint,
        artifact_version=artifact_version,
        created_at=started_at,
    )
    completed_at = _now()
    run = StageRunRecord(
        run_id=run_id,
        stage=PipelineStage.INGESTED,
        stage_fingerprint=stage_fingerprint,
        config_fingerprint=config_fingerprint,
        versions=versions,
        forced=force,
        started_at=started_at,
        completed_at=completed_at,
        artifacts=[reference],
    )
    completion = StageCompletion(
        stage=PipelineStage.INGESTED,
        active_run_id=run_id,
        stage_fingerprint=stage_fingerprint,
        completed_at=completed_at,
        artifacts=[reference],
    )
    completions = list(state.completed_stages)
    if completions:
        completions[0] = completion
    else:
        completions.append(completion)
    new_state = state.model_copy(
        update={
            "current_stage": state.current_stage or PipelineStage.INGESTED,
            "revision": state.revision + 1,
            "completed_stages": completions,
            "run_history": [*state.run_history, run],
            "updated_at": completed_at,
        }
    )
    new_state = validate_payload(PipelineState, new_state.model_dump(mode="json"))

    verify_source_assets(manifest.source_assets)
    _interrupt(interrupt_hook, "before_state_commit")
    atomic_write_model(state_path, new_state, model_type=PipelineState)
    verify_case(case_dir)
    return IngestResult(case_dir, manifest, new_state, reused=False)


def commit_stage_artifact(
    case_dir: Path,
    *,
    stage: PipelineStage,
    artifact_name: str,
    payload: Any,
    model_type: type[PayloadT],
    schema_id: str,
    config: Mapping[str, Any],
    versions: ImplementationVersions,
    inputs: Sequence[Mapping[str, Any]],
    force: bool = False,
    human_confirmed: bool = False,
    interrupt_hook: InterruptHook | None = None,
) -> StageResult:
    """Validate and commit one future-stage JSON artifact.

    Future phases call this primitive with their own newly introduced Pydantic
    model. Phase 1 deliberately does not define those business models.
    """

    case_dir = case_dir.resolve(strict=True)
    manifest, state = verify_case(case_dir)
    if stage == PipelineStage.INGESTED:
        raise InvalidTransitionError("ingested is completed only by ingest_case")
    _validate_transition(state, stage, force=force)
    if stage == PipelineStage.PAGES_READY:
        from student_analyze.page_models import PageManifest

        if model_type is not PageManifest or schema_id != "page_manifest.schema.json":
            raise CaseValidationError(
                "pages_ready can only be completed with the production PageManifest contract"
            )
    if stage == PipelineStage.MAPPED:
        from student_analyze.document_models import DocumentGraph

        if model_type is not DocumentGraph or schema_id != "document_graph.schema.json":
            raise CaseValidationError(
                "mapped can only be completed with the production DocumentGraph contract"
            )
    if stage == PipelineStage.MASTER_READY:
        from student_analyze.exam_master_models import ExamMaster

        if model_type is not ExamMaster or schema_id != "exam_master.schema.json":
            raise CaseValidationError(
                "master_ready can only be completed with the production ExamMaster contract"
            )
    if stage == PipelineStage.SUBMISSION_READY:
        from student_analyze.submission_models import Submission

        if model_type is not Submission or schema_id != "submission.schema.json":
            raise CaseValidationError(
                "submission_ready can only be completed with the production Submission contract"
            )
    if stage in {PipelineStage.GRADED, PipelineStage.REVIEWED}:
        from student_analyze.grading_models import Grading, GradingPhase

        if model_type is not Grading or schema_id != "grading.schema.json":
            raise CaseValidationError(
                f"{stage.value} can only be completed with the production Grading contract"
            )
    if stage == PipelineStage.REPORTED:
        from student_analyze.learning_models import ReportManifest

        if model_type is not ReportManifest or schema_id != "report_manifest.schema.json":
            raise CaseValidationError(
                "reported can only be completed with the production ReportManifest contract"
            )
    stage_fingerprint, fingerprint_config = build_stage_fingerprint(
        stage=stage,
        model_type=model_type,
        schema_id=schema_id,
        config=config,
        versions=versions,
        inputs=inputs,
    )

    existing = next(
        (item for item in state.completed_stages if item.stage == stage),
        None,
    )
    if existing and existing.stage_fingerprint == stage_fingerprint and not force:
        _verify_artifact_references(case_dir, existing.artifacts)
        return StageResult(state, tuple(existing.artifacts), reused=True)
    if existing and not force:
        raise InvalidTransitionError(
            f"{stage.value} is already complete with a different fingerprint; "
            "use force to create a preserved artifact version"
        )

    validated = validate_payload(model_type, payload)
    if stage == PipelineStage.PAGES_READY:
        if validated.requires_review:
            raise ReviewRequiredError(
                "page manifest requires review and cannot complete pages_ready"
            )
        from student_analyze.page_verification import verify_page_outputs

        verify_page_outputs(case_dir, manifest, validated)
    if stage == PipelineStage.MASTER_READY and validated.requires_review:
        raise ReviewRequiredError(
            "Exam Master requires review and cannot complete master_ready"
        )
    if stage == PipelineStage.GRADED and validated.phase != GradingPhase.GRADED:
        raise CaseValidationError("graded requires a graded-phase Grading artifact")
    if stage == PipelineStage.REVIEWED:
        if validated.phase != GradingPhase.REVIEWED:
            raise CaseValidationError("reviewed requires a reviewed-phase Grading artifact")
        if validated.requires_review:
            raise ReviewRequiredError(
                "reviewed grading still contains unresolved review items"
            )
    if stage == PipelineStage.REPORTED:
        if validated.requires_review:
            raise ReviewRequiredError("report manifest still contains unresolved review items")
        if validated.stage_fingerprint != stage_fingerprint:
            raise CaseValidationError(
                "report manifest stage fingerprint differs from the computed fingerprint"
            )

    started_at = _now()
    run_id = _new_run_id()
    artifact_version = 1 + sum(run.stage == stage for run in state.run_history)
    safe_name = _validate_artifact_name(artifact_name)
    if force:
        relative_path = Path("artifacts", stage.value, stage_fingerprint, "runs", run_id, safe_name)
    else:
        relative_path = Path("artifacts", stage.value, stage_fingerprint, safe_name)
    artifact_path = case_dir / relative_path

    if artifact_path.exists():
        persisted = _read_model(artifact_path, model_type)
        if serialize_model(persisted) != serialize_model(validated):
            raise CaseValidationError(
                f"fingerprint collision or incomplete artifact at {artifact_path}"
            )
    else:
        atomic_write_model(artifact_path, validated, model_type=model_type)
    _interrupt(interrupt_hook, "after_artifact_commit")

    artifact_sha256, artifact_size = artifact_digest(artifact_path)
    completed_at = _now()
    reference = ArtifactReference(
        artifact_id=_artifact_id(stage, relative_path.as_posix(), artifact_sha256),
        stage=stage,
        run_id=run_id,
        relative_path=relative_path.as_posix(),
        sha256=artifact_sha256,
        size_bytes=artifact_size,
        schema_id=schema_id,
        schema_version=str(validated.model_dump(mode="json").get("schema_version", SCHEMA_VERSION)),
        stage_fingerprint=stage_fingerprint,
        artifact_version=artifact_version,
        created_at=started_at,
        human_confirmed=human_confirmed,
    )
    run = StageRunRecord(
        run_id=run_id,
        stage=stage,
        stage_fingerprint=stage_fingerprint,
        config_fingerprint=digest_value(fingerprint_config),
        versions=versions,
        forced=force,
        started_at=started_at,
        completed_at=completed_at,
        artifacts=[reference],
    )
    completion = StageCompletion(
        stage=stage,
        active_run_id=run_id,
        stage_fingerprint=stage_fingerprint,
        completed_at=completed_at,
        artifacts=[reference],
    )
    completions = list(state.completed_stages)
    stage_index = STAGE_ORDER.index(stage)
    if stage_index < len(completions):
        completions[stage_index] = completion
    else:
        completions.append(completion)
    new_state = PipelineState(
        case_id=state.case_id,
        current_stage=max(
            (item.stage for item in completions),
            key=STAGE_ORDER.index,
        ),
        revision=state.revision + 1,
        completed_stages=completions,
        run_history=[*state.run_history, run],
        updated_at=completed_at,
    )

    verify_source_assets(manifest.source_assets)
    _interrupt(interrupt_hook, "before_state_commit")
    atomic_write_model(case_dir / "pipeline_state.json", new_state, model_type=PipelineState)
    verify_case(case_dir)
    return StageResult(new_state, (reference,), reused=False)


def verify_case(case_dir: Path) -> tuple[CaseManifest, PipelineState]:
    manifest = _read_model(case_dir / "case_manifest.json", CaseManifest)
    state = _read_model(case_dir / "pipeline_state.json", PipelineState)
    if manifest.case_id != state.case_id:
        raise CaseValidationError("manifest and state use different case_id values")
    expected_input_fingerprint = compute_input_fingerprint(
        manifest.source_root, manifest.source_assets
    )
    if expected_input_fingerprint != manifest.input_fingerprint:
        raise CaseValidationError("manifest input_fingerprint does not match source_assets")
    verify_source_assets(manifest.source_assets)
    for run in state.run_history:
        _verify_artifact_references(case_dir, run.artifacts)
    active_page_manifest = None
    active_page_manifest_path = None
    page_completion = next(
        (
            completion
            for completion in state.completed_stages
            if completion.stage == PipelineStage.PAGES_READY
        ),
        None,
    )
    if page_completion is not None:
        page_references = [
            reference
            for reference in page_completion.artifacts
            if reference.schema_id == "page_manifest.schema.json"
        ]
        if len(page_references) != 1:
            raise CaseValidationError(
                "pages_ready must have exactly one active page manifest"
            )
        from student_analyze.page_verification import (
            read_page_manifest,
            verify_page_outputs,
        )

        reference = page_references[0]
        active_page_manifest_path = case_dir / reference.relative_path
        active_page_manifest = read_page_manifest(active_page_manifest_path)
        if active_page_manifest.stage_fingerprint != page_completion.stage_fingerprint:
            raise CaseValidationError(
                "page manifest stage fingerprint differs from pipeline state"
            )
        verify_page_outputs(case_dir, manifest, active_page_manifest)

    active_document_graph = None
    mapping_completion = next(
        (
            completion
            for completion in state.completed_stages
            if completion.stage == PipelineStage.MAPPED
        ),
        None,
    )
    if mapping_completion is not None:
        if active_page_manifest is None or active_page_manifest_path is None:
            raise CaseValidationError("mapped requires an active page manifest")
        graph_references = [
            reference
            for reference in mapping_completion.artifacts
            if reference.schema_id == "document_graph.schema.json"
        ]
        if len(graph_references) != 1:
            raise CaseValidationError(
                "mapped must have exactly one active document graph"
            )
        from student_analyze.document_mapper import (
            read_document_graph,
            verify_document_graph,
        )

        graph_reference = graph_references[0]
        graph = read_document_graph(case_dir / graph_reference.relative_path)
        active_document_graph = graph
        if graph.stage_fingerprint != mapping_completion.stage_fingerprint:
            raise CaseValidationError(
                "document graph stage fingerprint differs from pipeline state"
            )
        page_manifest_sha256, _ = artifact_digest(active_page_manifest_path)
        verify_document_graph(
            graph,
            active_page_manifest,
            page_manifest_sha256=page_manifest_sha256,
        )

    active_exam_master = None
    active_exam_master_path = None
    master_completion = next(
        (
            completion
            for completion in state.completed_stages
            if completion.stage == PipelineStage.MASTER_READY
        ),
        None,
    )
    if master_completion is not None:
        if active_document_graph is None or mapping_completion is None:
            raise CaseValidationError("master_ready requires an active document graph")
        master_references = [
            reference
            for reference in master_completion.artifacts
            if reference.schema_id == "exam_master.schema.json"
        ]
        if len(master_references) != 1:
            raise CaseValidationError(
                "master_ready must have exactly one active Exam Master"
            )
        from student_analyze.exam_master import read_exam_master, verify_exam_master

        master_reference = master_references[0]
        active_exam_master_path = case_dir / master_reference.relative_path
        master = read_exam_master(active_exam_master_path)
        active_exam_master = master
        if master.stage_fingerprint != master_completion.stage_fingerprint:
            raise CaseValidationError(
                "Exam Master stage fingerprint differs from pipeline state"
            )
        graph_reference = next(
            reference
            for reference in mapping_completion.artifacts
            if reference.schema_id == "document_graph.schema.json"
        )
        graph_sha256, _ = artifact_digest(case_dir / graph_reference.relative_path)
        verify_exam_master(
            case_dir,
            master,
            active_document_graph,
            graph_sha256=graph_sha256,
        )

    active_submission = None
    active_submission_path = None
    submission_completion = next(
        (
            completion
            for completion in state.completed_stages
            if completion.stage == PipelineStage.SUBMISSION_READY
        ),
        None,
    )
    if submission_completion is not None:
        if (
            active_document_graph is None
            or active_page_manifest is None
            or active_page_manifest_path is None
            or active_exam_master is None
            or active_exam_master_path is None
            or mapping_completion is None
        ):
            raise CaseValidationError(
                "submission_ready requires page, graph, and Exam Master artifacts"
            )
        submission_references = [
            reference
            for reference in submission_completion.artifacts
            if reference.schema_id == "submission.schema.json"
        ]
        if len(submission_references) != 1:
            raise CaseValidationError(
                "submission_ready must have exactly one active Submission"
            )
        from student_analyze.submission import read_submission, verify_submission

        submission_reference = submission_references[0]
        active_submission_path = case_dir / submission_reference.relative_path
        submission = read_submission(active_submission_path)
        active_submission = submission
        if submission.stage_fingerprint != submission_completion.stage_fingerprint:
            raise CaseValidationError(
                "Submission stage fingerprint differs from pipeline state"
            )
        graph_reference = next(
            reference
            for reference in mapping_completion.artifacts
            if reference.schema_id == "document_graph.schema.json"
        )
        graph_sha256, _ = artifact_digest(case_dir / graph_reference.relative_path)
        page_manifest_sha256, _ = artifact_digest(active_page_manifest_path)
        exam_master_sha256, _ = artifact_digest(active_exam_master_path)
        verify_submission(
            case_dir,
            submission,
            active_document_graph,
            active_page_manifest,
            active_exam_master,
            graph_sha256=graph_sha256,
            page_manifest_sha256=page_manifest_sha256,
            exam_master_sha256=exam_master_sha256,
        )

    active_graded = None
    active_graded_path = None
    graded_completion = next(
        (
            completion
            for completion in state.completed_stages
            if completion.stage == PipelineStage.GRADED
        ),
        None,
    )
    if graded_completion is not None:
        if (
            active_exam_master is None
            or active_exam_master_path is None
            or active_submission is None
            or active_submission_path is None
        ):
            raise CaseValidationError(
                "graded requires active Exam Master and Submission artifacts"
            )
        grading_references = [
            reference
            for reference in graded_completion.artifacts
            if reference.schema_id == "grading.schema.json"
        ]
        if len(grading_references) != 1:
            raise CaseValidationError("graded must have exactly one active Grading")
        from student_analyze.grading import read_grading, verify_grading

        grading_reference = grading_references[0]
        active_graded_path = case_dir / grading_reference.relative_path
        active_graded = read_grading(active_graded_path)
        if active_graded.stage_fingerprint != graded_completion.stage_fingerprint:
            raise CaseValidationError(
                "Grading stage fingerprint differs from pipeline state"
            )
        exam_master_sha256, _ = artifact_digest(active_exam_master_path)
        submission_sha256, _ = artifact_digest(active_submission_path)
        verify_grading(
            active_graded,
            active_exam_master,
            active_submission,
            exam_master_sha256=exam_master_sha256,
            submission_sha256=submission_sha256,
        )

    active_reviewed = None
    active_reviewed_path = None
    reviewed_completion = next(
        (
            completion
            for completion in state.completed_stages
            if completion.stage == PipelineStage.REVIEWED
        ),
        None,
    )
    if reviewed_completion is not None:
        if (
            active_exam_master is None
            or active_exam_master_path is None
            or active_submission is None
            or active_submission_path is None
            or active_graded is None
            or active_graded_path is None
        ):
            raise CaseValidationError(
                "reviewed requires active Exam Master, Submission, and Grading artifacts"
            )
        reviewed_references = [
            reference
            for reference in reviewed_completion.artifacts
            if reference.schema_id == "grading.schema.json"
        ]
        if len(reviewed_references) != 1:
            raise CaseValidationError("reviewed must have exactly one active Grading")
        from student_analyze.grading import read_grading, verify_grading

        reviewed_reference = reviewed_references[0]
        active_reviewed_path = case_dir / reviewed_reference.relative_path
        reviewed = read_grading(active_reviewed_path)
        active_reviewed = reviewed
        if reviewed.stage_fingerprint != reviewed_completion.stage_fingerprint:
            raise CaseValidationError(
                "Reviewed grading stage fingerprint differs from pipeline state"
            )
        exam_master_sha256, _ = artifact_digest(active_exam_master_path)
        submission_sha256, _ = artifact_digest(active_submission_path)
        active_graded_sha256, _ = artifact_digest(active_graded_path)
        verify_grading(
            reviewed,
            active_exam_master,
            active_submission,
            exam_master_sha256=exam_master_sha256,
            submission_sha256=submission_sha256,
            source_grading=active_graded,
            source_grading_sha256=active_graded_sha256,
        )

    reported_completion = next(
        (
            completion
            for completion in state.completed_stages
            if completion.stage == PipelineStage.REPORTED
        ),
        None,
    )
    if reported_completion is not None:
        if active_reviewed is None or active_reviewed_path is None:
            raise CaseValidationError("reported requires an active reviewed grading artifact")
        report_references = [
            reference
            for reference in reported_completion.artifacts
            if reference.schema_id == "report_manifest.schema.json"
        ]
        if len(report_references) != 1:
            raise CaseValidationError("reported must have exactly one active report manifest")
        from student_analyze.reporting import read_report_manifest, verify_report_assets

        report_reference = report_references[0]
        report = read_report_manifest(case_dir / report_reference.relative_path)
        if report.stage_fingerprint != reported_completion.stage_fingerprint:
            raise CaseValidationError(
                "report manifest stage fingerprint differs from pipeline state"
            )
        reviewed_sha256, _ = artifact_digest(active_reviewed_path)
        if report.analysis.input_manifest.reviewed_grading_sha256 != reviewed_sha256:
            raise CaseValidationError("report references a different reviewed grading artifact")
        verify_report_assets(case_dir, report)
    return manifest, state


def _read_model(path: Path, model_type: type[PayloadT]) -> PayloadT:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise CaseValidationError(f"cannot read required case file {path}: {exc}") from exc
    return validate_json(model_type, raw)


def _verify_artifact_references(
    case_dir: Path, references: Sequence[ArtifactReference]
) -> None:
    resolved_case = case_dir.resolve(strict=True)
    for reference in references:
        path = (resolved_case / reference.relative_path).resolve(strict=False)
        if path != resolved_case and resolved_case not in path.parents:
            raise CaseValidationError(
                f"artifact escapes case directory: {reference.relative_path}"
            )
        if not path.is_file():
            raise CaseValidationError(f"referenced artifact is missing: {path}")
        digest, size = artifact_digest(path)
        if digest != reference.sha256 or size != reference.size_bytes:
            raise CaseValidationError(f"referenced artifact changed: {path}")


def _validate_transition(
    state: PipelineState, target: PipelineStage, *, force: bool
) -> None:
    target_index = STAGE_ORDER.index(target)
    completed_count = len(state.completed_stages)
    if target_index == completed_count:
        return
    if target_index == completed_count - 1:
        return
    if target_index < completed_count:
        suffix = " (use force only for the current stage)" if not force else ""
        raise InvalidTransitionError(
            f"cannot rerun {target.value} below current stage {state.current_stage.value}{suffix}"
        )
    required = STAGE_ORDER[target_index - 1]
    raise InvalidTransitionError(
        f"cannot enter {target.value}; required previous stage is {required.value}"
    )


def _reject_changed_known_source(cases_dir: Path, candidate: CaseManifest) -> None:
    for manifest_path in cases_dir.glob("case-*/case_manifest.json"):
        known = _read_model(manifest_path, CaseManifest)
        if known.source_root != candidate.source_root:
            continue
        if known.input_fingerprint != candidate.input_fingerprint:
            raise SourceIntegrityError(
                "source root was previously ingested but its file list or content changed: "
                f"{candidate.source_root}; existing case={known.case_id}"
            )


def _require_same_manifest_identity(
    persisted: CaseManifest, candidate: CaseManifest
) -> None:
    if persisted.case_id != candidate.case_id or persisted.input_fingerprint != candidate.input_fingerprint:
        raise CaseValidationError("existing case manifest conflicts with current input")


def _validate_artifact_name(name: str) -> str:
    candidate = Path(name)
    if candidate.name != name or name in {"", ".", ".."}:
        raise CaseValidationError("artifact_name must be one plain filename")
    return name


def _new_run_id() -> str:
    timestamp = _now().strftime("%Y%m%dt%H%M%S%fZ").lower()
    return f"run-{timestamp}-{uuid4().hex[:8]}"


def _artifact_id(
    stage: PipelineStage, relative_path: str, artifact_sha256: str
) -> str:
    identity = digest_value(
        {
            "stage": stage.value,
            "relative_path": relative_path,
            "sha256": artifact_sha256,
        }
    )
    return f"artifact-{identity[:20]}"


def _now() -> datetime:
    return datetime.now(UTC)


def _interrupt(hook: InterruptHook | None, event: str) -> None:
    if hook is not None:
        hook(event)
