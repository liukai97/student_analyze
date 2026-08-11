"""Generate deterministic JSON Schemas from the phase 1 Pydantic models."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from student_analyze.models import (
    ArtifactReference,
    CaseManifest,
    PipelineState,
    SourceAsset,
)
from student_analyze.document_models import DocumentGraph, DocumentGraphDecisionSet
from student_analyze.exam_master_models import (
    AnswerEvidenceDecisionSet,
    ExamMaster,
    ExamMasterDecisionSet,
    ExamReviewDecisionSet,
    QuestionReconstructionDecisionSet,
    SolverInputDecisionSet,
    SolverInputManifest,
)
from student_analyze.page_models import PageDecisionSet, PageManifest
from student_analyze.submission_models import (
    Submission,
    SubmissionInputManifest,
    SubmissionMappingDecisionSet,
    SubmissionStructureManifest,
    SubmissionTranscriptionDecisionSet,
)


SCHEMA_MODELS: dict[str, type[BaseModel]] = {
    "artifact_reference.schema.json": ArtifactReference,
    "case_manifest.schema.json": CaseManifest,
    "document_graph.schema.json": DocumentGraph,
    "document_graph_decision.schema.json": DocumentGraphDecisionSet,
    "answer_evidence_decision.schema.json": AnswerEvidenceDecisionSet,
    "exam_master.schema.json": ExamMaster,
    "exam_master_decision.schema.json": ExamMasterDecisionSet,
    "exam_review_decision.schema.json": ExamReviewDecisionSet,
    "question_reconstruction_decision.schema.json": QuestionReconstructionDecisionSet,
    "pipeline_state.schema.json": PipelineState,
    "page_decision.schema.json": PageDecisionSet,
    "page_manifest.schema.json": PageManifest,
    "source_asset.schema.json": SourceAsset,
    "solver_input_decision.schema.json": SolverInputDecisionSet,
    "solver_input_manifest.schema.json": SolverInputManifest,
    "submission.schema.json": Submission,
    "submission_input_manifest.schema.json": SubmissionInputManifest,
    "submission_mapping_decision.schema.json": SubmissionMappingDecisionSet,
    "submission_structure_manifest.schema.json": SubmissionStructureManifest,
    "submission_transcription_decision.schema.json": SubmissionTranscriptionDecisionSet,
}
SCHEMA_BASE_ID = "https://local.student-analyze/schemas/"


def schema_document(filename: str, model_type: type[BaseModel]) -> dict[str, object]:
    schema = model_type.model_json_schema(mode="validation")
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = f"{SCHEMA_BASE_ID}{filename}"
    return schema


def schema_bytes(filename: str, model_type: type[BaseModel]) -> bytes:
    content = json.dumps(
        schema_document(filename, model_type),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    return (content + "\n").encode("utf-8")


def write_schemas(output_dir: Path, *, check: bool = False) -> list[Path]:
    changed: list[Path] = []
    for filename, model_type in SCHEMA_MODELS.items():
        path = output_dir / filename
        expected = schema_bytes(filename, model_type)
        if check:
            if not path.exists() or path.read_bytes() != expected:
                changed.append(path)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.read_bytes() != expected:
            path.write_bytes(expected)
            changed.append(path)
    return changed
