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


SCHEMA_MODELS: dict[str, type[BaseModel]] = {
    "artifact_reference.schema.json": ArtifactReference,
    "case_manifest.schema.json": CaseManifest,
    "pipeline_state.schema.json": PipelineState,
    "source_asset.schema.json": SourceAsset,
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
