"""Foundational production contracts for phase 1.

These Pydantic models are the single source for the committed JSON Schemas.
Business models for pages, questions, submissions, grading, and databases are
intentionally deferred to the phases that first use them.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


SCHEMA_VERSION = "1.0.0"
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
StableId = Annotated[str, Field(pattern=r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PipelineStage(str, Enum):
    INGESTED = "ingested"
    PAGES_READY = "pages_ready"
    MAPPED = "mapped"
    MASTER_READY = "master_ready"
    SUBMISSION_READY = "submission_ready"
    GRADED = "graded"
    REVIEWED = "reviewed"
    REPORTED = "reported"


STAGE_ORDER: tuple[PipelineStage, ...] = tuple(PipelineStage)


class ImplementationVersions(StrictModel):
    code: str = Field(min_length=1)
    base_schema: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    config: str = Field(min_length=1)


class PrivacyFlags(StrictModel):
    contains_personal_data: bool = True
    local_only: bool = True


class SourceAsset(StrictModel):
    asset_id: StableId
    relative_path: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    sha256: Sha256
    size_bytes: int = Field(ge=0)
    modified_time_ns: int = Field(ge=0)
    media_type: str | None = None

    @model_validator(mode="after")
    def validate_relative_path(self) -> SourceAsset:
        normalized = self.relative_path.replace("\\", "/")
        parts = normalized.split("/")
        if normalized.startswith("/") or any(part in {"", ".", ".."} for part in parts):
            raise ValueError("relative_path must be a normalized path below source_root")
        if normalized != self.relative_path:
            raise ValueError("relative_path must use forward slashes")
        return self


class ArtifactReference(StrictModel):
    artifact_id: StableId
    stage: PipelineStage
    run_id: StableId
    relative_path: str = Field(min_length=1)
    sha256: Sha256
    size_bytes: int = Field(ge=0)
    schema_id: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    stage_fingerprint: Sha256
    artifact_version: int = Field(ge=1)
    created_at: datetime
    human_confirmed: bool = False

    @model_validator(mode="after")
    def validate_relative_path(self) -> ArtifactReference:
        normalized = self.relative_path.replace("\\", "/")
        parts = normalized.split("/")
        if normalized.startswith("/") or any(part in {"", ".", ".."} for part in parts):
            raise ValueError("artifact relative_path must stay below the case directory")
        if normalized != self.relative_path:
            raise ValueError("artifact relative_path must use forward slashes")
        return self


class StageRunRecord(StrictModel):
    run_id: StableId
    stage: PipelineStage
    stage_fingerprint: Sha256
    config_fingerprint: Sha256
    versions: ImplementationVersions
    forced: bool = False
    started_at: datetime
    completed_at: datetime
    artifacts: list[ArtifactReference] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_artifacts(self) -> StageRunRecord:
        if self.completed_at < self.started_at:
            raise ValueError("completed_at cannot precede started_at")
        if any(artifact.stage != self.stage for artifact in self.artifacts):
            raise ValueError("every artifact must belong to the run stage")
        if any(artifact.run_id != self.run_id for artifact in self.artifacts):
            raise ValueError("every artifact must reference the containing run")
        if any(
            artifact.stage_fingerprint != self.stage_fingerprint
            for artifact in self.artifacts
        ):
            raise ValueError("every artifact must use the run stage fingerprint")
        return self


class StageCompletion(StrictModel):
    stage: PipelineStage
    active_run_id: StableId
    stage_fingerprint: Sha256
    completed_at: datetime
    artifacts: list[ArtifactReference] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_artifacts(self) -> StageCompletion:
        if any(artifact.stage != self.stage for artifact in self.artifacts):
            raise ValueError("every active artifact must belong to the completed stage")
        if any(artifact.run_id != self.active_run_id for artifact in self.artifacts):
            raise ValueError("active artifacts must reference active_run_id")
        if any(
            artifact.stage_fingerprint != self.stage_fingerprint
            for artifact in self.artifacts
        ):
            raise ValueError("active artifacts must use the completion fingerprint")
        return self


class PipelineState(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    case_id: StableId
    current_stage: PipelineStage | None = None
    revision: int = Field(default=0, ge=0)
    completed_stages: list[StageCompletion] = Field(default_factory=list)
    run_history: list[StageRunRecord] = Field(default_factory=list)
    updated_at: datetime

    @model_validator(mode="after")
    def validate_state_machine(self) -> PipelineState:
        actual = [completion.stage for completion in self.completed_stages]
        expected = list(STAGE_ORDER[: len(actual)])
        if actual != expected:
            raise ValueError("completed_stages must be a contiguous pipeline prefix")
        expected_current = actual[-1] if actual else None
        if self.current_stage != expected_current:
            raise ValueError("current_stage must equal the last completed stage")

        run_ids = [run.run_id for run in self.run_history]
        if len(run_ids) != len(set(run_ids)):
            raise ValueError("run_history contains duplicate run_id values")
        runs_by_id = {run.run_id: run for run in self.run_history}
        for completion in self.completed_stages:
            run = runs_by_id.get(completion.active_run_id)
            if run is None or run.stage != completion.stage:
                raise ValueError("each completed stage must reference a matching run")
            if run.stage_fingerprint != completion.stage_fingerprint:
                raise ValueError("completed stage fingerprint must match its active run")
            if run.artifacts != completion.artifacts:
                raise ValueError("completed stage artifacts must match its active run")
        if self.revision != len(self.run_history):
            raise ValueError("revision must equal the number of committed runs")
        return self


class CaseManifest(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    case_id: StableId
    student_profile_id: str | None = None
    source_root: str = Field(min_length=1)
    input_fingerprint: Sha256
    config_fingerprint: Sha256
    source_assets: list[SourceAsset] = Field(min_length=1)
    versions: ImplementationVersions
    privacy: PrivacyFlags = Field(default_factory=PrivacyFlags)
    created_at: datetime

    @model_validator(mode="after")
    def validate_identity(self) -> CaseManifest:
        expected_case_id = f"case-{self.input_fingerprint[:20]}"
        if self.case_id != expected_case_id:
            raise ValueError("case_id must be derived from input_fingerprint")
        asset_ids = [asset.asset_id for asset in self.source_assets]
        relative_paths = [asset.relative_path for asset in self.source_assets]
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("source_assets contains duplicate asset_id values")
        if len(relative_paths) != len(set(relative_paths)):
            raise ValueError("source_assets contains duplicate relative_path values")
        if relative_paths != sorted(relative_paths, key=str.casefold):
            raise ValueError("source_assets must be sorted by relative_path")
        return self
