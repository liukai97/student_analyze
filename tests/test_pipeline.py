from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict

from student_analyze.config import AppConfig
from student_analyze.errors import CaseValidationError, InvalidTransitionError
from student_analyze.models import PipelineStage
from student_analyze.pipeline import commit_stage_artifact, ingest_case, verify_case


class PageStub(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"] = "1.0.0"
    pages: list[str]


def _ingested_case(tmp_path: Path):
    source = tmp_path / "raw"
    source.mkdir()
    (source / "exam.jpg").write_bytes(b"exam")
    config = AppConfig(cases_dir=tmp_path / "cases")
    return ingest_case(source, config), config


def _commit_pages(result, config, **overrides):
    arguments = {
        "case_dir": result.case_dir,
        "stage": PipelineStage.PAGES_READY,
        "artifact_name": "page_stub.json",
        "payload": {"schema_version": "1.0.0", "pages": ["page-1"]},
        "model_type": PageStub,
        "schema_id": "page_stub.schema.json",
        "config": config.fingerprint_payload(),
        "versions": result.manifest.versions,
        "inputs": [{"input_fingerprint": result.manifest.input_fingerprint}],
    }
    arguments.update(overrides)
    return commit_stage_artifact(**arguments)


def test_invalid_payload_does_not_create_artifact_or_advance(tmp_path: Path) -> None:
    result, config = _ingested_case(tmp_path)

    with pytest.raises(CaseValidationError, match="PageStub"):
        _commit_pages(result, config, payload={"schema_version": "1.0.0"})

    _, state = verify_case(result.case_dir)
    assert state.current_stage == PipelineStage.INGESTED
    assert not (result.case_dir / "artifacts").exists()


def test_stage_cannot_skip_required_predecessor(tmp_path: Path) -> None:
    result, config = _ingested_case(tmp_path)

    with pytest.raises(InvalidTransitionError, match="required previous stage"):
        commit_stage_artifact(
            result.case_dir,
            stage=PipelineStage.MAPPED,
            artifact_name="mapping.json",
            payload={"schema_version": "1.0.0", "pages": []},
            model_type=PageStub,
            schema_id="mapping.schema.json",
            config=config.fingerprint_payload(),
            versions=result.manifest.versions,
            inputs=[],
        )


def test_interruption_between_artifact_and_state_is_recoverable(tmp_path: Path) -> None:
    result, config = _ingested_case(tmp_path)

    def interrupt(event: str) -> None:
        if event == "before_state_commit":
            raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        _commit_pages(result, config, interrupt_hook=interrupt)

    _, previous_state = verify_case(result.case_dir)
    assert previous_state.current_stage == PipelineStage.INGESTED
    orphaned_artifacts = list((result.case_dir / "artifacts").rglob("page_stub.json"))
    assert len(orphaned_artifacts) == 1

    recovered = _commit_pages(result, config)
    assert not recovered.reused
    assert recovered.state.current_stage == PipelineStage.PAGES_READY
    assert list((result.case_dir / "artifacts").rglob("page_stub.json")) == orphaned_artifacts


def test_same_fingerprint_reuses_complete_stage(tmp_path: Path) -> None:
    result, config = _ingested_case(tmp_path)
    first = _commit_pages(result, config)
    second = _commit_pages(result, config)

    assert not first.reused
    assert second.reused
    assert second.state.revision == first.state.revision
    assert len(second.state.run_history) == len(first.state.run_history)


def test_changed_fingerprint_requires_force(tmp_path: Path) -> None:
    result, config = _ingested_case(tmp_path)
    _commit_pages(result, config)

    with pytest.raises(InvalidTransitionError, match="different fingerprint"):
        _commit_pages(
            result,
            config,
            inputs=[{"input_fingerprint": "different"}],
        )


def test_tampered_artifact_blocks_verification_and_next_stage(tmp_path: Path) -> None:
    result, config = _ingested_case(tmp_path)
    pages = _commit_pages(result, config)
    artifact_path = result.case_dir / pages.artifacts[0].relative_path
    artifact_path.write_text("{}", encoding="utf-8")

    with pytest.raises(CaseValidationError, match="referenced artifact changed"):
        verify_case(result.case_dir)


def test_force_rerun_versions_artifact_and_preserves_previous_run(tmp_path: Path) -> None:
    result, config = _ingested_case(tmp_path)
    first = _commit_pages(result, config)
    forced = _commit_pages(
        result,
        config,
        force=True,
        payload={"schema_version": "1.0.0", "pages": ["page-1", "page-2"]},
    )

    assert len(forced.state.run_history) == len(first.state.run_history) + 1
    assert forced.artifacts[0].artifact_version == 2
    assert forced.artifacts[0].relative_path != first.artifacts[0].relative_path
    assert (result.case_dir / first.artifacts[0].relative_path).is_file()
    assert (result.case_dir / forced.artifacts[0].relative_path).is_file()


def test_changed_ingestion_cannot_be_forced_below_later_stage(tmp_path: Path) -> None:
    result, config = _ingested_case(tmp_path)
    _commit_pages(result, config)
    changed_config = AppConfig(
        config_version="2",
        cases_dir=config.cases_dir,
    )

    with pytest.raises(InvalidTransitionError, match="downstream invalidation"):
        ingest_case(
            Path(result.manifest.source_root),
            changed_config,
            force=True,
        )
