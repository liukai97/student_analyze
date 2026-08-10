from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from PIL import Image
import pytest
from pydantic import BaseModel, ConfigDict

from student_analyze.assets import artifact_digest
from student_analyze.config import AppConfig
from student_analyze.errors import (
    CaseValidationError,
    InvalidTransitionError,
    ReviewRequiredError,
)
from student_analyze.image_preprocess import prepare_logical_pages
from student_analyze.models import PipelineStage
from student_analyze.page_models import (
    LogicalPageDecision,
    PageDecisionProvenance,
    PageDecisionSet,
    PageLayout,
    PageManifest,
    PagePosition,
    PixelBox,
    SourcePageDecision,
)
from student_analyze.pipeline import commit_stage_artifact, ingest_case, verify_case


class MappingStub(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"] = "1.0.0"
    nodes: list[str]


def _ingested_case(tmp_path: Path):
    source = tmp_path / "raw"
    source.mkdir()
    Image.new("RGB", (8, 6), "white").save(source / "exam.jpg", format="JPEG")
    config = AppConfig(cases_dir=tmp_path / "cases")
    return ingest_case(source, config), config


def _decisions(result, *, rotation: int = 0, requires_review: bool = False):
    source = result.manifest.source_assets[0]
    manifest_sha256, _ = artifact_digest(result.case_dir / "case_manifest.json")
    return PageDecisionSet(
        case_id=result.manifest.case_id,
        source_manifest_sha256=manifest_sha256,
        provenance=PageDecisionProvenance(
            method="deterministic_test_fixture",
            model_identifier="test-model",
            prompt_version="test-page-prompt-v1",
            decided_at=datetime(2026, 8, 9, tzinfo=UTC),
        ),
        assets=[
            SourcePageDecision(
                source_asset_id=source.asset_id,
                source_sha256=source.sha256,
                layout=PageLayout.SINGLE_PAGE,
                layout_confidence=1.0,
                pages=[
                    LogicalPageDecision(
                        position=PagePosition.SINGLE,
                        crop_box=PixelBox(left=0, top=0, right=8, bottom=6),
                        rotation_clockwise=rotation,
                        orientation_confidence=1.0,
                        boundary_confidence=1.0,
                        requires_review=requires_review,
                        evidence=["deterministic fixture"],
                    )
                ],
            )
        ],
    )


def _prepare(result, config, **overrides):
    decisions = overrides.pop("decisions", _decisions(result))
    return prepare_logical_pages(
        result.case_dir,
        decisions,
        config,
        **overrides,
    )


def test_invalid_page_payload_does_not_create_artifact_or_advance(tmp_path: Path) -> None:
    result, config = _ingested_case(tmp_path)

    with pytest.raises(CaseValidationError, match="PageManifest"):
        commit_stage_artifact(
            result.case_dir,
            stage=PipelineStage.PAGES_READY,
            artifact_name="page_manifest.json",
            payload={"schema_version": "1.0.0"},
            model_type=PageManifest,
            schema_id="page_manifest.schema.json",
            config=config.page_fingerprint_payload(),
            versions=result.manifest.versions,
            inputs=[],
        )

    _, state = verify_case(result.case_dir)
    assert state.current_stage == PipelineStage.INGESTED
    assert not (result.case_dir / "artifacts").exists()


def test_stage_cannot_skip_required_predecessor(tmp_path: Path) -> None:
    result, config = _ingested_case(tmp_path)

    with pytest.raises(InvalidTransitionError, match="required previous stage"):
        commit_stage_artifact(
            result.case_dir,
            stage=PipelineStage.SUBMISSION_READY,
            artifact_name="master.json",
            payload={"schema_version": "1.0.0", "nodes": []},
            model_type=MappingStub,
            schema_id="submission.schema.json",
            config=config.fingerprint_payload(),
            versions=result.manifest.versions,
            inputs=[],
        )


def test_interruption_between_page_manifest_and_state_is_recoverable(tmp_path: Path) -> None:
    result, config = _ingested_case(tmp_path)

    def interrupt(event: str) -> None:
        if event == "before_state_commit":
            raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        _prepare(result, config, interrupt_hook=interrupt)

    _, previous_state = verify_case(result.case_dir)
    assert previous_state.current_stage == PipelineStage.INGESTED
    assert len(list((result.case_dir / "artifacts").rglob("page_manifest.json"))) == 1
    assert len(list((result.case_dir / "artifacts").rglob("*.jpg"))) == 1

    recovered = _prepare(result, config)
    assert not recovered.reused
    assert recovered.state.current_stage == PipelineStage.PAGES_READY
    assert len(list((result.case_dir / "artifacts").rglob("page_manifest.json"))) == 1


def test_same_fingerprint_reuses_complete_pages(tmp_path: Path) -> None:
    result, config = _ingested_case(tmp_path)
    first = _prepare(result, config)
    second = _prepare(result, config)

    assert not first.reused
    assert second.reused
    assert second.state.revision == first.state.revision
    assert len(second.state.run_history) == len(first.state.run_history)
    derived_path = result.case_dir / first.manifest.pages[0].derived.relative_path
    with Image.open(derived_path) as image:
        assert image.size == (8, 6)
        assert image.getexif().get(274) is None


def test_changed_page_fingerprint_requires_force(tmp_path: Path) -> None:
    result, config = _ingested_case(tmp_path)
    _prepare(result, config)
    changed_config = AppConfig(cases_dir=config.cases_dir, jpeg_quality=90)

    with pytest.raises(InvalidTransitionError, match="different fingerprint"):
        _prepare(result, changed_config)


def test_tampered_derived_page_blocks_case_verification(tmp_path: Path) -> None:
    result, config = _ingested_case(tmp_path)
    pages = _prepare(result, config)
    derived_path = result.case_dir / pages.manifest.pages[0].derived.relative_path
    derived_path.write_bytes(b"tampered")

    with pytest.raises(CaseValidationError, match="derived page changed"):
        verify_case(result.case_dir)


def test_force_rerun_versions_outputs_and_preserves_previous_run(tmp_path: Path) -> None:
    result, config = _ingested_case(tmp_path)
    first = _prepare(result, config)
    forced = _prepare(
        result,
        config,
        force=True,
        decisions=_decisions(result, rotation=90),
    )

    assert len(forced.state.run_history) == len(first.state.run_history) + 1
    first_manifest = next(
        artifact
        for artifact in first.state.completed_stages[-1].artifacts
        if artifact.schema_id == "page_manifest.schema.json"
    )
    forced_manifest = next(
        artifact
        for artifact in forced.state.completed_stages[-1].artifacts
        if artifact.schema_id == "page_manifest.schema.json"
    )
    assert forced_manifest.artifact_version == 2
    assert forced_manifest.relative_path != first_manifest.relative_path
    assert (result.case_dir / first_manifest.relative_path).is_file()
    assert (result.case_dir / forced_manifest.relative_path).is_file()
    assert forced.manifest.pages[0].page_image_size.width == 6
    assert forced.manifest.pages[0].page_image_size.height == 8


def test_unreviewed_decision_cannot_advance_or_generate_outputs(tmp_path: Path) -> None:
    result, config = _ingested_case(tmp_path)

    with pytest.raises(ReviewRequiredError, match="require review"):
        _prepare(
            result,
            config,
            decisions=_decisions(result, requires_review=True),
        )

    _, state = verify_case(result.case_dir)
    assert state.current_stage == PipelineStage.INGESTED
    assert not (result.case_dir / "artifacts").exists()


def test_changed_ingestion_cannot_be_forced_below_pages_ready(tmp_path: Path) -> None:
    result, config = _ingested_case(tmp_path)
    _prepare(result, config)
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
