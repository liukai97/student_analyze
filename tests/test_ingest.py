from __future__ import annotations

import os
from pathlib import Path

import pytest

from student_analyze.config import AppConfig
from student_analyze.errors import (
    ConfigurationError,
    InvalidTransitionError,
    SourceIntegrityError,
)
from student_analyze.pipeline import ingest_case, verify_case


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "raw"
    source.mkdir()
    (source / "1.jpg").write_bytes(b"first-image")
    (source / "2.png").write_bytes(b"second-image")
    return source


def _config(tmp_path: Path) -> AppConfig:
    return AppConfig(cases_dir=tmp_path / "cases")


def test_repeated_ingest_reuses_one_stable_case(tmp_path: Path) -> None:
    source = _source(tmp_path)
    before = {path.name: path.read_bytes() for path in source.iterdir()}

    first = ingest_case(source, _config(tmp_path))
    second = ingest_case(source, _config(tmp_path))

    assert first.manifest.case_id == second.manifest.case_id
    assert first.case_dir == second.case_dir
    assert not first.reused
    assert second.reused
    assert len(second.manifest.source_assets) == 2
    assert len(second.state.run_history) == 1
    assert len(list((_config(tmp_path).cases_dir).glob("case-*"))) == 1
    assert {path.name: path.read_bytes() for path in source.iterdir()} == before


def test_force_ingest_adds_run_without_overwriting_manifest(tmp_path: Path) -> None:
    source = _source(tmp_path)
    first = ingest_case(source, _config(tmp_path))
    manifest_path = first.case_dir / "case_manifest.json"
    manifest_bytes = manifest_path.read_bytes()

    forced = ingest_case(source, _config(tmp_path), force=True)

    assert not forced.reused
    assert manifest_path.read_bytes() == manifest_bytes
    assert len(forced.state.run_history) == 2
    assert forced.state.run_history[-1].forced


def test_source_change_after_ingestion_is_rejected(tmp_path: Path) -> None:
    source = _source(tmp_path)
    result = ingest_case(source, _config(tmp_path))
    (source / "1.jpg").write_bytes(b"changed-image")

    with pytest.raises(SourceIntegrityError, match="previously ingested"):
        ingest_case(source, _config(tmp_path))
    with pytest.raises(SourceIntegrityError, match="source asset changed"):
        verify_case(result.case_dir)


def test_mtime_only_change_does_not_invalidate_identical_source(tmp_path: Path) -> None:
    source = _source(tmp_path)
    result = ingest_case(source, _config(tmp_path))
    source_path = source / "1.jpg"
    original = source_path.stat()
    os.utime(
        source_path,
        ns=(original.st_atime_ns, original.st_mtime_ns + 1_000_000_000),
    )

    verify_case(result.case_dir)


def test_interruption_after_manifest_recovers_on_retry(tmp_path: Path) -> None:
    source = _source(tmp_path)

    def interrupt(event: str) -> None:
        if event == "after_artifact_commit":
            raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        ingest_case(source, _config(tmp_path), interrupt_hook=interrupt)

    case_dirs = list(_config(tmp_path).cases_dir.glob("case-*"))
    assert len(case_dirs) == 1
    assert (case_dirs[0] / "case_manifest.json").is_file()
    assert not (case_dirs[0] / "pipeline_state.json").exists()

    recovered = ingest_case(source, _config(tmp_path))
    assert recovered.state.current_stage.value == "ingested"
    verify_case(recovered.case_dir)


def test_change_during_ingest_never_advances_state(tmp_path: Path) -> None:
    source = _source(tmp_path)

    def mutate_source(event: str) -> None:
        if event == "after_artifact_commit":
            (source / "1.jpg").write_bytes(b"changed-during-run")

    with pytest.raises(SourceIntegrityError, match="source asset changed"):
        ingest_case(source, _config(tmp_path), interrupt_hook=mutate_source)

    case_dir = next(_config(tmp_path).cases_dir.glob("case-*"))
    assert (case_dir / "case_manifest.json").is_file()
    assert not (case_dir / "pipeline_state.json").exists()


def test_case_output_cannot_be_created_inside_source_root(tmp_path: Path) -> None:
    source = _source(tmp_path)
    config = AppConfig(cases_dir=source / "generated-cases")

    with pytest.raises(ConfigurationError, match="read-only source root"):
        ingest_case(source, config)

    assert not config.cases_dir.exists()


def test_changed_config_requires_force_and_is_recorded_per_run(tmp_path: Path) -> None:
    source = _source(tmp_path)
    initial_config = _config(tmp_path)
    initial = ingest_case(source, initial_config)
    changed_config = AppConfig(
        config_version="2",
        cases_dir=initial_config.cases_dir,
    )

    with pytest.raises(InvalidTransitionError, match="different configuration"):
        ingest_case(source, changed_config)

    forced = ingest_case(source, changed_config, force=True)
    assert forced.manifest.versions.config == "1"
    assert forced.state.run_history[-1].versions.config == "2"
    assert (
        forced.state.run_history[-1].config_fingerprint
        != initial.state.run_history[-1].config_fingerprint
    )
