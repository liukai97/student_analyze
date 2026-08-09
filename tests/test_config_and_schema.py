from __future__ import annotations

from pathlib import Path

import pytest

from student_analyze.config import load_config
from student_analyze.errors import CaseValidationError, ConfigurationError
from student_analyze.models import SourceAsset
from student_analyze.schema import SCHEMA_MODELS, schema_bytes, write_schemas
from student_analyze.validation import validate_payload


def test_toml_config_is_read_and_normalized(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.toml"
    config_path.write_text(
        "[student_analyze]\n"
        'config_version = "2"\n'
        'cases_dir = "output"\n'
        'log_level = "debug"\n'
        'source_extensions = ["JPG", ".png", "jpg"]\n',
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.config_version == "2"
    assert config.cases_dir == tmp_path / "output"
    assert config.log_level == "DEBUG"
    assert config.source_extensions == (".jpg", ".png")


def test_unknown_config_key_is_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.toml"
    config_path.write_text("[student_analyze]\nunknown = true\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="unknown config keys"):
        load_config(config_path)


def test_committed_schemas_match_pydantic_models() -> None:
    schema_dir = Path(__file__).parents[1] / "schemas"
    for filename, model_type in SCHEMA_MODELS.items():
        assert (schema_dir / filename).read_bytes() == schema_bytes(filename, model_type)


def test_schema_check_reports_drift(tmp_path: Path) -> None:
    write_schemas(tmp_path)
    target = tmp_path / next(iter(SCHEMA_MODELS))
    target.write_text("{}", encoding="utf-8")

    assert target in write_schemas(tmp_path, check=True)


def test_json_schema_rejects_types_pydantic_might_coerce() -> None:
    payload = {
        "asset_id": "asset-1234",
        "relative_path": "exam.jpg",
        "source_path": "C:/exam.jpg",
        "sha256": "a" * 64,
        "size_bytes": "4",
        "modified_time_ns": 1,
        "media_type": "image/jpeg",
    }

    with pytest.raises(CaseValidationError, match="SourceAsset"):
        validate_payload(SourceAsset, payload)
