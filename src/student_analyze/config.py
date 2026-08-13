"""Small TOML configuration layer for the local CLI."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import tomllib

from student_analyze.errors import ConfigurationError


DEFAULT_EXTENSIONS = (
    ".jpeg",
    ".jpg",
    ".pdf",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
)
VALID_LOG_LEVELS = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}


@dataclass(frozen=True, slots=True)
class AppConfig:
    config_version: str = "1"
    cases_dir: Path = Path("cases")
    log_level: str = "INFO"
    source_extensions: tuple[str, ...] = DEFAULT_EXTENSIONS
    jpeg_quality: int = 95

    def fingerprint_payload(self) -> dict[str, object]:
        return {
            "config_version": self.config_version,
            "source_extensions": list(self.source_extensions),
        }

    def page_fingerprint_payload(self) -> dict[str, object]:
        return {
            "config_version": self.config_version,
            "jpeg_quality": self.jpeg_quality,
            "output_format": "JPEG",
            "jpeg_subsampling": 0,
            "exif_transpose": False,
        }

    def mapping_fingerprint_payload(self) -> dict[str, object]:
        return {
            "config_version": self.config_version,
            "graph_id_strategy": "case-kind-decision-ref-sha256-v1",
            "supersedes_direction": "newer_to_older",
            "effective_version_strategy": "unique_unsuperseded_root-v1",
        }

    def master_input_fingerprint_payload(self) -> dict[str, object]:
        return {
            "config_version": self.config_version,
            "jpeg_quality": self.jpeg_quality,
            "output_format": "JPEG",
            "jpeg_subsampling": 0,
            "crop_policy": "reviewed-printed-region-only-v1",
            "routing_policy": "evidence-first-selective-blind-solve-v1",
        }

    def master_fingerprint_payload(self) -> dict[str, object]:
        return {
            "config_version": self.config_version,
            "answer_source_policy": "official-confirmed-teacher-independent-v1",
            "approval_policy": "validated-source-or-independent-review-v1",
            "conflict_policy": "explicit-review-no-silent-override-v1",
        }

    def submission_context_fingerprint_payload(self) -> dict[str, object]:
        return {
            "config_version": self.config_version,
            "redaction_policy": "structure-only-no-answer-rubric-solution-v1",
            "navigation_policy": "answer-question-and-scratch-pages-v1",
        }

    def submission_input_fingerprint_payload(self) -> dict[str, object]:
        return {
            "config_version": self.config_version,
            "jpeg_quality": self.jpeg_quality,
            "output_format": "JPEG",
            "jpeg_subsampling": 0,
            "crop_policy": "validated-response-unit-original-detail-v1",
        }

    def submission_fingerprint_payload(self) -> dict[str, object]:
        return {
            "config_version": self.config_version,
            "source_priority": "answer-sheet-over-scratch-v1",
            "normalization_policy": "format-only-no-semantic-expansion-v1",
            "review_policy": "low-confidence-alternatives-role-conflict-v1",
        }

    def grading_context_fingerprint_payload(self) -> dict[str, object]:
        return {
            "config_version": self.config_version,
            "routing_policy": "objective-and-blank-auto-otherwise-llm-v1",
            "input_policy": "phase-5-review-must-be-complete-v1",
        }

    def grading_fingerprint_payload(self) -> dict[str, object]:
        return {
            "config_version": self.config_version,
            "routing_policy": "objective-and-blank-auto-otherwise-llm-v1",
            "objective_match_policy": "normalized-option-set-exact-v1",
            "blank_policy": "all-formal-slots-blank-zero-v1",
            "rubric_policy": "criterion-complete-evidence-backed-v1",
        }

    def grading_review_fingerprint_payload(self) -> dict[str, object]:
        return {
            "config_version": self.config_version,
            "review_policy": "required-targets-human-overlay-v1",
        }


def load_config(path: Path | None = None, *, cases_dir: Path | None = None) -> AppConfig:
    config = AppConfig()
    if path is not None:
        try:
            with path.open("rb") as handle:
                parsed = tomllib.load(handle)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ConfigurationError(f"cannot read config {path}: {exc}") from exc

        section = parsed.get("student_analyze")
        if not isinstance(section, dict):
            raise ConfigurationError("config must contain a [student_analyze] table")
        unknown = set(section) - {
            "config_version",
            "cases_dir",
            "log_level",
            "source_extensions",
            "jpeg_quality",
        }
        if unknown:
            raise ConfigurationError(f"unknown config keys: {', '.join(sorted(unknown))}")

        config_version = section.get("config_version", config.config_version)
        configured_cases = section.get("cases_dir", str(config.cases_dir))
        log_level = section.get("log_level", config.log_level)
        extensions = section.get("source_extensions", list(config.source_extensions))
        jpeg_quality = section.get("jpeg_quality", config.jpeg_quality)
        if not isinstance(config_version, str) or not config_version:
            raise ConfigurationError("config_version must be a non-empty string")
        if not isinstance(configured_cases, str) or not configured_cases:
            raise ConfigurationError("cases_dir must be a non-empty string")
        if not isinstance(log_level, str) or log_level.upper() not in VALID_LOG_LEVELS:
            raise ConfigurationError("log_level must be a standard Python log level")
        if not isinstance(extensions, list) or not extensions or not all(
            isinstance(item, str) and item for item in extensions
        ):
            raise ConfigurationError("source_extensions must be a non-empty string array")
        if not isinstance(jpeg_quality, int) or isinstance(jpeg_quality, bool):
            raise ConfigurationError("jpeg_quality must be an integer")
        if not 1 <= jpeg_quality <= 100:
            raise ConfigurationError("jpeg_quality must be between 1 and 100")

        configured_path = Path(configured_cases)
        if not configured_path.is_absolute():
            configured_path = path.parent / configured_path
        normalized_extensions = _normalize_extensions(extensions)
        config = AppConfig(
            config_version=config_version,
            cases_dir=configured_path,
            log_level=log_level.upper(),
            source_extensions=normalized_extensions,
            jpeg_quality=jpeg_quality,
        )

    if cases_dir is not None:
        config = replace(config, cases_dir=cases_dir)
    return config


def _normalize_extensions(extensions: list[str]) -> tuple[str, ...]:
    normalized = {item.lower() if item.startswith(".") else f".{item.lower()}" for item in extensions}
    return tuple(sorted(normalized))
