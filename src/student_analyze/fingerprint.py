"""Deterministic fingerprints for cases, configuration, and pipeline stages."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

from student_analyze.models import ImplementationVersions, PipelineStage, SourceAsset


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def digest_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def compute_input_fingerprint(source_root: str, assets: Sequence[SourceAsset]) -> str:
    return digest_value(
        {
            "source_root": source_root,
            "assets": [
                {
                    "relative_path": asset.relative_path,
                    "sha256": asset.sha256,
                    "size_bytes": asset.size_bytes,
                }
                for asset in assets
            ],
        }
    )


def compute_stage_fingerprint(
    stage: PipelineStage,
    *,
    inputs: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    versions: ImplementationVersions,
) -> str:
    return digest_value(
        {
            "stage": stage.value,
            "inputs": list(inputs),
            "config": dict(config),
            "versions": versions.model_dump(mode="json"),
        }
    )
