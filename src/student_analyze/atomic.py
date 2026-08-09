"""Validated same-directory temporary writes followed by atomic replacement."""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable, TypeVar

from pydantic import BaseModel

from student_analyze.errors import AtomicCommitError
from student_analyze.validation import validate_json


ModelT = TypeVar("ModelT", bound=BaseModel)
InterruptHook = Callable[[str], None]


def serialize_model(model: BaseModel) -> bytes:
    text = json.dumps(
        model.model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    return (text + "\n").encode("utf-8")


def atomic_write_model(
    path: Path,
    model: ModelT,
    *,
    model_type: type[ModelT],
    interrupt_hook: InterruptHook | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(serialize_model(model))
            handle.flush()
            os.fsync(handle.fileno())

        raw = temporary_path.read_bytes()
        validate_json(model_type, raw)
        if interrupt_hook is not None:
            interrupt_hook("after_temporary_validation")
        os.replace(temporary_path, path)
        temporary_path = None
        _fsync_directory(path.parent)
    except Exception as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        if isinstance(exc, AtomicCommitError):
            raise
        raise AtomicCommitError(f"atomic write failed for {path}: {exc}") from exc


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
