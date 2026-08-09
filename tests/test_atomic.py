from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from student_analyze.atomic import atomic_write_model
from student_analyze.errors import AtomicCommitError


class TinyModel(BaseModel):
    value: int


def test_interruption_before_replace_leaves_previous_file_intact(tmp_path: Path) -> None:
    destination = tmp_path / "state.json"
    destination.write_text('{"value": 1}\n', encoding="utf-8")

    def interrupt(event: str) -> None:
        assert event == "after_temporary_validation"
        raise RuntimeError("stop before replace")

    with pytest.raises(AtomicCommitError, match="stop before replace"):
        atomic_write_model(
            destination,
            TinyModel(value=2),
            model_type=TinyModel,
            interrupt_hook=interrupt,
        )

    assert destination.read_text(encoding="utf-8") == '{"value": 1}\n'
    assert not list(tmp_path.glob("*.tmp"))
