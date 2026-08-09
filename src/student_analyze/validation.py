"""Pydantic plus generated JSON Schema validation helpers."""

from __future__ import annotations

import json
from typing import Any, TypeVar

from jsonschema import Draft202012Validator
from pydantic import BaseModel

from student_analyze.errors import CaseValidationError


ModelT = TypeVar("ModelT", bound=BaseModel)


def validate_payload(model_type: type[ModelT], payload: Any) -> ModelT:
    try:
        schema = model_type.model_json_schema(mode="validation")
        Draft202012Validator.check_schema(schema)
        json_payload = (
            payload.model_dump(mode="json")
            if isinstance(payload, BaseModel)
            else payload
        )
        Draft202012Validator(schema).validate(json_payload)
        model = model_type.model_validate(json_payload)
    except Exception as exc:
        if isinstance(exc, CaseValidationError):
            raise
        raise CaseValidationError(
            f"payload does not satisfy {model_type.__name__}: {exc}"
        ) from exc
    return model


def validate_json(model_type: type[ModelT], raw_json: str | bytes) -> ModelT:
    try:
        payload = json.loads(raw_json)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
        raise CaseValidationError(
            f"invalid JSON for {model_type.__name__}: {exc}"
        ) from exc
    return validate_payload(model_type, payload)
