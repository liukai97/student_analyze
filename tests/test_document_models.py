from __future__ import annotations

from datetime import UTC, datetime

import pytest

from student_analyze.document_models import (
    DecisionMethod,
    DocumentDecision,
    DocumentDecisionProvenance,
    DocumentGraphDecisionSet,
    DocumentRole,
    PageClassificationDecision,
    QuestionDecision,
    QuestionVersionDecision,
    RegionDecision,
    RegionKind,
    RelationDecision,
    RelationDecisionStatus,
    RelationType,
)
from student_analyze.errors import CaseValidationError
from student_analyze.page_models import PixelBox
from student_analyze.validation import validate_payload


def _two_version_payload() -> dict[str, object]:
    decision = DocumentGraphDecisionSet(
        case_id="case-1234",
        page_manifest_sha256="a" * 64,
        provenance=DocumentDecisionProvenance(
            method=DecisionMethod.DETERMINISTIC_TEST_FIXTURE,
            model_identifier="test-model",
            prompt_version="test-prompt",
            skill_version="test-skill",
            decided_at=datetime(2026, 8, 9, tzinfo=UTC),
        ),
        documents=[
            DocumentDecision(
                ref="booklet",
                role=DocumentRole.QUESTION_BOOKLET,
                confidence=1.0,
                evidence=["fixture"],
            )
        ],
        pages=[
            PageClassificationDecision(
                page_id="page-1234-single",
                document_ref="booklet",
                order=1,
                confidence=1.0,
                evidence=["fixture"],
            )
        ],
        questions=[
            QuestionDecision(
                ref="q1",
                printed_label="1",
                order=1,
                confidence=1.0,
                evidence=["fixture"],
            )
        ],
        question_versions=[
            QuestionVersionDecision(
                ref="q1-old",
                question_ref="q1",
                label="old",
                region_refs=["q1-old-region"],
                confidence=1.0,
                evidence=["fixture"],
            ),
            QuestionVersionDecision(
                ref="q1-new",
                question_ref="q1",
                label="new",
                region_refs=["q1-new-region"],
                confidence=1.0,
                evidence=["fixture"],
            ),
        ],
        regions=[
            RegionDecision(
                ref="q1-old-region",
                page_id="page-1234-single",
                kind=RegionKind.PRINTED_QUESTION,
                bbox=PixelBox(left=0, top=0, right=5, bottom=5),
                order=1,
                confidence=1.0,
                evidence=["fixture"],
            ),
            RegionDecision(
                ref="q1-new-region",
                page_id="page-1234-single",
                kind=RegionKind.PRINTED_QUESTION,
                bbox=PixelBox(left=5, top=0, right=10, bottom=5),
                order=2,
                confidence=1.0,
                evidence=["fixture"],
            ),
        ],
    )
    return decision.model_dump(mode="json")


def test_supersedes_must_connect_versions_of_the_same_question() -> None:
    payload = _two_version_payload()
    payload["questions"].append(
        {
            "ref": "q2",
            "printed_label": "2",
            "parent_ref": None,
            "order": 2,
            "confidence": 1.0,
            "evidence": ["fixture"],
            "requires_review": False,
            "warnings": [],
        }
    )
    payload["question_versions"].append(
        {
            "ref": "q2-original",
            "question_ref": "q2",
            "label": "original",
            "region_refs": ["q2-region"],
            "confidence": 1.0,
            "evidence": ["fixture"],
            "requires_review": False,
            "warnings": [],
        }
    )
    payload["regions"].append(
        {
            "ref": "q2-region",
            "page_id": "page-1234-single",
            "kind": "printed_question",
            "bbox": {"left": 0, "top": 5, "right": 5, "bottom": 10},
            "order": 3,
            "confidence": 1.0,
            "evidence": ["fixture"],
            "requires_review": False,
            "warnings": [],
        }
    )
    payload["relations"] = [
        {
            "type": "supersedes",
            "from_ref": "q1-new",
            "to_ref": "q2-original",
            "status": "accepted",
            "confidence": 1.0,
            "evidence": ["invalid cross-question relation"],
            "requires_review": False,
        }
    ]

    with pytest.raises(CaseValidationError, match="same question"):
        validate_payload(DocumentGraphDecisionSet, payload)


def test_accepted_supersedes_cycle_is_rejected() -> None:
    payload = _two_version_payload()
    payload["relations"] = [
        {
            "type": "supersedes",
            "from_ref": "q1-new",
            "to_ref": "q1-old",
            "status": RelationDecisionStatus.ACCEPTED.value,
            "confidence": 1.0,
            "evidence": ["fixture"],
            "requires_review": False,
        },
        {
            "type": "supersedes",
            "from_ref": "q1-old",
            "to_ref": "q1-new",
            "status": RelationDecisionStatus.ACCEPTED.value,
            "confidence": 1.0,
            "evidence": ["fixture"],
            "requires_review": False,
        },
    ]

    with pytest.raises(CaseValidationError, match="acyclic"):
        validate_payload(DocumentGraphDecisionSet, payload)


def test_candidate_relationship_must_require_review() -> None:
    payload = _two_version_payload()
    payload["relations"] = [
        {
            "type": RelationType.SUPERSEDES.value,
            "from_ref": "q1-new",
            "to_ref": "q1-old",
            "status": RelationDecisionStatus.CANDIDATE.value,
            "confidence": 0.6,
            "evidence": ["ambiguous notice"],
            "requires_review": False,
        }
    ]

    with pytest.raises(CaseValidationError, match="must require review"):
        validate_payload(DocumentGraphDecisionSet, payload)
