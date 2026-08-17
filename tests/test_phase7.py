from __future__ import annotations

from contextlib import closing
from pathlib import Path
import sqlite3

import pytest

from student_analyze.assets import artifact_digest
from student_analyze.atomic import atomic_write_model
from student_analyze.database import (
    initialize_student_profile,
    rebuild_database,
    verify_database,
)
from student_analyze.document_models import DecisionMethod
from student_analyze.errors import CaseValidationError
from student_analyze.grading import (
    build_grading,
    finalize_grading_review,
    prepare_grading_context,
)
from student_analyze.grading_models import AcademicErrorType, GradingReviewDecisionSet
from student_analyze.learning import (
    compile_learning_analysis,
    prepare_learning_context,
    prepare_learning_review,
)
from student_analyze.learning_models import (
    AttributionScope,
    ExamMetadataDecision,
    KnowledgeMappingDecision,
    KnowledgePointProposal,
    LearningAnalysisDecision,
    MappingStatus,
    MetadataSource,
)
from student_analyze.models import PipelineStage
from student_analyze.pipeline import verify_case
from student_analyze.reporting import build_report
from tests.test_grading import (
    _grading_decisions,
    _source_grading_sha256,
    _submission_ready,
)
from tests.test_submission import _provenance


ROOT = Path(__file__).parents[1]


def _reviewed_objective_case(tmp_path: Path, *, answer: str = "A"):
    case_dir, config, _ = _submission_ready(tmp_path, observed_content=answer)
    context = prepare_grading_context(case_dir, config)
    build_grading(
        case_dir,
        context.manifest,
        _grading_decisions(context.manifest, []),
        config,
    )
    review = GradingReviewDecisionSet(
        case_id=context.manifest.case_id,
        source_grading_sha256=_source_grading_sha256(case_dir),
        provenance=_provenance(
            prompt="grading-human-review-v1.0.0",
            skill="exam-grading-v1.0.0",
        ).model_copy(update={"method": DecisionMethod.HUMAN_REVIEW}),
        decisions=[],
    )
    finalize_grading_review(case_dir, review, config)
    return case_dir, config


def _learning_inputs(
    tmp_path: Path,
    *,
    answer: str = "A",
    occurred_at: str = "2026-08-11",
    catalog_dir: Path | None = None,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    case_dir, config = _reviewed_objective_case(tmp_path, answer=answer)
    metadata = ExamMetadataDecision(
        case_id=case_dir.name,
        subject="Mathematics",
        title="Unit check",
        occurred_at=occurred_at,
        occurred_at_precision="day",
        source=MetadataSource.USER_PROVIDED,
        confidence=1,
        evidence=["test fixture metadata"],
        human_confirmed=True,
        human_review_note="Fixture metadata reviewed by the test author.",
        provenance=_provenance(
            prompt="exam-metadata-test-v1",
            skill="exam-learning-analysis-v1.0.0",
        ).model_copy(update={"method": DecisionMethod.HUMAN_REVIEW}),
    )
    metadata_path = tmp_path / "exam_metadata.json"
    atomic_write_model(metadata_path, metadata, model_type=ExamMetadataDecision)
    context = prepare_learning_context(
        case_dir,
        config,
        metadata_path=metadata_path,
        catalog_dir=catalog_dir or tmp_path / "catalogs",
    )
    target = context.manifest.targets[0].target
    rubric_ref = target.rubric[0].ref
    decision = LearningAnalysisDecision(
        case_id=case_dir.name,
        learning_input_manifest_sha256=artifact_digest(context.manifest_path)[0],
        provenance=_provenance(
            prompt="learning-analysis-v1.0.0",
            skill="exam-learning-analysis-v1.0.0",
        ),
        proposed_points=(
            [
                KnowledgePointProposal(
                    point_id="point-choice-selection",
                    name="Choice selection",
                    description="Select the expected answer from the supplied alternatives.",
                    confidence=0.95,
                    evidence_target_ids=[target.target_id],
                    human_confirmed=True,
                    human_review_note="Fixture point reviewed by the test author.",
                )
            ]
            if not context.manifest.catalog.points
            else []
        ),
        mappings=[
            KnowledgeMappingDecision(
                mapping_id="mapping-choice-selection",
                target_id=target.target_id,
                rubric_ref=rubric_ref,
                status=MappingStatus.MAPPED,
                point_id="point-choice-selection",
                attribution_scopes=[AttributionScope.ASSESSED],
                weight=1,
                confidence=0.95,
                rationale="The rubric directly assesses selecting the expected choice.",
            )
        ],
    )
    decision_path = tmp_path / "learning_decision.json"
    atomic_write_model(decision_path, decision, model_type=LearningAnalysisDecision)
    return case_dir, config, context.manifest_path, decision_path


def test_phase7_report_persists_and_database_rebuilds(tmp_path: Path) -> None:
    case_dir, config, manifest_path, decision_path = _learning_inputs(tmp_path)
    profile_path = tmp_path / "data" / "student_profile.json"
    db_path = tmp_path / "data" / "student.sqlite3"
    initialize_student_profile(profile_path, display_name="Test Student")

    result = build_report(
        case_dir,
        config,
        manifest_path=manifest_path,
        decision_path=decision_path,
        db_path=db_path,
        migrations_dir=ROOT / "migrations",
        profile_path=profile_path,
        catalog_dir=tmp_path / "catalogs",
    )

    assert result.state.current_stage == PipelineStage.REPORTED
    assert result.report.final_score == 1
    assert len(result.report.claims) == 1
    assert result.report.analysis.snapshots[0].state.value == "insufficient_evidence"
    markdown_asset = next(
        item for item in result.report.assets if item.name == "markdown-report"
    )
    markdown = (case_dir / markdown_asset.relative_path).read_text(encoding="utf-8")
    assert "独立考试：1" in markdown
    assert "target=" in markdown
    assert "rubric=" in markdown
    assert "来源：" in markdown
    status = verify_database(db_path, ROOT / "migrations")
    assert status.active_analyses == 1
    assert status.evidence == 1
    assert status.snapshots == 1
    with closing(sqlite3.connect(db_path)) as connection:
        assert connection.execute("SELECT count(*) FROM review_event").fetchone()[0] == 2
    _, verified = verify_case(case_dir)
    assert verified.current_stage == PipelineStage.REPORTED

    rebuilt_path = tmp_path / "data" / "rebuilt.sqlite3"
    rebuilt = rebuild_database(
        rebuilt_path,
        ROOT / "migrations",
        profile_path=profile_path,
        cases_dir=config.cases_dir,
    )
    assert rebuilt.active_analyses == 1
    assert rebuilt.evidence == status.evidence
    replaced = rebuild_database(
        db_path,
        ROOT / "migrations",
        profile_path=profile_path,
        cases_dir=config.cases_dir,
    )
    assert replaced.logical_sha256 == status.logical_sha256


def test_generic_objective_error_cannot_be_diagnostic_mapping(tmp_path: Path) -> None:
    case_dir, config, manifest_path, decision_path = _learning_inputs(
        tmp_path, answer="B"
    )
    decision = LearningAnalysisDecision.model_validate_json(decision_path.read_bytes())
    mapping = decision.mappings[0].model_copy(
        update={
            "attribution_scopes": [
                AttributionScope.ASSESSED,
                AttributionScope.DIAGNOSTIC,
            ],
            "source_error_types": [AcademicErrorType.INCORRECT_OBJECTIVE],
        }
    )
    invalid = decision.model_copy(update={"mappings": [mapping]})
    atomic_write_model(decision_path, invalid, model_type=LearningAnalysisDecision)

    with pytest.raises(CaseValidationError, match="cannot diagnose"):
        compile_learning_analysis(
            manifest_path,
            decision_path,
            config,
            db_path=tmp_path / "missing.sqlite3",
        )


def test_new_knowledge_point_candidate_requires_review_packet(tmp_path: Path) -> None:
    _, config, manifest_path, decision_path = _learning_inputs(tmp_path)
    decision = LearningAnalysisDecision.model_validate_json(decision_path.read_bytes())
    candidate = decision.model_copy(
        update={
            "proposed_points": [
                decision.proposed_points[0].model_copy(
                    update={
                        "human_confirmed": False,
                        "human_review_note": None,
                        "requires_review": True,
                    }
                )
            ]
        }
    )
    atomic_write_model(decision_path, candidate, model_type=LearningAnalysisDecision)

    packet = prepare_learning_review(manifest_path, decision_path)

    assert packet.manifest.requires_review
    assert packet.manifest.items[0].entity_id == "point-choice-selection"
    with pytest.raises(CaseValidationError, match="unresolved review"):
        compile_learning_analysis(
            manifest_path,
            decision_path,
            config,
            db_path=tmp_path / "missing.sqlite3",
        )


def test_mapping_review_packet_shows_human_readable_context(tmp_path: Path) -> None:
    _, _, manifest_path, decision_path = _learning_inputs(tmp_path)
    decision = LearningAnalysisDecision.model_validate_json(decision_path.read_bytes())
    mapping = decision.mappings[0].model_copy(update={"requires_review": True})
    atomic_write_model(
        decision_path,
        decision.model_copy(update={"mappings": [mapping]}),
        model_type=LearningAnalysisDecision,
    )

    packet = prepare_learning_review(manifest_path, decision_path)
    html = packet.html_path.read_text(encoding="utf-8")

    assert "知识映射复核" in html
    assert "题干" in html
    assert "评分项" in html
    assert "建议映射" in html
    assert "Choice selection" in html
    assert "The rubric directly assesses selecting the expected choice." in html


def test_one_rubric_can_split_evidence_across_multiple_points(tmp_path: Path) -> None:
    _, config, manifest_path, decision_path = _learning_inputs(tmp_path)
    decision = LearningAnalysisDecision.model_validate_json(decision_path.read_bytes())
    first_mapping = decision.mappings[0].model_copy(update={"weight": 0.5})
    target_id = first_mapping.target_id
    second_point = KnowledgePointProposal(
        point_id="point-choice-reasoning",
        name="Choice reasoning",
        description="Reason about the supplied alternatives.",
        confidence=0.95,
        evidence_target_ids=[target_id],
        human_confirmed=True,
        human_review_note="Fixture point reviewed by the test author.",
    )
    second_mapping = first_mapping.model_copy(
        update={
            "mapping_id": "mapping-choice-reasoning",
            "point_id": second_point.point_id,
        }
    )
    split = decision.model_copy(
        update={
            "proposed_points": [*decision.proposed_points, second_point],
            "mappings": [first_mapping, second_mapping],
        }
    )
    atomic_write_model(decision_path, split, model_type=LearningAnalysisDecision)

    compiled = compile_learning_analysis(
        manifest_path,
        decision_path,
        config,
        db_path=tmp_path / "missing.sqlite3",
    )

    assert len(compiled.analysis.evidence) == 2
    assert {item.point_id for item in compiled.analysis.evidence} == {
        "point-choice-selection",
        "point-choice-reasoning",
    }
    assert all(item.allocated_points == 0.5 for item in compiled.analysis.evidence)


def test_second_exam_uses_exact_catalog_history_for_longitudinal_state(
    tmp_path: Path,
) -> None:
    shared = tmp_path / "shared"
    catalog_dir = shared / "catalogs"
    profile_path = shared / "student_profile.json"
    db_path = shared / "student.sqlite3"
    initialize_student_profile(profile_path)

    first = _learning_inputs(
        tmp_path / "first",
        occurred_at="2026-08-11",
        catalog_dir=catalog_dir,
    )
    build_report(
        first[0],
        first[1],
        manifest_path=first[2],
        decision_path=first[3],
        db_path=db_path,
        migrations_dir=ROOT / "migrations",
        profile_path=profile_path,
        catalog_dir=catalog_dir,
    )

    second = _learning_inputs(
        tmp_path / "second",
        occurred_at="2026-08-12",
        catalog_dir=catalog_dir,
    )
    result = build_report(
        second[0],
        second[1],
        manifest_path=second[2],
        decision_path=second[3],
        db_path=db_path,
        migrations_dir=ROOT / "migrations",
        profile_path=profile_path,
        catalog_dir=catalog_dir,
    )

    snapshot = result.report.analysis.snapshots[0]
    assert result.report.analysis.catalog.version == 2
    assert snapshot.independent_exam_count == 2
    assert snapshot.state.value == "consistent"
    assert snapshot.allowed_claim_strength.value == "longitudinal"
    assert snapshot.trend.value == "stable"
    assert len(result.report.analysis.historical_evidence) == 1
    replay = compile_learning_analysis(
        first[2], first[3], first[1], db_path=db_path
    )
    assert replay.analysis.snapshots[0].independent_exam_count == 1
    assert replay.analysis.historical_evidence == []
