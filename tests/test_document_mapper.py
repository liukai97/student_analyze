from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from PIL import Image
import pytest

from student_analyze.assets import artifact_digest
from student_analyze.config import AppConfig
from student_analyze.document_mapper import map_documents
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
    VersionResolutionStatus,
)
from student_analyze.errors import CaseValidationError
from student_analyze.image_preprocess import prepare_logical_pages
from student_analyze.models import PipelineStage
from student_analyze.page_models import (
    LogicalPageDecision,
    PageDecisionProvenance,
    PageDecisionSet,
    PageLayout,
    PagePosition,
    PixelBox,
    SourcePageDecision,
)
from student_analyze.pipeline import ingest_case, verify_case


def _pages_ready_case(tmp_path: Path):
    source = tmp_path / "raw"
    source.mkdir()
    Image.new("RGB", (40, 20), "white").save(source / "exam.jpg", format="JPEG")
    config = AppConfig(cases_dir=tmp_path / "cases")
    ingested = ingest_case(source, config)
    asset = ingested.manifest.source_assets[0]
    case_manifest_sha256, _ = artifact_digest(ingested.case_dir / "case_manifest.json")
    page_decisions = PageDecisionSet(
        case_id=ingested.manifest.case_id,
        source_manifest_sha256=case_manifest_sha256,
        provenance=PageDecisionProvenance(
            method="deterministic_test_fixture",
            model_identifier="test-model",
            prompt_version="test-page-prompt-v1",
            decided_at=datetime(2026, 8, 9, tzinfo=UTC),
        ),
        assets=[
            SourcePageDecision(
                source_asset_id=asset.asset_id,
                source_sha256=asset.sha256,
                layout=PageLayout.DOUBLE_PAGE,
                layout_confidence=1.0,
                pages=[
                    LogicalPageDecision(
                        position=PagePosition.LEFT,
                        crop_box=PixelBox(left=0, top=0, right=20, bottom=20),
                        rotation_clockwise=0,
                        orientation_confidence=1.0,
                        boundary_confidence=1.0,
                        evidence=["test left page"],
                    ),
                    LogicalPageDecision(
                        position=PagePosition.RIGHT,
                        crop_box=PixelBox(left=20, top=0, right=40, bottom=20),
                        rotation_clockwise=0,
                        orientation_confidence=1.0,
                        boundary_confidence=1.0,
                        evidence=["test right page"],
                    ),
                ],
            )
        ],
    )
    pages = prepare_logical_pages(ingested.case_dir, page_decisions, config)
    reference = pages.state.completed_stages[-1].artifacts[0]
    page_manifest_sha256, _ = artifact_digest(
        ingested.case_dir / reference.relative_path
    )
    return ingested, pages, config, page_manifest_sha256


def _mapping_decisions(
    pages,
    page_manifest_sha256: str,
    *,
    versions: int = 1,
    accept_supersedes: bool = False,
    candidate_supersedes: bool = False,
) -> DocumentGraphDecisionSet:
    question_page, answer_page = pages.manifest.pages
    regions = [
        RegionDecision(
            ref="q1-original-region",
            page_id=question_page.page_id,
            kind=RegionKind.PRINTED_QUESTION,
            bbox=PixelBox(left=1, top=1, right=19, bottom=9),
            order=1,
            confidence=1.0,
            evidence=["printed question 1"],
        ),
        RegionDecision(
            ref="answer-q1",
            page_id=answer_page.page_id,
            kind=RegionKind.ANSWER_AREA,
            bbox=PixelBox(left=1, top=1, right=19, bottom=19),
            order=1,
            confidence=1.0,
            evidence=["printed answer area 1"],
        ),
    ]
    question_versions = [
        QuestionVersionDecision(
            ref="q1-original",
            question_ref="q1",
            label="original",
            region_refs=["q1-original-region"],
            confidence=1.0,
            evidence=["original question 1"],
        )
    ]
    relations: list[RelationDecision] = []
    answer_target = "q1-original"
    if versions == 2:
        regions.append(
            RegionDecision(
                ref="q1-replacement-region",
                page_id=question_page.page_id,
                kind=RegionKind.PRINTED_QUESTION,
                bbox=PixelBox(left=1, top=10, right=19, bottom=19),
                order=2,
                confidence=1.0,
                evidence=["replacement question 1"],
            )
        )
        question_versions.append(
            QuestionVersionDecision(
                ref="q1-replacement",
                question_ref="q1",
                label="replacement",
                region_refs=["q1-replacement-region"],
                confidence=1.0,
                evidence=["replacement question 1"],
            )
        )
        answer_target = "q1-replacement"
        if accept_supersedes:
            relations.append(
                RelationDecision(
                    type=RelationType.SUPERSEDES,
                    from_ref="q1-replacement",
                    to_ref="q1-original",
                    status=RelationDecisionStatus.ACCEPTED,
                    confidence=1.0,
                    evidence=["explicit printed replacement notice"],
                    requires_review=False,
                )
            )
        elif candidate_supersedes:
            relations.append(
                RelationDecision(
                    type=RelationType.SUPERSEDES,
                    from_ref="q1-replacement",
                    to_ref="q1-original",
                    status=RelationDecisionStatus.CANDIDATE,
                    confidence=0.6,
                    evidence=["possible but ambiguous replacement notice"],
                    requires_review=True,
                )
            )
    relations.append(
        RelationDecision(
            type=RelationType.ANSWERS,
            from_ref="answer-q1",
            to_ref=answer_target,
            status=RelationDecisionStatus.ACCEPTED,
            confidence=1.0,
            evidence=["answer area is printed for question 1"],
            requires_review=False,
        )
    )
    return DocumentGraphDecisionSet(
        case_id=pages.manifest.case_id,
        page_manifest_sha256=page_manifest_sha256,
        provenance=DocumentDecisionProvenance(
            method=DecisionMethod.DETERMINISTIC_TEST_FIXTURE,
            model_identifier="test-model",
            prompt_version="test-document-prompt-v1",
            skill_version="test-exam-analysis-v1",
            decided_at=datetime(2026, 8, 9, tzinfo=UTC),
        ),
        documents=[
            DocumentDecision(
                ref="question-booklet",
                role=DocumentRole.QUESTION_BOOKLET,
                confidence=1.0,
                evidence=["question booklet fixture"],
            ),
            DocumentDecision(
                ref="answer-sheet",
                role=DocumentRole.ANSWER_SHEET,
                confidence=1.0,
                evidence=["answer sheet fixture"],
            ),
        ],
        pages=[
            PageClassificationDecision(
                page_id=question_page.page_id,
                document_ref="question-booklet",
                order=1,
                printed_page_number="1",
                confidence=1.0,
                evidence=["question page fixture"],
            ),
            PageClassificationDecision(
                page_id=answer_page.page_id,
                document_ref="answer-sheet",
                order=1,
                printed_page_number="1",
                confidence=1.0,
                evidence=["answer page fixture"],
            ),
        ],
        questions=[
            QuestionDecision(
                ref="q1",
                printed_label="1",
                order=1,
                confidence=1.0,
                evidence=["printed label 1"],
            )
        ],
        question_versions=question_versions,
        regions=regions,
        relations=relations,
    )


def test_map_documents_commits_reuses_and_derives_edges(tmp_path: Path) -> None:
    ingested, pages, config, page_manifest_sha256 = _pages_ready_case(tmp_path)
    decisions = _mapping_decisions(pages, page_manifest_sha256)

    first = map_documents(ingested.case_dir, decisions, config)
    second = map_documents(ingested.case_dir, decisions, config)

    assert not first.reused
    assert second.reused
    assert first.state.current_stage == PipelineStage.MAPPED
    assert not first.graph.requires_review
    assert len(first.graph.documents) == 2
    assert len(first.graph.questions) == 1
    assert first.graph.questions[0].effective_version_id is not None
    assert first.graph.question_versions[0].resolution_status == VersionResolutionStatus.EFFECTIVE
    relation_types = [relation.type for relation in first.graph.relationships]
    assert relation_types.count(RelationType.CONTAINS_PAGE) == 2
    assert relation_types.count(RelationType.DERIVED_FROM) == 2
    assert relation_types.count(RelationType.ANSWERS) == 1
    verify_case(ingested.case_dir)


def test_python_does_not_infer_supersedes_from_duplicate_question_versions(
    tmp_path: Path,
) -> None:
    ingested, pages, config, page_manifest_sha256 = _pages_ready_case(tmp_path)
    decisions = _mapping_decisions(
        pages,
        page_manifest_sha256,
        versions=2,
        accept_supersedes=False,
    )

    result = map_documents(ingested.case_dir, decisions, config)

    assert result.graph.requires_review
    assert result.graph.questions[0].effective_version_id is None
    assert {
        version.resolution_status for version in result.graph.question_versions
    } == {VersionResolutionStatus.UNRESOLVED}
    assert not any(
        relation.type == RelationType.SUPERSEDES
        for relation in result.graph.relationships
    )


def test_accepted_supersedes_selects_only_the_unsuperseded_version(
    tmp_path: Path,
) -> None:
    ingested, pages, config, page_manifest_sha256 = _pages_ready_case(tmp_path)
    decisions = _mapping_decisions(
        pages,
        page_manifest_sha256,
        versions=2,
        accept_supersedes=True,
    )

    result = map_documents(ingested.case_dir, decisions, config)

    versions = {
        version.decision_ref: version for version in result.graph.question_versions
    }
    assert not result.graph.requires_review
    assert versions["q1-replacement"].resolution_status == VersionResolutionStatus.EFFECTIVE
    assert versions["q1-original"].resolution_status == VersionResolutionStatus.SUPERSEDED
    assert (
        result.graph.questions[0].effective_version_id
        == versions["q1-replacement"].version_id
    )


def test_candidate_supersedes_is_preserved_without_selecting_a_version(
    tmp_path: Path,
) -> None:
    ingested, pages, config, page_manifest_sha256 = _pages_ready_case(tmp_path)
    decisions = _mapping_decisions(
        pages,
        page_manifest_sha256,
        versions=2,
        candidate_supersedes=True,
    )

    result = map_documents(ingested.case_dir, decisions, config)

    assert result.graph.requires_review
    assert result.graph.questions[0].effective_version_id is None
    assert len(result.graph.unresolved_relations) == 1
    assert result.graph.unresolved_relations[0].type == RelationType.SUPERSEDES
    assert any(
        item.blocks_master_ready and item.blocks_submission_ready
        for item in result.graph.review_items
    )


def test_region_outside_logical_page_is_rejected(tmp_path: Path) -> None:
    ingested, pages, config, page_manifest_sha256 = _pages_ready_case(tmp_path)
    decisions = _mapping_decisions(pages, page_manifest_sha256)
    bad_region = decisions.regions[0].model_copy(
        update={"bbox": PixelBox(left=1, top=1, right=21, bottom=9)}
    )
    decisions = decisions.model_copy(
        update={"regions": [bad_region, *decisions.regions[1:]]}
    )

    with pytest.raises(CaseValidationError, match="exceeds logical page"):
        map_documents(ingested.case_dir, decisions, config)


def test_interrupted_mapping_reuses_orphan_artifact(tmp_path: Path) -> None:
    ingested, pages, config, page_manifest_sha256 = _pages_ready_case(tmp_path)
    decisions = _mapping_decisions(pages, page_manifest_sha256)

    def interrupt(event: str) -> None:
        if event == "before_state_commit":
            raise RuntimeError("simulated mapping interruption")

    with pytest.raises(RuntimeError, match="simulated mapping interruption"):
        map_documents(
            ingested.case_dir,
            decisions,
            config,
            interrupt_hook=interrupt,
        )

    _, state = verify_case(ingested.case_dir)
    assert state.current_stage == PipelineStage.PAGES_READY
    assert len(list((ingested.case_dir / "artifacts").rglob("document_graph.json"))) == 1

    recovered = map_documents(ingested.case_dir, decisions, config)
    assert not recovered.reused
    assert recovered.state.current_stage == PipelineStage.MAPPED
    assert len(list((ingested.case_dir / "artifacts").rglob("document_graph.json"))) == 1
