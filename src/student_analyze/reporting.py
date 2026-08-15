"""Evidence-constrained phase 7 report rendering and two-phase persistence."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from html import escape
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from student_analyze import __version__
from student_analyze.assets import artifact_digest
from student_analyze.atomic import (
    InterruptHook,
    atomic_write_bytes,
    atomic_write_model,
    serialize_model,
)
from student_analyze.config import AppConfig
from student_analyze.database import (
    activate_analysis_import,
    prepare_analysis_import,
    read_student_profile,
)
from student_analyze.errors import CaseValidationError, InvalidTransitionError
from student_analyze.fingerprint import digest_value
from student_analyze.learning import compile_learning_analysis
from student_analyze.learning_models import (
    ClaimKind,
    ClaimStrength,
    KnowledgeCatalog,
    LearningAnalysis,
    MasterySnapshot,
    MasteryState,
    ReportAsset,
    ReportClaim,
    ReportManifest,
    TrendStatus,
)
from student_analyze.models import ImplementationVersions, PipelineStage, PipelineState
from student_analyze.pipeline import (
    StageResult,
    build_stage_fingerprint,
    commit_stage_artifact,
    verify_case,
)
from student_analyze.validation import validate_json


ModelT = TypeVar("ModelT", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class ReportBuildResult:
    case_dir: Path
    report: ReportManifest
    state: PipelineState
    reused: bool
    database_preexisted: bool


def read_report_manifest(path: Path) -> ReportManifest:
    return _read_model_file(path, ReportManifest, "report manifest")


def build_report(
    case_dir: Path,
    config: AppConfig,
    *,
    manifest_path: Path,
    decision_path: Path,
    db_path: Path,
    migrations_dir: Path,
    profile_path: Path,
    catalog_dir: Path,
    force: bool = False,
    interrupt_hook: InterruptHook | None = None,
) -> ReportBuildResult:
    """Compile, render, prepare SQLite, commit REPORTED, then activate SQLite."""

    case_dir = case_dir.resolve(strict=True)
    case_manifest, state = verify_case(case_dir)
    if state.current_stage not in {PipelineStage.REVIEWED, PipelineStage.REPORTED}:
        raise InvalidTransitionError("phase 7 report requires an active reviewed stage")
    compilation = compile_learning_analysis(
        manifest_path,
        decision_path,
        config,
        db_path=db_path,
    )
    analysis = compilation.analysis
    if analysis.case_id != case_manifest.case_id:
        raise CaseValidationError("learning analysis belongs to another case")
    reviewed_completion = next(
        item
        for item in state.completed_stages
        if item.stage == PipelineStage.REVIEWED
    )
    reviewed_references = [
        item
        for item in reviewed_completion.artifacts
        if item.schema_id == "grading.schema.json"
    ]
    if (
        len(reviewed_references) != 1
        or reviewed_references[0].sha256
        != analysis.input_manifest.reviewed_grading_sha256
    ):
        raise CaseValidationError("learning input does not reference the active reviewed grading")

    versions = ImplementationVersions(
        code=__version__,
        config=config.config_version,
        model=(
            analysis.decision.provenance.model_identifier
            or analysis.decision.provenance.model_identifier_unavailable_reason
        ),
        prompt=analysis.decision.provenance.prompt_version,
        skill=analysis.decision.provenance.skill_version,
    )
    fingerprint_inputs = [
        {
            "learning_input_manifest_sha256": analysis.learning_input_manifest_sha256,
            "learning_analysis_decision_sha256": analysis.learning_analysis_decision_sha256,
            "reviewed_grading_sha256": analysis.input_manifest.reviewed_grading_sha256,
            "knowledge_catalog_sha256": analysis.knowledge_catalog_sha256,
            "analysis_id": analysis.analysis_id,
        }
    ]
    stage_fingerprint, _ = build_stage_fingerprint(
        stage=PipelineStage.REPORTED,
        model_type=ReportManifest,
        schema_id="report_manifest.schema.json",
        config=config.reporting_fingerprint_payload(),
        versions=versions,
        inputs=fingerprint_inputs,
    )
    existing_report = next(
        (
            item
            for item in state.completed_stages
            if item.stage == PipelineStage.REPORTED
        ),
        None,
    )
    if (
        existing_report is not None
        and existing_report.stage_fingerprint != stage_fingerprint
        and not force
    ):
        raise InvalidTransitionError(
            "reported is already complete with a different fingerprint; use --force"
        )
    claims = _build_claims(analysis)
    summary = _longitudinal_summary(analysis)
    disclaimers = [
        "知识点结论只基于已审阅评分证据，不等同于对学生能力的全面测量。",
        "未达到跨考试可比条件时，报告不会给出进步或退步结论。",
    ]
    final_score = sum(item.result.final_score or 0 for item in analysis.input_manifest.targets)
    max_score = sum(item.result.max_points for item in analysis.input_manifest.targets)
    if max_score <= 0:
        raise CaseValidationError("phase 7 cannot report an assessment with zero maximum score")

    output_dir = case_dir / "reports" / analysis.analysis_id
    analysis_path = output_dir / "learning_analysis.json"
    catalog_path = output_dir / "knowledge_catalog.json"
    markdown_path = output_dir / "report.md"
    html_path = output_dir / "report.html"
    atomic_write_model(analysis_path, analysis, model_type=LearningAnalysis)
    atomic_write_model(catalog_path, analysis.catalog, model_type=KnowledgeCatalog)
    markdown = _render_markdown(
        analysis,
        claims,
        summary,
        disclaimers,
        final_score=final_score,
        max_score=max_score,
    )
    _write_text(markdown_path, markdown)
    _write_text(html_path, _render_html(markdown, analysis, claims, summary, disclaimers))

    catalog_store = catalog_dir / analysis.catalog.catalog_id / (
        analysis.knowledge_catalog_sha256 + ".json"
    )
    _write_or_verify_model(catalog_store, analysis.catalog, KnowledgeCatalog)
    assets = [
        _report_asset(case_dir, "learning-analysis", analysis_path, "application/json"),
        _report_asset(case_dir, "knowledge-catalog", catalog_path, "application/json"),
        _report_asset(case_dir, "markdown-report", markdown_path, "text/markdown"),
        _report_asset(case_dir, "html-report", html_path, "text/html"),
    ]
    report = ReportManifest(
        report_id="report-" + digest_value(
            {"analysis_id": analysis.analysis_id, "stage_fingerprint": stage_fingerprint}
        )[:20],
        case_id=analysis.case_id,
        analysis_id=analysis.analysis_id,
        stage_fingerprint=stage_fingerprint,
        created_at=analysis.created_at,
        final_score=final_score,
        max_score=max_score,
        longitudinal_summary=summary,
        analysis=analysis,
        claims=claims,
        assets=assets,
        disclaimers=disclaimers,
    )
    report_sha256 = sha256(serialize_model(report)).hexdigest()
    profile = read_student_profile(profile_path)
    database_preexisted = prepare_analysis_import(
        db_path,
        migrations_dir,
        profile=profile,
        case_manifest=case_manifest,
        report=report,
        report_manifest_sha256=report_sha256,
    )
    stage_result: StageResult = commit_stage_artifact(
        case_dir,
        stage=PipelineStage.REPORTED,
        artifact_name="report_manifest.json",
        payload=report,
        model_type=ReportManifest,
        schema_id="report_manifest.schema.json",
        config=config.reporting_fingerprint_payload(),
        versions=versions,
        inputs=fingerprint_inputs,
        force=force,
        interrupt_hook=interrupt_hook,
    )
    committed_sha256 = stage_result.artifacts[0].sha256
    if committed_sha256 != report_sha256:
        raise CaseValidationError("committed report manifest hash differs from prepared import")
    activate_analysis_import(db_path, analysis.analysis_id)
    verify_report_assets(case_dir, report)
    return ReportBuildResult(
        case_dir=case_dir,
        report=report,
        state=stage_result.state,
        reused=stage_result.reused,
        database_preexisted=database_preexisted,
    )


def verify_report_assets(case_dir: Path, report: ReportManifest) -> None:
    resolved_case = case_dir.resolve(strict=True)
    if report.case_id != resolved_case.name:
        raise CaseValidationError("report manifest belongs to another case directory")
    for asset in report.assets:
        path = (resolved_case / asset.relative_path).resolve(strict=False)
        if resolved_case not in path.parents:
            raise CaseValidationError(f"report asset escapes case directory: {asset.relative_path}")
        if not path.is_file():
            raise CaseValidationError(f"report asset is missing: {path}")
        digest, size = artifact_digest(path)
        if digest != asset.sha256 or size != asset.size_bytes:
            raise CaseValidationError(f"report asset changed: {path}")
    expected_catalog = sha256(serialize_model(report.analysis.catalog)).hexdigest()
    if expected_catalog != report.analysis.knowledge_catalog_sha256:
        raise CaseValidationError("report embeds a knowledge catalog with a different hash")


def _build_claims(analysis: LearningAnalysis) -> list[ReportClaim]:
    points = {item.point_id: item for item in analysis.catalog.points}
    claims: list[ReportClaim] = []
    for snapshot in analysis.snapshots:
        point = points[snapshot.point_id]
        kind, text = _claim_text(point.name, snapshot)
        claims.append(
            ReportClaim(
                claim_id="claim-" + digest_value(
                    {
                        "analysis_id": analysis.analysis_id,
                        "point_id": snapshot.point_id,
                        "kind": kind.value,
                    }
                )[:20],
                point_id=snapshot.point_id,
                kind=kind,
                strength=snapshot.allowed_claim_strength,
                text=text,
                evidence_ids=snapshot.evidence_ids,
            )
        )
    return claims


def _claim_text(point_name: str, snapshot: MasterySnapshot) -> tuple[ClaimKind, str]:
    score = round(snapshot.performance_index * 100)
    if snapshot.state == MasteryState.INSUFFICIENT_EVIDENCE:
        return (
            ClaimKind.INSUFFICIENT,
            f"“{point_name}”目前只有单项证据（加权表现 {score}%），不足以判断稳定掌握状态。",
        )
    if snapshot.state == MasteryState.SINGLE_EXAM_SIGNAL:
        if snapshot.performance_index >= 0.8:
            kind = ClaimKind.OBSERVED_STRENGTH
            description = "本次考试表现较好"
        elif snapshot.performance_index < 0.6:
            kind = ClaimKind.NEEDS_PRACTICE
            description = "本次考试显示需要练习"
        else:
            kind = ClaimKind.MIXED
            description = "本次考试表现不稳定"
        return kind, f"“{point_name}”{description}（加权表现 {score}%），尚无跨考试结论。"
    if snapshot.state == MasteryState.NEEDS_PRACTICE:
        kind = ClaimKind.NEEDS_PRACTICE
        description = "跨考试证据显示仍需练习"
    elif snapshot.state == MasteryState.CONSISTENT:
        kind = ClaimKind.OBSERVED_STRENGTH
        description = "跨考试证据表现一致"
    else:
        kind = ClaimKind.MIXED
        description = "跨考试证据表现混合"
    trend = {
        TrendStatus.IMPROVING: "，最近可比考试呈改善",
        TrendStatus.DECLINING: "，最近可比考试呈下降",
        TrendStatus.STABLE: "，最近可比考试基本稳定",
        TrendStatus.NOT_COMPARABLE: "",
    }[snapshot.trend]
    return kind, f"“{point_name}”{description}（加权表现 {score}%）{trend}。"


def _longitudinal_summary(analysis: LearningAnalysis) -> str:
    longitudinal = [
        item
        for item in analysis.snapshots
        if item.allowed_claim_strength == ClaimStrength.LONGITUDINAL
    ]
    comparable = [
        item for item in longitudinal if item.trend != TrendStatus.NOT_COMPARABLE
    ]
    if not longitudinal:
        return "当前没有达到跨考试结论所需的独立证据；本报告仅陈述单题或单次考试信号。"
    if not comparable:
        return "已有部分跨考试知识点证据，但考试日期精度或时间顺序不足以计算趋势。"
    improving = sum(item.trend == TrendStatus.IMPROVING for item in comparable)
    declining = sum(item.trend == TrendStatus.DECLINING for item in comparable)
    stable = sum(item.trend == TrendStatus.STABLE for item in comparable)
    return f"可比较知识点共 {len(comparable)} 个：改善 {improving}、下降 {declining}、稳定 {stable}。"


def _render_markdown(
    analysis: LearningAnalysis,
    claims: list[ReportClaim],
    summary: str,
    disclaimers: list[str],
    *,
    final_score: float,
    max_score: float,
) -> str:
    points = {item.point_id: item for item in analysis.catalog.points}
    lines = [
        f"# {analysis.input_manifest.metadata.title or '学习分析报告'}",
        "",
        f"- 科目：{analysis.input_manifest.metadata.subject}",
        f"- 得分：{final_score:g} / {max_score:g}",
        f"- 考试日期：{analysis.input_manifest.metadata.occurred_at or '未知'}",
        "",
        "## 知识点证据",
        "",
    ]
    if claims:
        snapshots = {item.point_id: item for item in analysis.snapshots}
        for claim in claims:
            snapshot = snapshots[claim.point_id]
            lines.extend(
                [
                    f"### {points[claim.point_id].name}",
                    "",
                    claim.text,
                    "",
                    f"- 结论强度：`{claim.strength.value}`",
                    f"- 证据数：{snapshot.evidence_count}",
                    f"- 独立评分目标：{snapshot.independent_target_count}",
                    f"- 独立考试：{snapshot.independent_exam_count}",
                    f"- 表现指数：{snapshot.performance_index:.3f}",
                    "",
                    "证据明细：",
                    "",
                    *_evidence_markdown_lines(analysis, claim),
                    "",
                ]
            )
    else:
        lines.extend(["没有获得可归因的知识点证据。", ""])
    lines.extend(["## 历史概览", "", summary, "", "## 建议", ""])
    if analysis.recommendations:
        for item in sorted(analysis.recommendations, key=lambda value: value.priority):
            names = "、".join(points[point_id].name for point_id in item.point_ids)
            lines.append(
                f"- P{item.priority} · {names} · {item.practice_type.value} / "
                f"{item.difficulty.value}：{item.action}"
            )
    else:
        lines.append("- 本次没有形成证据充分的定向建议。")
    if analysis.unmapped_rubric_refs:
        lines.extend(
            [
                "",
                "## 未映射评分标准",
                "",
                "、".join(f"`{item}`" for item in analysis.unmapped_rubric_refs),
            ]
        )
    lines.extend(["", "## 说明", ""])
    lines.extend(f"- {item}" for item in disclaimers)
    return "\n".join(lines) + "\n"


def _render_html(
    markdown: str,
    analysis: LearningAnalysis,
    claims: list[ReportClaim],
    summary: str,
    disclaimers: list[str],
) -> str:
    points = {item.point_id: item for item in analysis.catalog.points}
    snapshots = {item.point_id: item for item in analysis.snapshots}
    claim_html = "".join(
        "<section><h2>"
        + escape(points[item.point_id].name)
        + "</h2><p>"
        + escape(item.text)
        + "</p><ul><li>结论强度：<code>"
        + escape(item.strength.value)
        + "</code></li><li>证据数："
        + str(snapshots[item.point_id].evidence_count)
        + "</li><li>独立评分目标："
        + str(snapshots[item.point_id].independent_target_count)
        + "</li><li>独立考试："
        + str(snapshots[item.point_id].independent_exam_count)
        + "</li><li>表现指数："
        + f"{snapshots[item.point_id].performance_index:.3f}"
        + "</li></ul><h3>证据明细</h3><ul class=\"evidence\">"
        + _evidence_html_items(analysis, item)
        + "</ul></section>"
        for item in claims
    ) or "<p>没有获得可归因的知识点证据。</p>"
    disclaimer_html = "".join(f"<li>{escape(item)}</li>" for item in disclaimers)
    return (
        "<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<title>学习分析报告</title><style>body{max-width:900px;margin:2rem auto;"
        "padding:0 1rem;font:16px/1.65 system-ui,sans-serif;color:#17202a}"
        "section{border-top:1px solid #ddd;padding-top:.5rem}.evidence{color:#566573;"
        "font-size:.9rem}code{word-break:break-all}</style></head><body>"
        f"<h1>{escape(analysis.input_manifest.metadata.title or '学习分析报告')}</h1>"
        f"<p>{escape(summary)}</p>{claim_html}<h2>说明</h2><ul>{disclaimer_html}</ul>"
        "<details><summary>机器可读 Markdown</summary><pre>"
        f"{escape(markdown)}</pre></details></body></html>"
    )


def _report_asset(
    case_dir: Path, name: str, path: Path, media_type: str
) -> ReportAsset:
    digest, size = artifact_digest(path)
    return ReportAsset(
        name=name,
        relative_path=path.relative_to(case_dir).as_posix(),
        sha256=digest,
        size_bytes=size,
        media_type=media_type,
    )


def _evidence_markdown_lines(
    analysis: LearningAnalysis, claim: ReportClaim
) -> list[str]:
    return [
        "- " + label + "；来源：" + "、".join(f"[{index + 1}](<{href}>)" for index, href in enumerate(hrefs))
        for label, hrefs in _evidence_rows(analysis, claim)
    ]


def _evidence_html_items(analysis: LearningAnalysis, claim: ReportClaim) -> str:
    return "".join(
        "<li>"
        + escape(label)
        + "；来源："
        + "、".join(
            f'<a href="{escape(href, quote=True)}">{index + 1}</a>'
            for index, href in enumerate(hrefs)
        )
        + "</li>"
        for label, hrefs in _evidence_rows(analysis, claim)
    )


def _evidence_rows(
    analysis: LearningAnalysis, claim: ReportClaim
) -> list[tuple[str, list[str]]]:
    current = {item.evidence_id: item for item in analysis.evidence}
    historical = {item.evidence_id: item for item in analysis.historical_evidence}
    targets = {item.target.target_id: item for item in analysis.input_manifest.targets}
    rows: list[tuple[str, list[str]]] = []
    for evidence_id in claim.evidence_ids:
        item = current.get(evidence_id)
        if item is not None:
            target = targets[item.target_id]
            source_paths = sorted(
                {
                    response.crop.relative_path
                    for response in target.target.responses
                    if response.submission_item_id in item.submission_item_ids
                }
            )
            hrefs = [f"../../{path}" for path in source_paths]
            rows.append(
                (
                    f"{item.printed_label}；target={item.target_id}；rubric={item.rubric_ref}；"
                    f"outcome={item.outcome:.3f}；evidence={item.evidence_id}",
                    hrefs,
                )
            )
            continue
        old = historical[evidence_id]
        rows.append(
            (
                f"{old.printed_label}；target={old.target_id}；rubric={old.rubric_ref}；"
                f"outcome={old.outcome:.3f}；case={old.case_id}；evidence={old.evidence_id}",
                [f"../../../{old.case_id}/{path}" for path in old.source_relative_paths],
            )
        )
    return rows


def _write_text(path: Path, content: str) -> None:
    raw = content.encode("utf-8")

    def validate(temporary: Path) -> None:
        if temporary.read_bytes() != raw:
            raise CaseValidationError("temporary text asset differs from requested content")

    atomic_write_bytes(path, raw, validator=validate)


def _write_or_verify_model(
    path: Path, model: ModelT, model_type: type[ModelT]
) -> None:
    expected = serialize_model(model)
    if path.exists():
        persisted = _read_model_file(path, model_type, "persisted knowledge catalog")
        if serialize_model(persisted) != expected:
            raise CaseValidationError(f"conflicting knowledge catalog at {path}")
        return
    atomic_write_model(path, model, model_type=model_type)


def _read_model_file(path: Path, model_type: type[ModelT], label: str) -> ModelT:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise CaseValidationError(f"cannot read {label} {path}: {exc}") from exc
    return validate_json(model_type, raw)
