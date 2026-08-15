"""SQLite migrations and the rebuildable phase 7 query projection."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Iterable, Iterator

from student_analyze.atomic import atomic_write_model
from student_analyze.errors import CaseValidationError, ConfigurationError
from student_analyze.learning_models import (
    KnowledgeCatalog,
    LearningAnalysis,
    ReportManifest,
    StudentProfile,
)
from student_analyze.models import CaseManifest, PipelineStage, PipelineState
from student_analyze.validation import validate_json


MIGRATION_NAME = re.compile(r"^(?P<version>\d{4})_[a-z0-9_]+\.sql$")


@dataclass(frozen=True, slots=True)
class HistoryEvidence:
    analysis_id: str
    case_id: str
    point_id: str
    evidence_id: str
    question_id: str
    target_id: str
    rubric_ref: str
    printed_label: str
    outcome: float
    effective_weight: float
    occurred_at: str | None
    occurred_at_precision: str
    source_relative_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DatabaseStatus:
    path: Path
    schema_version: int
    active_analyses: int
    prepared_analyses: int
    exams: int
    evidence: int
    snapshots: int
    logical_sha256: str


def initialize_student_profile(
    profile_path: Path,
    *,
    display_name: str | None = None,
    student_profile_id: str = "student-default",
) -> tuple[StudentProfile, bool]:
    """Create or reuse the single local student profile JSON source."""

    if profile_path.exists():
        profile = validate_json(StudentProfile, profile_path.read_bytes())
        if display_name is not None and profile.display_name != display_name:
            raise ConfigurationError(
                "student profile already exists with a different display name"
            )
        return profile, True
    profile = StudentProfile(
        student_profile_id=student_profile_id,
        display_name=display_name,
        created_at=datetime.now(UTC),
    )
    atomic_write_model(profile_path, profile, model_type=StudentProfile)
    return profile, False


def read_student_profile(profile_path: Path) -> StudentProfile:
    try:
        raw = profile_path.read_bytes()
    except OSError as exc:
        raise ConfigurationError(f"cannot read student profile {profile_path}: {exc}") from exc
    return validate_json(StudentProfile, raw)


@contextmanager
def connect_database(db_path: Path) -> Iterator[sqlite3.Connection]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    try:
        yield connection
    finally:
        connection.close()


def apply_migrations(db_path: Path, migrations_dir: Path) -> int:
    migrations = _discover_migrations(migrations_dir)
    if not migrations:
        raise ConfigurationError(f"no SQLite migrations found in {migrations_dir}")
    with connect_database(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migration (
                version INTEGER PRIMARY KEY,
                filename TEXT NOT NULL UNIQUE,
                sha256 TEXT NOT NULL,
                applied_at TEXT NOT NULL
            )
            """
        )
        connection.commit()
        applied = {
            int(row["version"]): (str(row["filename"]), str(row["sha256"]))
            for row in connection.execute(
                "SELECT version, filename, sha256 FROM schema_migration ORDER BY version"
            )
        }
        for version, path, content, digest in migrations:
            if version in applied:
                filename, known_digest = applied[version]
                if filename != path.name or known_digest != digest:
                    raise CaseValidationError(
                        f"applied migration {version} differs from {path.name}"
                    )
                continue
            expected = max(applied, default=0) + 1
            if version != expected:
                raise CaseValidationError(
                    f"migration sequence is not contiguous: expected {expected:04d}"
                )
            applied_at = datetime.now(UTC).isoformat()
            escaped_name = path.name.replace("'", "''")
            script = (
                "BEGIN IMMEDIATE;\n"
                f"{content}\n"
                "INSERT INTO schema_migration(version, filename, sha256, applied_at) "
                f"VALUES ({version}, '{escaped_name}', '{digest}', '{applied_at}');\n"
                f"PRAGMA user_version = {version};\n"
                "COMMIT;\n"
            )
            try:
                connection.executescript(script)
            except sqlite3.DatabaseError as exc:
                connection.rollback()
                raise CaseValidationError(
                    f"cannot apply migration {path.name}: {exc}"
                ) from exc
            applied[version] = (path.name, digest)
        user_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if user_version != max(applied):
            raise CaseValidationError("SQLite user_version differs from migration ledger")
        return user_version


def prepare_analysis_import(
    db_path: Path,
    migrations_dir: Path,
    *,
    profile: StudentProfile,
    case_manifest: CaseManifest,
    report: ReportManifest,
    report_manifest_sha256: str,
) -> bool:
    """Insert one complete analysis as prepared; return True when it already exists."""

    apply_migrations(db_path, migrations_dir)
    analysis = report.analysis
    if analysis.case_id != case_manifest.case_id:
        raise CaseValidationError("learning analysis belongs to another case")
    with connect_database(db_path) as connection:
        existing = connection.execute(
            "SELECT activation_status FROM analysis_run WHERE analysis_id = ?",
            (analysis.analysis_id,),
        ).fetchone()
        if existing is not None:
            _verify_existing_analysis(connection, analysis, report_manifest_sha256)
            return True
        try:
            connection.execute("BEGIN IMMEDIATE")
            _insert_profile(connection, profile)
            connection.execute(
                "INSERT OR IGNORE INTO exam(case_id, student_profile_id, input_fingerprint, created_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    case_manifest.case_id,
                    profile.student_profile_id,
                    case_manifest.input_fingerprint,
                    case_manifest.created_at.isoformat(),
                ),
            )
            _insert_catalog(connection, analysis)
            _insert_analysis_run(connection, analysis)
            connection.execute(
                "INSERT INTO import_run(import_id, analysis_id, activation_status, prepared_at) "
                "VALUES (?, ?, 'prepared', ?)",
                (
                    f"import-{analysis.analysis_id.removeprefix('analysis-')}",
                    analysis.analysis_id,
                    datetime.now(UTC).isoformat(),
                ),
            )
            _insert_exam_revision(connection, report)
            _insert_artifacts(connection, analysis, report_manifest_sha256)
            _insert_document_assets(connection, analysis, case_manifest)
            _insert_assessment_facts(connection, analysis)
            _insert_knowledge_results(connection, analysis)
            _insert_review_events(connection, analysis)
            connection.execute(
                "INSERT INTO report_record(report_id, analysis_id, report_manifest_sha256, created_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    report.report_id,
                    analysis.analysis_id,
                    report_manifest_sha256,
                    report.created_at.isoformat(),
                ),
            )
            connection.commit()
        except sqlite3.DatabaseError as exc:
            connection.rollback()
            raise CaseValidationError(f"cannot import phase 7 analysis: {exc}") from exc
    return False


def activate_analysis_import(db_path: Path, analysis_id: str) -> None:
    with connect_database(db_path) as connection:
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT case_id FROM analysis_run WHERE analysis_id = ?", (analysis_id,)
            ).fetchone()
            if row is None:
                raise CaseValidationError(f"unknown prepared analysis {analysis_id}")
            replaced = [
                str(item["analysis_id"])
                for item in connection.execute(
                    "SELECT analysis_id FROM analysis_run "
                    "WHERE case_id = ? AND activation_status = 'active' AND analysis_id <> ?",
                    (row["case_id"], analysis_id),
                )
            ]
            if replaced:
                placeholders = ", ".join("?" for _ in replaced)
                connection.execute(
                    f"UPDATE analysis_run SET activation_status = 'superseded' "
                    f"WHERE analysis_id IN ({placeholders})",
                    replaced,
                )
                connection.execute(
                    f"UPDATE import_run SET activation_status = 'superseded' "
                    f"WHERE analysis_id IN ({placeholders})",
                    replaced,
                )
            result = connection.execute(
                "UPDATE analysis_run SET activation_status = 'active' WHERE analysis_id = ?",
                (analysis_id,),
            )
            connection.execute(
                "UPDATE import_run SET activation_status = 'active', activated_at = ? "
                "WHERE analysis_id = ?",
                (datetime.now(UTC).isoformat(), analysis_id),
            )
            if result.rowcount != 1:
                raise CaseValidationError(f"cannot activate analysis {analysis_id}")
            connection.commit()
        except sqlite3.DatabaseError as exc:
            connection.rollback()
            raise CaseValidationError(f"cannot activate phase 7 analysis: {exc}") from exc


def load_active_history(
    db_path: Path,
    *,
    subject: str,
    catalog_sha256: str,
) -> list[HistoryEvidence]:
    if not db_path.exists():
        return []
    with connect_database(db_path) as connection:
        try:
            rows = connection.execute(
                """
                SELECT ar.analysis_id, ar.case_id, ke.point_id, ke.evidence_id,
                       ke.question_id, ke.target_id, ke.rubric_ref, gt.printed_label,
                       ke.outcome, ke.effective_weight,
                       er.occurred_at, er.occurred_at_precision,
                       group_concat(
                           DISTINCT hex(CAST(ao.source_relative_path AS BLOB))
                       ) AS source_paths_hex
                  FROM analysis_run ar
                  JOIN exam_revision er ON er.analysis_id = ar.analysis_id
                  JOIN knowledge_evidence ke ON ke.analysis_id = ar.analysis_id
                  JOIN grading_target gt
                    ON gt.analysis_id = ke.analysis_id AND gt.target_id = ke.target_id
                  JOIN grading_target_answer gta
                    ON gta.analysis_id = ke.analysis_id AND gta.target_id = ke.target_id
                  JOIN answer_observation ao
                    ON ao.analysis_id = gta.analysis_id
                   AND ao.submission_item_id = gta.submission_item_id
                 WHERE ar.activation_status = 'active'
                   AND ar.catalog_sha256 = ?
                   AND lower(er.subject) = lower(?)
                 GROUP BY ar.analysis_id, ar.case_id, ke.point_id, ke.evidence_id,
                          ke.question_id, ke.target_id, ke.rubric_ref, gt.printed_label,
                          ke.outcome, ke.effective_weight, er.occurred_at,
                          er.occurred_at_precision
                 ORDER BY ar.created_at, ke.evidence_id
                """,
                (catalog_sha256, subject),
            ).fetchall()
        except sqlite3.DatabaseError:
            return []
    return [
        HistoryEvidence(
            analysis_id=str(row["analysis_id"]),
            case_id=str(row["case_id"]),
            point_id=str(row["point_id"]),
            evidence_id=str(row["evidence_id"]),
            question_id=str(row["question_id"]),
            target_id=str(row["target_id"]),
            rubric_ref=str(row["rubric_ref"]),
            printed_label=str(row["printed_label"]),
            outcome=float(row["outcome"]),
            effective_weight=float(row["effective_weight"]),
            occurred_at=row["occurred_at"],
            occurred_at_precision=str(row["occurred_at_precision"]),
            source_relative_paths=tuple(
                bytes.fromhex(item).decode("utf-8")
                for item in str(row["source_paths_hex"]).split(",")
            ),
        )
        for row in rows
    ]


def verify_database(db_path: Path, migrations_dir: Path) -> DatabaseStatus:
    schema_version = apply_migrations(db_path, migrations_dir)
    with connect_database(db_path) as connection:
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity != "ok":
            raise CaseValidationError(f"SQLite integrity_check failed: {integrity}")
        foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
        if foreign_keys:
            raise CaseValidationError("SQLite foreign_key_check found violations")
        counts = {
            "active": int(
                connection.execute(
                    "SELECT count(*) FROM analysis_run WHERE activation_status = 'active'"
                ).fetchone()[0]
            ),
            "prepared": int(
                connection.execute(
                    "SELECT count(*) FROM analysis_run WHERE activation_status = 'prepared'"
                ).fetchone()[0]
            ),
            "exams": int(connection.execute("SELECT count(*) FROM exam").fetchone()[0]),
            "evidence": int(
                connection.execute("SELECT count(*) FROM knowledge_evidence").fetchone()[0]
            ),
            "snapshots": int(
                connection.execute("SELECT count(*) FROM mastery_snapshot").fetchone()[0]
            ),
        }
        logical_sha256 = _logical_digest(connection)
    return DatabaseStatus(
        path=db_path.resolve(),
        schema_version=schema_version,
        active_analyses=counts["active"],
        prepared_analyses=counts["prepared"],
        exams=counts["exams"],
        evidence=counts["evidence"],
        snapshots=counts["snapshots"],
        logical_sha256=logical_sha256,
    )


def rebuild_database(
    db_path: Path,
    migrations_dir: Path,
    *,
    profile_path: Path,
    cases_dir: Path,
) -> DatabaseStatus:
    """Rebuild into a sibling file and atomically replace the query projection."""

    profile = read_student_profile(profile_path)
    target = db_path.resolve(strict=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.rebuild")
    if temporary.exists():
        temporary.unlink()
    expected_logical_sha256: str | None = None
    if target.exists():
        expected_logical_sha256 = verify_database(target, migrations_dir).logical_sha256
    reports_by_analysis: dict[
        str, tuple[ReportManifest, CaseManifest, str, bool]
    ] = {}
    for state_path in sorted(cases_dir.glob("case-*/pipeline_state.json")):
        case_dir = state_path.parent
        from student_analyze.pipeline import verify_case
        from student_analyze.reporting import verify_report_assets

        manifest, state = verify_case(case_dir)
        active_completion = next(
            (item for item in state.completed_stages if item.stage == PipelineStage.REPORTED),
            None,
        )
        active_run_id = active_completion.active_run_id if active_completion else None
        for run in state.run_history:
            if run.stage != PipelineStage.REPORTED:
                continue
            references = [
                item
                for item in run.artifacts
                if item.schema_id == "report_manifest.schema.json"
            ]
            if len(references) != 1:
                raise CaseValidationError("reported run must have one report manifest")
            reference = references[0]
            report = validate_json(
                ReportManifest, (case_dir / reference.relative_path).read_bytes()
            )
            verify_report_assets(case_dir, report)
            active = run.run_id == active_run_id
            existing = reports_by_analysis.get(report.analysis_id)
            candidate = (report, manifest, reference.sha256, active)
            if existing is not None:
                if existing[2] != reference.sha256:
                    raise CaseValidationError(
                        "one analysis_id has conflicting report artifacts"
                    )
                candidate = (*existing[:3], existing[3] or active)
            reports_by_analysis[report.analysis_id] = candidate
    reports = sorted(
        reports_by_analysis.values(),
        key=lambda item: (item[0].analysis.created_at, item[0].analysis_id),
    )
    try:
        for report, manifest, digest, _ in reports:
            prepare_analysis_import(
                temporary,
                migrations_dir,
                profile=profile,
                case_manifest=manifest,
                report=report,
                report_manifest_sha256=digest,
            )
        for report, _, _, active in [
            *[item for item in reports if not item[3]],
            *[item for item in reports if item[3]],
        ]:
            activate_analysis_import(temporary, report.analysis_id)
        status = verify_database(temporary, migrations_dir)
        if (
            expected_logical_sha256 is not None
            and status.logical_sha256 != expected_logical_sha256
        ):
            raise CaseValidationError(
                "rebuilt database logical digest differs from the current projection"
            )
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return DatabaseStatus(path=target, **{name: getattr(status, name) for name in (
        "schema_version", "active_analyses", "prepared_analyses", "exams", "evidence", "snapshots",
        "logical_sha256"
    )})


def _logical_digest(connection: sqlite3.Connection) -> str:
    """Hash committed query facts while excluding import timestamps and prepared rows."""

    queries = {
        "analysis_run": (
            "SELECT analysis_id, case_id, catalog_sha256, learning_input_manifest_sha256, "
            "learning_analysis_decision_sha256, reviewed_grading_sha256, "
            "mapping_policy_version, mastery_algorithm_version, report_policy_version, "
            "created_at, activation_status FROM analysis_run "
            "WHERE activation_status <> 'prepared' ORDER BY analysis_id"
        ),
        "exam_revision": (
            "SELECT er.* FROM exam_revision er JOIN analysis_run ar USING (analysis_id) "
            "WHERE ar.activation_status <> 'prepared' ORDER BY er.analysis_id"
        ),
        "rubric_evaluation": (
            "SELECT tbl.* FROM rubric_evaluation tbl "
            "JOIN analysis_run ar USING (analysis_id) "
            "WHERE ar.activation_status <> 'prepared' "
            "ORDER BY tbl.analysis_id, tbl.target_id, tbl.rubric_ref"
        ),
        "target_knowledge_mapping": (
            "SELECT tbl.* FROM target_knowledge_mapping tbl "
            "JOIN analysis_run ar USING (analysis_id) "
            "WHERE ar.activation_status <> 'prepared' "
            "ORDER BY tbl.analysis_id, tbl.mapping_id"
        ),
        "knowledge_evidence": (
            "SELECT tbl.* FROM knowledge_evidence tbl "
            "JOIN analysis_run ar USING (analysis_id) "
            "WHERE ar.activation_status <> 'prepared' "
            "ORDER BY tbl.analysis_id, tbl.evidence_id"
        ),
        "mastery_snapshot": (
            "SELECT tbl.* FROM mastery_snapshot tbl "
            "JOIN analysis_run ar USING (analysis_id) "
            "WHERE ar.activation_status <> 'prepared' "
            "ORDER BY tbl.analysis_id, tbl.snapshot_id"
        ),
        "report_record": (
            "SELECT tbl.* FROM report_record tbl "
            "JOIN analysis_run ar USING (analysis_id) "
            "WHERE ar.activation_status <> 'prepared' "
            "ORDER BY tbl.analysis_id"
        ),
    }
    payload = {
        name: [list(row) for row in connection.execute(query).fetchall()]
        for name, query in queries.items()
    }
    return sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _discover_migrations(
    migrations_dir: Path,
) -> list[tuple[int, Path, str, str]]:
    discovered: list[tuple[int, Path, str, str]] = []
    try:
        paths = sorted(migrations_dir.glob("*.sql"))
    except OSError as exc:
        raise ConfigurationError(f"cannot list migrations {migrations_dir}: {exc}") from exc
    for path in paths:
        match = MIGRATION_NAME.fullmatch(path.name)
        if match is None:
            raise ConfigurationError(f"invalid migration filename: {path.name}")
        content = path.read_text(encoding="utf-8")
        digest = sha256(content.encode("utf-8")).hexdigest()
        discovered.append((int(match.group("version")), path, content, digest))
    versions = [item[0] for item in discovered]
    if len(versions) != len(set(versions)):
        raise ConfigurationError("migration versions must be unique")
    return discovered


def _insert_profile(connection: sqlite3.Connection, profile: StudentProfile) -> None:
    connection.execute(
        "INSERT OR IGNORE INTO student_profile(student_profile_id, display_name, created_at) "
        "VALUES (?, ?, ?)",
        (profile.student_profile_id, profile.display_name, profile.created_at.isoformat()),
    )
    row = connection.execute(
        "SELECT student_profile_id, display_name FROM student_profile WHERE singleton_key = 1"
    ).fetchone()
    if row is None or row["student_profile_id"] != profile.student_profile_id:
        raise CaseValidationError("database contains a different singleton student profile")
    if row["display_name"] != profile.display_name:
        raise CaseValidationError("database student profile differs from its JSON source")


def _insert_catalog(connection: sqlite3.Connection, analysis: LearningAnalysis) -> None:
    if analysis.input_manifest.knowledge_catalog_sha256 != analysis.knowledge_catalog_sha256:
        _insert_catalog_version(
            connection,
            analysis.input_manifest.catalog,
            analysis.input_manifest.knowledge_catalog_sha256,
        )
    _insert_catalog_version(
        connection,
        analysis.catalog,
        analysis.knowledge_catalog_sha256,
    )


def _insert_catalog_version(
    connection: sqlite3.Connection,
    catalog: KnowledgeCatalog,
    catalog_sha256: str,
) -> None:
    existing = connection.execute(
        "SELECT catalog_id, subject, version, parent_catalog_sha256 "
        "FROM knowledge_scheme_version WHERE catalog_sha256 = ?",
        (catalog_sha256,),
    ).fetchone()
    if existing is not None:
        expected = (
            catalog.catalog_id,
            catalog.subject,
            catalog.version,
            catalog.parent_catalog_sha256,
        )
        if tuple(existing) != expected:
            raise CaseValidationError("stored knowledge catalog metadata differs from artifact")
        return
    connection.execute(
        "INSERT INTO knowledge_scheme_version "
        "(catalog_sha256, catalog_id, subject, version, parent_catalog_sha256, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            catalog_sha256,
            catalog.catalog_id,
            catalog.subject,
            catalog.version,
            catalog.parent_catalog_sha256,
            catalog.created_at.isoformat(),
        ),
    )
    for point in catalog.points:
        connection.execute(
            "INSERT OR IGNORE INTO knowledge_point(point_id, subject, created_catalog_sha256) "
            "VALUES (?, ?, ?)",
            (point.point_id, catalog.subject, catalog_sha256),
        )
        stored = connection.execute(
            "SELECT subject FROM knowledge_point WHERE point_id = ?", (point.point_id,)
        ).fetchone()
        if stored is None or str(stored["subject"]).casefold() != catalog.subject.casefold():
            raise CaseValidationError(
                f"knowledge point ID is already used by another subject: {point.point_id}"
            )
    for point in catalog.points:
        connection.execute(
            "INSERT INTO knowledge_point_revision "
            "(catalog_sha256, point_id, name, description, parent_point_id, status) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                catalog_sha256,
                point.point_id,
                point.name,
                point.description,
                point.parent_point_id,
                point.status.value,
            ),
        )
        for alias in point.aliases:
            connection.execute(
                "INSERT INTO knowledge_alias(catalog_sha256, point_id, alias) VALUES (?, ?, ?)",
                (catalog_sha256, point.point_id, alias),
            )
    for relation in catalog.relations:
        connection.execute(
            "INSERT INTO knowledge_relation "
            "(catalog_sha256, relation_id, source_point_id, target_point_id, relation_type) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                catalog_sha256,
                relation.relation_id,
                relation.source_point_id,
                relation.target_point_id,
                relation.relation_type.value,
            ),
        )
    for change in catalog.changes:
        connection.execute(
            "INSERT INTO knowledge_change "
            "(catalog_sha256, change_id, change_type, rationale, human_confirmed) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                catalog_sha256,
                change.change_id,
                change.change_type.value,
                change.rationale,
                int(change.human_confirmed),
            ),
        )
        for direction, point_ids in (
            ("source", change.source_point_ids),
            ("target", change.target_point_ids),
        ):
            for point_id in point_ids:
                connection.execute(
                    "INSERT INTO knowledge_change_mapping "
                    "(catalog_sha256, change_id, direction, point_id) VALUES (?, ?, ?, ?)",
                    (catalog_sha256, change.change_id, direction, point_id),
                )


def _insert_analysis_run(connection: sqlite3.Connection, analysis: LearningAnalysis) -> None:
    connection.execute(
        "INSERT INTO analysis_run "
        "(analysis_id, case_id, catalog_sha256, learning_input_manifest_sha256, "
        "learning_analysis_decision_sha256, reviewed_grading_sha256, "
        "mapping_policy_version, mastery_algorithm_version, report_policy_version, "
        "created_at, activation_status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'prepared')",
        (
            analysis.analysis_id,
            analysis.case_id,
            analysis.knowledge_catalog_sha256,
            analysis.learning_input_manifest_sha256,
            analysis.learning_analysis_decision_sha256,
            analysis.input_manifest.reviewed_grading_sha256,
            analysis.mapping_policy_version,
            analysis.mastery_algorithm_version,
            analysis.report_policy_version,
            analysis.created_at.isoformat(),
        ),
    )


def _insert_exam_revision(connection: sqlite3.Connection, report: ReportManifest) -> None:
    metadata = report.analysis.input_manifest.metadata
    connection.execute(
        "INSERT INTO exam_revision "
        "(analysis_id, case_id, subject, title, grade_level, term, occurred_at, "
        "occurred_at_precision, metadata_source, metadata_confidence, final_score, max_score) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            report.analysis_id,
            report.case_id,
            metadata.subject,
            metadata.title,
            metadata.grade_level,
            metadata.term,
            metadata.occurred_at,
            metadata.occurred_at_precision.value,
            metadata.source.value,
            metadata.confidence,
            report.final_score,
            report.max_score,
        ),
    )


def _insert_artifacts(
    connection: sqlite3.Connection,
    analysis: LearningAnalysis,
    report_manifest_sha256: str,
) -> None:
    artifacts = {
        "exam_master": analysis.input_manifest.exam_master_sha256,
        "submission": analysis.input_manifest.submission_sha256,
        "reviewed_grading": analysis.input_manifest.reviewed_grading_sha256,
        "exam_metadata": analysis.input_manifest.exam_metadata_sha256,
        "learning_input_manifest": analysis.learning_input_manifest_sha256,
        "learning_analysis_decision": analysis.learning_analysis_decision_sha256,
        "knowledge_catalog": analysis.knowledge_catalog_sha256,
        "report_manifest": report_manifest_sha256,
    }
    for kind, digest in artifacts.items():
        connection.execute(
            "INSERT INTO artifact_record(analysis_id, artifact_kind, sha256) VALUES (?, ?, ?)",
            (analysis.analysis_id, kind, digest),
        )


def _insert_document_assets(
    connection: sqlite3.Connection,
    analysis: LearningAnalysis,
    case_manifest: CaseManifest,
) -> None:
    for asset in case_manifest.source_assets:
        connection.execute(
            "INSERT INTO document_asset "
            "(analysis_id, asset_id, relative_path, sha256, size_bytes, media_type) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                analysis.analysis_id,
                asset.asset_id,
                asset.relative_path,
                asset.sha256,
                asset.size_bytes,
                asset.media_type,
            ),
        )


def _insert_assessment_facts(
    connection: sqlite3.Connection, analysis: LearningAnalysis
) -> None:
    grouped: dict[str, list] = {}
    for item in analysis.input_manifest.targets:
        grouped.setdefault(item.target.question_id, []).append(item)
    for question_id, items in grouped.items():
        first = items[0].target
        connection.execute(
            "INSERT INTO question "
            "(analysis_id, case_id, question_id, version_id, printed_label, question_type, max_points) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                analysis.analysis_id,
                analysis.case_id,
                question_id,
                first.version_id,
                first.printed_label.split("(", 1)[0],
                first.question_type.value,
                sum(item.target.max_points for item in items),
            ),
        )
    inserted_answers: set[str] = set()
    for item in analysis.input_manifest.targets:
        target = item.target
        result = item.result
        for response in target.responses:
            if response.submission_item_id in inserted_answers:
                continue
            inserted_answers.add(response.submission_item_id)
            connection.execute(
                "INSERT INTO answer_observation "
                "(analysis_id, submission_item_id, observed_content, normalized_answer, "
                "is_blank, source_relative_path, source_sha256) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    analysis.analysis_id,
                    response.submission_item_id,
                    response.observed_content,
                    response.normalized_answer,
                    int(response.is_blank),
                    response.crop.relative_path,
                    response.crop.sha256,
                ),
            )
        connection.execute(
            "INSERT INTO grading_target "
            "(analysis_id, target_id, question_id, part_id, printed_label, method, "
            "final_score, max_points, confidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                analysis.analysis_id,
                target.target_id,
                target.question_id,
                target.part_id,
                target.printed_label,
                result.method.value,
                result.final_score,
                result.max_points,
                result.confidence,
            ),
        )
        for response in target.responses:
            connection.execute(
                "INSERT INTO grading_target_answer(analysis_id, target_id, submission_item_id) "
                "VALUES (?, ?, ?)",
                (analysis.analysis_id, target.target_id, response.submission_item_id),
            )
        rubric_points = {rubric.ref: rubric.points for rubric in target.rubric}
        for evaluation in result.rubric_evaluations:
            max_points = rubric_points[evaluation.rubric_ref]
            if max_points is None or max_points <= 0 or evaluation.awarded_points is None:
                raise CaseValidationError("phase 7 requires positive, fully scored rubric items")
            connection.execute(
                "INSERT INTO rubric_evaluation "
                "(analysis_id, target_id, rubric_ref, status, awarded_points, max_points, rationale) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    analysis.analysis_id,
                    target.target_id,
                    evaluation.rubric_ref,
                    evaluation.status.value,
                    evaluation.awarded_points,
                    max_points,
                    evaluation.rationale,
                ),
            )
        for index, diagnosis in enumerate(result.error_diagnoses):
            connection.execute(
                "INSERT INTO error_diagnosis "
                "(analysis_id, target_id, diagnosis_index, error_type, diagnosis, "
                "rubric_refs_json, evidence_item_ids_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    analysis.analysis_id,
                    target.target_id,
                    index,
                    diagnosis.error_type.value,
                    diagnosis.diagnosis,
                    _json(item.value if hasattr(item, "value") else item for item in diagnosis.rubric_refs),
                    _json(diagnosis.evidence_item_ids),
                ),
            )


def _insert_knowledge_results(
    connection: sqlite3.Connection, analysis: LearningAnalysis
) -> None:
    for mapping in analysis.decision.mappings:
        connection.execute(
            "INSERT INTO target_knowledge_mapping "
            "(analysis_id, mapping_id, target_id, rubric_ref, status, point_id, role, "
            "attribution_scopes_json, weight, confidence, rationale, unmapped_reason) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                analysis.analysis_id,
                mapping.mapping_id,
                mapping.target_id,
                mapping.rubric_ref,
                mapping.status.value,
                mapping.point_id,
                mapping.role.value,
                _json(item.value for item in mapping.attribution_scopes),
                mapping.weight,
                mapping.confidence,
                mapping.rationale,
                mapping.unmapped_reason,
            ),
        )
    for evidence in analysis.evidence:
        connection.execute(
            "INSERT INTO knowledge_evidence "
            "(analysis_id, evidence_id, mapping_id, point_id, question_id, target_id, "
            "rubric_ref, outcome, awarded_points, max_points, allocated_points, "
            "effective_weight, attribution_scopes_json, error_types_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                analysis.analysis_id,
                evidence.evidence_id,
                evidence.mapping_id,
                evidence.point_id,
                evidence.question_id,
                evidence.target_id,
                evidence.rubric_ref,
                evidence.outcome,
                evidence.awarded_points,
                evidence.max_points,
                evidence.allocated_points,
                evidence.effective_weight,
                _json(item.value for item in evidence.attribution_scopes),
                _json(item.value for item in evidence.error_types),
            ),
        )
    evidence_owners = {
        row["evidence_id"]: row["analysis_id"]
        for row in connection.execute(
            "SELECT analysis_id, evidence_id FROM knowledge_evidence"
        )
    }
    for snapshot in analysis.snapshots:
        connection.execute(
            "INSERT INTO mastery_snapshot "
            "(analysis_id, snapshot_id, point_id, performance_index, state, trend, "
            "allowed_claim_strength, evidence_count, independent_target_count, "
            "independent_question_count, independent_exam_count, effective_weight) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                analysis.analysis_id,
                snapshot.snapshot_id,
                snapshot.point_id,
                snapshot.performance_index,
                snapshot.state.value,
                snapshot.trend.value,
                snapshot.allowed_claim_strength.value,
                snapshot.evidence_count,
                snapshot.independent_target_count,
                snapshot.independent_question_count,
                snapshot.independent_exam_count,
                snapshot.effective_weight,
            ),
        )
        for evidence_id in snapshot.evidence_ids:
            owner = evidence_owners.get(evidence_id)
            if owner is None:
                raise CaseValidationError(
                    f"snapshot references unknown evidence {evidence_id}"
                )
            connection.execute(
                "INSERT INTO mastery_snapshot_evidence "
                "(analysis_id, snapshot_id, evidence_analysis_id, evidence_id) "
                "VALUES (?, ?, ?, ?)",
                (analysis.analysis_id, snapshot.snapshot_id, owner, evidence_id),
            )


def _verify_existing_analysis(
    connection: sqlite3.Connection,
    analysis: LearningAnalysis,
    report_manifest_sha256: str,
) -> None:
    row = connection.execute(
        "SELECT learning_input_manifest_sha256, learning_analysis_decision_sha256, "
        "catalog_sha256, mastery_algorithm_version FROM analysis_run WHERE analysis_id = ?",
        (analysis.analysis_id,),
    ).fetchone()
    expected = (
        analysis.learning_input_manifest_sha256,
        analysis.learning_analysis_decision_sha256,
        analysis.knowledge_catalog_sha256,
        analysis.mastery_algorithm_version,
    )
    if row is None or tuple(row) != expected:
        raise CaseValidationError("existing analysis_id has different source facts")
    report_row = connection.execute(
        "SELECT report_manifest_sha256 FROM report_record WHERE analysis_id = ?",
        (analysis.analysis_id,),
    ).fetchone()
    if report_row is None or report_row[0] != report_manifest_sha256:
        raise CaseValidationError("existing analysis has a different report manifest")


def _insert_review_events(
    connection: sqlite3.Connection, analysis: LearningAnalysis
) -> None:
    reviewed: list[tuple[str, str, str]] = []
    metadata = analysis.input_manifest.metadata
    if metadata.human_confirmed and metadata.human_review_note is not None:
        reviewed.append(("exam_metadata", analysis.case_id, metadata.human_review_note))
    for item in analysis.decision.proposed_points:
        if item.human_confirmed and item.human_review_note is not None:
            reviewed.append(("knowledge_point_proposal", item.point_id, item.human_review_note))
    for item in analysis.decision.knowledge_changes:
        if item.human_confirmed and item.human_review_note is not None:
            reviewed.append(("knowledge_change", item.change_id, item.human_review_note))
    for item in analysis.decision.mappings:
        if item.human_confirmed and item.human_review_note is not None:
            reviewed.append(("knowledge_mapping", item.mapping_id, item.human_review_note))
    for entity_kind, entity_id, reason in reviewed:
        event_id = "review-" + sha256(
            f"{analysis.analysis_id}:{entity_kind}:{entity_id}".encode("utf-8")
        ).hexdigest()[:20]
        connection.execute(
            "INSERT INTO review_event "
            "(review_event_id, analysis_id, entity_kind, entity_id, decision, reason, created_at) "
            "VALUES (?, ?, ?, ?, 'confirmed', ?, ?)",
            (
                event_id,
                analysis.analysis_id,
                entity_kind,
                entity_id,
                reason,
                analysis.created_at.isoformat(),
            ),
        )


def _json(values: Iterable[object]) -> str:
    return json.dumps(list(values), ensure_ascii=False, sort_keys=True)
