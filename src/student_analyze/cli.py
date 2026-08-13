"""Command-line interface for the local exam-analysis pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

from student_analyze.config import load_config
from student_analyze.document_mapper import load_document_decisions, map_documents
from student_analyze.errors import StudentAnalyzeError
from student_analyze.exam_master import (
    build_exam_master,
    load_answer_evidence_decisions,
    load_exam_master_decisions,
    load_exam_review_decisions,
    load_question_reconstruction_decisions,
    load_solver_input_decisions,
    prepare_solver_inputs,
    read_solver_input_manifest,
)
from student_analyze.image_preprocess import load_page_decisions, prepare_logical_pages
from student_analyze.grading import (
    build_grading,
    finalize_grading_review,
    load_grading_decisions,
    load_grading_input_manifest,
    load_grading_review_decisions,
    prepare_grading_context,
    prepare_grading_review,
)
from student_analyze.pipeline import ingest_case, verify_case
from student_analyze.schema import SCHEMA_MODELS, write_schemas
from student_analyze.submission import (
    build_submission,
    load_submission_mapping_decisions,
    load_submission_structure_manifest,
    load_submission_transcription_decisions,
    prepare_submission_inputs,
    prepare_submission_structure,
    read_submission_input_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="student-analyze")
    parser.add_argument("--config", type=Path, help="optional TOML configuration")
    parser.add_argument("--cases-dir", type=Path, help="override the case storage directory")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="hash and register a read-only source directory")
    ingest.add_argument("source_root", type=Path)
    ingest.add_argument("--force", action="store_true", help="record a new ingestion run")
    ingest.add_argument("--json", action="store_true", help="emit a machine-readable result")

    status = subparsers.add_parser("status", help="show the validated pipeline state")
    status.add_argument("case_dir", type=Path)

    verify = subparsers.add_parser("verify", help="verify source and artifact integrity")
    verify.add_argument("case_dir", type=Path)

    pages = subparsers.add_parser(
        "pages", help="generate logical pages from reviewed visual decisions"
    )
    pages.add_argument("case_dir", type=Path)
    pages.add_argument("decisions", type=Path)
    pages.add_argument("--force", action="store_true", help="preserve a new page run")
    pages.add_argument("--json", action="store_true", help="emit a machine-readable result")

    mapping = subparsers.add_parser(
        "map", help="build a document graph from reviewed semantic decisions"
    )
    mapping.add_argument("case_dir", type=Path)
    mapping.add_argument("decisions", type=Path)
    mapping.add_argument("--force", action="store_true", help="preserve a new mapping run")
    mapping.add_argument("--json", action="store_true", help="emit a machine-readable result")

    master_inputs = subparsers.add_parser(
        "master-inputs", help="build reviewed question-only inputs for phase 4"
    )
    master_inputs.add_argument("case_dir", type=Path)
    master_inputs.add_argument("evidence_decisions", type=Path)
    master_inputs.add_argument("reconstruction_decisions", type=Path)
    master_inputs.add_argument("crop_decisions", type=Path)
    master_inputs.add_argument(
        "--json", action="store_true", help="emit a machine-readable result"
    )

    master = subparsers.add_parser(
        "master", help="compile reviewed phase 4 decisions into an Exam Master"
    )
    master.add_argument("case_dir", type=Path)
    master.add_argument("evidence_decisions", type=Path)
    master.add_argument("solver_manifest", type=Path)
    master.add_argument("master_decisions", type=Path)
    master.add_argument("review_decisions", type=Path)
    master.add_argument("--force", action="store_true", help="preserve a new master run")
    master.add_argument("--json", action="store_true", help="emit a machine-readable result")

    submission_context = subparsers.add_parser(
        "submission-context",
        help="build an answer-key-redacted structure manifest for phase 5",
    )
    submission_context.add_argument("case_dir", type=Path)
    submission_context.add_argument(
        "--json", action="store_true", help="emit a machine-readable result"
    )

    submission_inputs = subparsers.add_parser(
        "submission-inputs",
        help="validate response mappings and create high-detail transcription crops",
    )
    submission_inputs.add_argument("case_dir", type=Path)
    submission_inputs.add_argument("structure_manifest", type=Path)
    submission_inputs.add_argument("mapping_decisions", type=Path)
    submission_inputs.add_argument(
        "--json", action="store_true", help="emit a machine-readable result"
    )

    submission = subparsers.add_parser(
        "submission",
        help="compile faithful transcription decisions into a phase-5 Submission",
    )
    submission.add_argument("case_dir", type=Path)
    submission.add_argument("input_manifest", type=Path)
    submission.add_argument("transcription_decisions", type=Path)
    submission.add_argument(
        "--force", action="store_true", help="preserve a new submission run"
    )
    submission.add_argument(
        "--json", action="store_true", help="emit a machine-readable result"
    )

    grading_context = subparsers.add_parser(
        "grading-context",
        help="route reviewed responses to deterministic or LLM grading",
    )
    grading_context.add_argument("case_dir", type=Path)
    grading_context.add_argument(
        "--json", action="store_true", help="emit a machine-readable result"
    )

    grade = subparsers.add_parser(
        "grade",
        help="combine deterministic grading and rubric decisions",
    )
    grade.add_argument("case_dir", type=Path)
    grade.add_argument("input_manifest", type=Path)
    grade.add_argument("grading_decisions", type=Path)
    grade.add_argument("--force", action="store_true", help="preserve a new grading run")
    grade.add_argument("--json", action="store_true", help="emit a machine-readable result")

    grading_review = subparsers.add_parser(
        "grading-review",
        help="render the local review packet for disputed grading items",
    )
    grading_review.add_argument("case_dir", type=Path)
    grading_review.add_argument(
        "--json", action="store_true", help="emit a machine-readable result"
    )

    review = subparsers.add_parser(
        "review",
        help="apply complete human grading decisions",
    )
    review.add_argument("case_dir", type=Path)
    review.add_argument("review_decisions", type=Path)
    review.add_argument("--force", action="store_true", help="preserve a new review run")
    review.add_argument("--json", action="store_true", help="emit a machine-readable result")

    schema = subparsers.add_parser("schema", help="generate JSON Schemas from Pydantic")
    schema.add_argument("--output-dir", type=Path, default=Path("schemas"))
    schema.add_argument("--check", action="store_true", help="fail if committed schemas differ")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, cases_dir=args.cases_dir)
        logging.basicConfig(
            level=getattr(logging, config.log_level),
            format="%(levelname)s %(name)s: %(message)s",
        )

        if args.command == "ingest":
            result = ingest_case(args.source_root, config, force=args.force)
            payload = {
                "case_id": result.manifest.case_id,
                "case_dir": str(result.case_dir.resolve()),
                "current_stage": result.state.current_stage.value,
                "reused": result.reused,
            }
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            else:
                action = "reused" if result.reused else "completed"
                print(f"ingest {action}: {payload['case_id']}")
                print(payload["case_dir"])
            return 0

        if args.command in {"status", "verify"}:
            manifest, state = verify_case(args.case_dir.resolve(strict=True))
            if args.command == "status":
                print(
                    json.dumps(
                        state.model_dump(mode="json"),
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    )
                )
            else:
                print(
                    f"verified {manifest.case_id}: stage={state.current_stage.value}, "
                    f"assets={len(manifest.source_assets)}, runs={len(state.run_history)}"
                )
            return 0

        if args.command == "pages":
            decisions = load_page_decisions(args.decisions)
            result = prepare_logical_pages(
                args.case_dir,
                decisions,
                config,
                force=args.force,
            )
            payload = {
                "case_id": result.manifest.case_id,
                "case_dir": str(result.case_dir),
                "current_stage": result.state.current_stage.value,
                "logical_pages": len(result.manifest.pages),
                "requires_review": result.manifest.requires_review,
                "reused": result.reused,
            }
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            else:
                action = "reused" if result.reused else "completed"
                print(
                    f"pages {action}: {payload['case_id']}, "
                    f"logical_pages={payload['logical_pages']}"
                )
            return 0

        if args.command == "map":
            decisions = load_document_decisions(args.decisions)
            result = map_documents(
                args.case_dir,
                decisions,
                config,
                force=args.force,
            )
            payload = {
                "case_id": result.graph.case_id,
                "case_dir": str(result.case_dir),
                "current_stage": result.state.current_stage.value,
                "documents": len(result.graph.documents),
                "questions": len(result.graph.questions),
                "question_versions": len(result.graph.question_versions),
                "unresolved_relations": len(result.graph.unresolved_relations),
                "requires_review": result.graph.requires_review,
                "reused": result.reused,
            }
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            else:
                action = "reused" if result.reused else "completed"
                print(
                    f"map {action}: {payload['case_id']}, "
                    f"documents={payload['documents']}, questions={payload['questions']}"
                )
            return 0

        if args.command == "master-inputs":
            evidence = load_answer_evidence_decisions(args.evidence_decisions)
            reconstructions = load_question_reconstruction_decisions(
                args.reconstruction_decisions
            )
            crop_decisions = load_solver_input_decisions(args.crop_decisions)
            result = prepare_solver_inputs(
                args.case_dir,
                evidence,
                reconstructions,
                crop_decisions,
                config,
            )
            payload = {
                "case_id": result.manifest.case_id,
                "case_dir": str(result.case_dir),
                "solver_input_manifest": str(result.manifest_path),
                "questions": len(result.manifest.questions),
                "solve_questions": sum(
                    item.task.value == "reconstruct_and_solve"
                    for item in result.manifest.questions
                ),
                "reused": result.reused,
            }
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            else:
                action = "reused" if result.reused else "completed"
                print(
                    f"master-inputs {action}: {payload['case_id']}, "
                    f"solve_questions={payload['solve_questions']}"
                )
                print(payload["solver_input_manifest"])
            return 0

        if args.command == "master":
            evidence = load_answer_evidence_decisions(args.evidence_decisions)
            solver_manifest = read_solver_input_manifest(args.solver_manifest)
            master_decisions = load_exam_master_decisions(args.master_decisions)
            review_decisions = load_exam_review_decisions(args.review_decisions)
            result = build_exam_master(
                args.case_dir,
                evidence,
                solver_manifest,
                master_decisions,
                review_decisions,
                config,
                force=args.force,
            )
            payload = {
                "case_id": result.master.case_id,
                "case_dir": str(result.case_dir),
                "current_stage": result.state.current_stage.value,
                "questions": len(result.master.questions),
                "requires_review": result.master.requires_review,
                "reused": result.reused,
            }
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            else:
                action = "reused" if result.reused else "completed"
                print(
                    f"master {action}: {payload['case_id']}, "
                    f"questions={payload['questions']}"
                )
            return 0

        if args.command == "submission-context":
            result = prepare_submission_structure(args.case_dir, config)
            payload = {
                "case_id": result.manifest.case_id,
                "case_dir": str(result.case_dir),
                "structure_manifest": str(result.manifest_path),
                "questions": len(result.manifest.questions),
                "navigation_pages": len(result.manifest.pages),
                "known_annotations": len(result.manifest.known_annotations),
                "reused": result.reused,
            }
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            else:
                action = "reused" if result.reused else "completed"
                print(
                    f"submission-context {action}: {payload['case_id']}, "
                    f"questions={payload['questions']}"
                )
                print(payload["structure_manifest"])
            return 0

        if args.command == "submission-inputs":
            structure = load_submission_structure_manifest(args.structure_manifest)
            mappings = load_submission_mapping_decisions(args.mapping_decisions)
            result = prepare_submission_inputs(
                args.case_dir,
                structure,
                mappings,
                config,
            )
            payload = {
                "case_id": result.manifest.case_id,
                "case_dir": str(result.case_dir),
                "submission_input_manifest": str(result.manifest_path),
                "response_units": len(result.manifest.items),
                "reused": result.reused,
            }
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            else:
                action = "reused" if result.reused else "completed"
                print(
                    f"submission-inputs {action}: {payload['case_id']}, "
                    f"response_units={payload['response_units']}"
                )
                print(payload["submission_input_manifest"])
            return 0

        if args.command == "submission":
            input_manifest = read_submission_input_manifest(args.input_manifest)
            transcriptions = load_submission_transcription_decisions(
                args.transcription_decisions
            )
            result = build_submission(
                args.case_dir,
                input_manifest,
                transcriptions,
                config,
                force=args.force,
            )
            payload = {
                "case_id": result.submission.case_id,
                "case_dir": str(result.case_dir),
                "current_stage": result.state.current_stage.value,
                "response_units": len(result.submission.items),
                "review_items": len(result.submission.review_items),
                "requires_review": result.submission.requires_review,
                "reused": result.reused,
            }
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            else:
                action = "reused" if result.reused else "completed"
                print(
                    f"submission {action}: {payload['case_id']}, "
                    f"response_units={payload['response_units']}, "
                    f"review_items={payload['review_items']}"
                )
            return 0

        if args.command == "grading-context":
            result = prepare_grading_context(args.case_dir, config)
            payload = {
                "case_id": result.manifest.case_id,
                "case_dir": str(result.case_dir),
                "grading_input_manifest": str(result.manifest_path),
                "targets": len(result.manifest.targets),
                "auto_objective": sum(
                    item.route.value == "auto_objective"
                    for item in result.manifest.targets
                ),
                "auto_blank": sum(
                    item.route.value == "auto_blank"
                    for item in result.manifest.targets
                ),
                "llm_targets": len(result.manifest.llm_target_ids),
                "reused": result.reused,
            }
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            else:
                action = "reused" if result.reused else "completed"
                print(
                    f"grading-context {action}: {payload['case_id']}, "
                    f"auto_objective={payload['auto_objective']}, "
                    f"auto_blank={payload['auto_blank']}, "
                    f"llm_targets={payload['llm_targets']}"
                )
                print(payload["grading_input_manifest"])
            return 0

        if args.command == "grade":
            input_manifest = load_grading_input_manifest(args.input_manifest)
            decisions = load_grading_decisions(args.grading_decisions)
            result = build_grading(
                args.case_dir,
                input_manifest,
                decisions,
                config,
                force=args.force,
            )
            payload = {
                "case_id": result.grading.case_id,
                "case_dir": str(result.case_dir),
                "current_stage": result.state.current_stage.value,
                "provisional_score": result.grading.provisional_score,
                "max_score": result.grading.max_score,
                "review_items": len(result.grading.review_items),
                "requires_review": result.grading.requires_review,
                "reused": result.reused,
            }
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            else:
                action = "reused" if result.reused else "completed"
                print(
                    f"grade {action}: {payload['case_id']}, "
                    f"score={payload['provisional_score']}/{payload['max_score']}, "
                    f"review_items={payload['review_items']}"
                )
            return 0

        if args.command == "grading-review":
            result = prepare_grading_review(args.case_dir)
            payload = {
                "case_id": result.manifest.case_id,
                "case_dir": str(result.case_dir),
                "review_manifest": str(result.manifest_path),
                "review_html": str(result.html_path),
                "review_items": len(result.manifest.items),
                "reused": result.reused,
            }
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            else:
                action = "reused" if result.reused else "completed"
                print(
                    f"grading-review {action}: {payload['case_id']}, "
                    f"review_items={payload['review_items']}"
                )
                print(payload["review_manifest"])
                print(payload["review_html"])
            return 0

        if args.command == "review":
            decisions = load_grading_review_decisions(args.review_decisions)
            result = finalize_grading_review(
                args.case_dir,
                decisions,
                config,
                force=args.force,
            )
            payload = {
                "case_id": result.grading.case_id,
                "case_dir": str(result.case_dir),
                "current_stage": result.state.current_stage.value,
                "final_score": result.grading.final_score,
                "max_score": result.grading.max_score,
                "reused": result.reused,
            }
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            else:
                action = "reused" if result.reused else "completed"
                print(
                    f"review {action}: {payload['case_id']}, "
                    f"score={payload['final_score']}/{payload['max_score']}"
                )
            return 0

        if args.command == "schema":
            changed = write_schemas(args.output_dir, check=args.check)
            if args.check and changed:
                for path in changed:
                    print(f"schema differs: {path}", file=sys.stderr)
                return 1
            action = "checked" if args.check else "generated"
            count = len(SCHEMA_MODELS) if args.check else len(changed)
            print(f"{action} {count} schemas")
            return 0
    except (StudentAnalyzeError, OSError, ValueError) as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 2

    parser.error(f"unknown command: {args.command}")
    return 2
