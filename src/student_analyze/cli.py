"""Command-line interface for phase 1 operations."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

from student_analyze.config import load_config
from student_analyze.errors import StudentAnalyzeError
from student_analyze.image_preprocess import load_page_decisions, prepare_logical_pages
from student_analyze.pipeline import ingest_case, verify_case
from student_analyze.schema import SCHEMA_MODELS, write_schemas


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
