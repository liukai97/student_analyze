---
name: exam-solver
description: Independently solve questions marked for solving from a reviewed, clean solver-input manifest, producing structured answers, rubrics, assumptions, and verification requests, or perform an isolated second review. Use for phase-4 Exam Master construction after printed questions have been reconstructed and answer sheets, teacher annotations, student work, and answer-derived hints have been excluded.
---

# Exam Solver

Produce structured phase 4 solver or independent-review decisions. Treat the supplied manifest as the complete allowlist.

## Solve workflow

1. Read `solver_input_manifest.json` and inspect only its listed question crops. Do not open other case assets.
2. Treat each manifest question reconstruction as reviewed printed fact. Use only its listed clean visual assets for any remaining diagram context.
3. For `reconstruct_only`, do not infer or emit a reference answer.
4. For `reconstruct_and_solve`, solve independently and provide concise answer entries, rubric criteria, solution summary, assumptions, and applicable deterministic verification requests.
5. Use subject-specific Skills when available. If no suitable Skill exists, state the capability gap and require review rather than guessing.
6. Validate the output against `exam_master_decision.schema.json`. Do not set final answer-source priority, `approved`, or pipeline state.

## Independent-review workflow

1. Start from the same clean solver manifest and visual assets without reading the first solver's reasoning.
2. Solve the target question independently, then compare with the candidate answer supplied for review.
3. Return `confirmed`, `conflict`, or `uncertain` with concise evidence. Never resolve a conflict by confidence alone.
4. Validate the output against `exam_review_decision.schema.json`.

## Isolation rules

- Reject manifests containing answer-sheet, teacher-annotation, student-handwriting, scratch, or answer-derived assets.
- Do not browse neighboring case files to fill gaps.
- Keep reconstruction facts separate from solution claims.
- Use provenance values `prompt_version=exam-solver-v1.0.0` and `skill_version=exam-solver-v1.0.0`; use `prompt_version=exam-review-v1.0.0` for independent review.
