---
name: exam-grading
description: Grade reviewed student exam responses against an approved Exam Master rubric, criterion by criterion, and diagnose evidence-backed academic errors. Use in phase 6 after mapping and transcription review is complete, for non-blank non-objective targets routed as `llm_rubric` in a `GradingInputManifest`; do not use for objective or blank targets, which Python grades deterministically.
---

# Exam Grading

Produce a `GradingDecisionSet` for the LLM-routed targets in phase 6. Leave
objective matching, blank scoring, point aggregation, validation, review routing,
hashing, and final artifact commits to Python.

## Workflow

1. Read only the supplied `grading_input_manifest.json`. Treat it as the complete
   reviewed grading context; do not reopen `submission.json`. Inspect a listed
   response crop only when the reviewed transcription explicitly preserves a
   diagram, structure, layout, or other visual answer that plain text cannot encode.
2. Process exactly the target IDs in `llm_target_ids`, in that order. Do not emit
   decisions for `auto_objective` or `auto_blank` targets.
3. Compare the reviewed response text with the approved reference answers,
   solution summary, assumptions, and each rubric criterion. Consider all
   response slots belonging to the target,
   but never modify or complete their transcription.
4. Emit exactly one evaluation for every rubric criterion. Use:
   - `met` only for full credit;
   - `partially_met` only for a point value strictly between zero and the
     criterion maximum;
   - `not_met` only for zero points;
   - `undetermined` when the supplied evidence cannot support a score.
5. Cite only `submission_item_id` values from that target as evidence. Give a
   concise rationale that states which visible element satisfies or misses the
   criterion.
6. Add an academic error diagnosis for every partially met or unmet criterion.
   Classify only what the response supports: `concept_error`,
   `calculation_error`, `notation_or_equation_error`, `reasoning_omission`, or
   `other`. Several criteria may share one diagnosis.
7. Set `visual_judgment_required=true` whenever awarded credit depends on direct
   inspection of a response crop. Require review for that case, an undetermined
   criterion, confidence below `0.8`, a
   genuinely ambiguous partial-credit boundary, or a visual/diagram judgment
   that cannot be resolved from the reviewed transcription. Give concrete
   `review_reasons`.
8. Validate against `schemas/grading_decision.schema.json`, then invoke the
   repository's `grade` command. Never edit `grading.json` directly.

## Scoring boundaries

- Judge only the written response, not likely intent, effort, or knowledge not
  shown in the evidence.
- Do not award credit because a final answer is plausible when the rubric
  explicitly requires reasoning, units, notation, or intermediate steps.
- Do not deduct twice for one omission unless separate rubric criteria explicitly
  require separate elements.
- Treat mathematically or scientifically equivalent expressions as equivalent
  when the rubric permits them. Use subject-specific Skills when available for
  domain conventions; if the equivalence remains uncertain, return
  `undetermined` and require review.
- For chemistry targets, load `skills/subjects/chemistry-exam-grading/SKILL.md`.
- Keep academic errors separate from transcription, mapping, and reference-answer
  uncertainty. Phase 6 accepts only inputs whose earlier human review is complete.
- A wrong objective choice is not evidence of a specific conceptual cause;
  Python records only `incorrect_objective` for it.

Use provenance `prompt_version=grading-v1.0.0` and
`skill_version=exam-grading-v1.0.0`.
