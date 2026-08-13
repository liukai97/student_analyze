# Phase 6 rubric grading v1.0.0

Use `skills/exam-grading/SKILL.md`.
For chemistry targets, also use
`skills/subjects/chemistry-exam-grading/SKILL.md`.

Read the supplied `grading_input_manifest.json` and produce one
`GradingDecisionSet` containing exactly the targets listed by `llm_target_ids`,
in that order. Evaluate every rubric criterion exactly once, cite only response
item IDs belonging to that target, and preserve the reviewed transcription.

Do not grade `auto_objective` or `auto_blank` targets. Do not calculate question
or exam totals. Do not revise the Exam Master, Submission, or source evidence.

Validate the result against `schemas/grading_decision.schema.json`.
