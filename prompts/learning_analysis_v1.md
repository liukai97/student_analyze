# Phase 7 knowledge mapping v1.0.0

Use `skills/exam-learning-analysis/SKILL.md`.

Read the supplied `learning_input_manifest.json` and produce exactly one
`LearningAnalysisDecision` JSON object. Semantically map every reviewed target
rubric to the embedded knowledge catalog, propose conservative new points only
when necessary, and add evidence-backed recommendations.

Do not recalculate scores, mastery, trends, evidence weights, database rows, or
report prose. Do not infer a diagnostic knowledge weakness from a blank or
incorrect objective response. Use an explicit unmapped decision when evidence
does not support a stable mapping.

Validate the result against
`schemas/learning_analysis_decision.schema.json`.
