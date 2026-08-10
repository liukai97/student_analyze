# Exam solver prompt — version 1.0.0

Use the repository's `exam-solver` Skill and only the assets in the supplied
`solver_input_manifest.json`. Return one `ExamMasterDecisionSet` that validates against
`exam_master_decision.schema.json`.

Preserve every reviewed question reconstruction exactly. Solve only `reconstruct_and_solve` entries;
for those entries provide structured reference answers, rubric criteria, a concise solution summary,
assumptions, and deterministic verification requests. Do not open answer sheets or set final approval.
