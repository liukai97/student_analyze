# Exam independent review prompt — version 1.0.0

Use the repository's `exam-solver` Skill in independent-review mode. Start from the same clean
solver manifest and solve each review target before comparing with the supplied candidate answer. Return one
`ExamReviewDecisionSet` that validates against `exam_review_decision.schema.json`.

Mark each target `confirmed`, `conflict`, or `uncertain` with concise evidence. Do not read student
work, teacher annotations, or the first solver's reasoning.
