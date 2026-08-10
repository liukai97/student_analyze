# Solver input review prompt — version 1.0.0

Inspect proposed clean visual crops requested by the reviewed question reconstruction. Return one
`SolverInputDecisionSet` that validates against `solver_input_decision.schema.json`.

Each crop must contain the necessary printed diagram content and exclude student handwriting,
teacher annotations, answer sheets, scratch work, and answer-derived hints. Tighten the bbox or mark
the crop for review when isolation cannot be guaranteed. Do not solve or transcribe answers.
