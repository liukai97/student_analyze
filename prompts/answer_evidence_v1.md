# Answer evidence prompt — version 1.0.0

Use the repository's `exam-answer-evidence` Skill to inspect the active mapped case before any
blind solving. Return one `AnswerEvidenceDecisionSet` that validates against
`answer_evidence_decision.schema.json`.

Record visible official answers and teacher annotations with actor, meaning, scope, answer/rubric
support, bbox, evidence, uncertainty, and human-confirmation state. A batch mark is only a candidate
until its actor, meaning, and complete scope are explicitly confirmed. Do not solve questions or set
final approval.
