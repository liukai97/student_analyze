---
name: exam-answer-evidence
description: Inspect mapped exam materials for existing reference-answer evidence, including official answer pages, teacher corrections, per-item marks, scores, full-credit statements, and annotations spanning multiple questions. Use before blind solving to propose evidence-backed actor, meaning, scope, answer content, rubric support, uncertainty, and review decisions without solving questions or treating student work as intrinsically correct.
---

# Exam Answer Evidence

Produce an `AnswerEvidenceDecisionSet` for phase 4. Leave reference validation, scope expansion, routing, approval, hashing, and artifact commits to Python.

## Workflow

1. Read the active `document_graph.json` and inspect all relevant logical pages. Use IDs and paths only as locators.
2. Find official answer material and possible teacher annotations, corrections, scores, full-credit statements, ticks, crosses, or marks spanning a section.
3. Separate visible facts from interpretation. Record the observed mark or text, page bbox, candidate actor, endorsement meaning, target questions, answer content if visible, and concrete visual evidence.
4. Treat actor, meaning, and scope as separate decisions. Do not infer teacher authorship from color alone or expand a nearby mark beyond the region it visibly governs.
5. A large tick, full score, or “all correct” over an objective-answer block may propose a batch endorsement. Mark it candidate and require review unless a human has explicitly confirmed actor, meaning, and scope.
6. For subjective work, distinguish an accepted answer example from a complete rubric. A tick or full score does not establish all equivalent answers or partial-credit rules.
7. Preserve conflicts between official answers, teacher corrections, and endorsed responses. Do not choose a winner or set final approval.
8. Validate the decision JSON against `answer_evidence_decision.schema.json`. Do not edit normalized evidence or `exam_master.json` by hand.

## Boundaries

- This context may inspect answer areas only to record existing answer evidence. Do not solve questions.
- Do not copy teacher annotations into the student submission layer.
- Use `unknown`, candidates, warnings, and `requires_review` instead of guessing.
- An “examined” or ambiguous tick is not equivalent to “correct.”
- Use provenance values `prompt_version=answer-evidence-v1.0.0` and `skill_version=exam-answer-evidence-v1.0.0`.
