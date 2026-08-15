---
name: exam-learning-analysis
description: Map finalized exam rubric evidence to a versioned knowledge catalog and propose evidence-backed practice recommendations. Use after phase 6 grading is reviewed, when producing a LearningAnalysisDecision JSON from learning_input_manifest.json; do not use for scoring, mastery calculations, historical trend computation, or prose report generation.
---

# Exam Learning Analysis

## Purpose

Produce only the semantic decisions that code cannot safely infer: rubric-to-knowledge-point mappings, conservative new-point proposals, and evidence-backed practice recommendations. The phase 7 compiler validates these decisions and performs all arithmetic, history aggregation, trend classification, persistence, and report rendering.

## Workflow

1. Read the complete `learning_input_manifest.json`. Treat its reviewed scores, rubric evaluations, error diagnoses, metadata, and catalog as immutable facts.
2. Read [references/decision-contract.md](references/decision-contract.md).
3. Reuse an active catalog point when its scope matches the assessed rubric. Never map by name similarity alone.
4. Propose a new point only when no active point represents the assessed concept. Give it a durable lowercase hyphen ID and cite the supporting target IDs.
5. Cover every `(target_id, rubric_ref)` pair exactly once as either a mapped group or one explicit unmapped decision.
6. Add recommendations only when their point IDs and mapping IDs are present in the same output.
7. Set `human_confirmed: false` and omit `human_review_note`. Mark new points, unmapped rubrics, low-confidence mappings, and any unresolved semantic ambiguity with `requires_review: true`; never self-confirm them.
8. Fill provenance truthfully and validate the final JSON against `schemas/learning_analysis_decision.schema.json`.

## Boundaries

- Do not alter scores, rubric awards, error diagnoses, transcriptions, metadata, or the input catalog.
- Do not calculate mastery, performance indices, trends, evidence weights, report claims, or database records.
- Do not diagnose knowledge weakness merely because an answer is blank or an objective answer is wrong.
- Do not invent historical evidence.
- Do not emit a change to an existing knowledge point unless a human has explicitly reviewed and confirmed that exact rename, move, alias, merge, split, or retirement. Omit unconfirmed changes.
- Prefer an explicit unmapped decision over a weak semantic guess.

## Output

Return one `LearningAnalysisDecision` JSON object and no surrounding prose. Preserve `case_id`, use the exact SHA-256 of the supplied learning input manifest, and use the phase 7 schema version. Keep rationales concise but specific enough for human audit. A human may later confirm review items in a revised decision JSON by setting `human_confirmed: true`, clearing `requires_review`, and recording `human_review_note`; that confirmation is not part of the model's task.
