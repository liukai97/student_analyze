---
name: exam-question-reconstruction
description: Reconstruct the printed text, options, subparts, points, diagrams, and knowledge areas of mapped exam questions into clean structured content, even when the source page also contains student or teacher marks. Use before blind solving when raw question regions are contaminated; omit all handwriting and answer-derived hints, identify visual regions still needed, and require human confirmation whenever student content was visible.
---

# Exam Question Reconstruction

Produce a `QuestionReconstructionDecisionSet` for phase 4. This is a transcription and structure task, not a solving task.

## Workflow

1. Read the active `document_graph.json` and inspect every effective printed-question region.
2. Reconstruct only printed content: prompt, options, subparts, diagrams, points, and knowledge areas. Do not interpret circles, selected letters, calculations, corrections, ticks, or other handwriting.
3. Record whether source regions contain student or teacher content. If student content was visible, require explicit human confirmation of the clean reconstruction before it can enter a blind-solver manifest.
4. Describe diagrams faithfully in structured text. List only visual region IDs that the blind solver must still inspect; each listed region will require a separately reviewed clean crop.
5. Preserve uncertain printed text or diagram semantics with candidates and `requires_review`; never use handwriting to fill a printed gap.
6. Validate the output against `question_reconstruction_decision.schema.json`. Do not solve questions or write the final Exam Master.

## Isolation boundary

- This reconstruction context may see marked question pages, but its output must contain no student answer, teacher judgment, scratch reasoning, or answer-derived hint.
- Human confirmation attests only that the reconstruction matches printed content; it does not approve an answer.
- Use provenance values `prompt_version=question-reconstruction-v1.0.0` and `skill_version=exam-question-reconstruction-v1.0.0`.
