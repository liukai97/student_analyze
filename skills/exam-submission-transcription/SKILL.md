---
name: exam-submission-transcription
description: Map answer-sheet response units and optional question-booklet scratch work to an approved Exam Master, then faithfully transcribe handwriting, selections, formulas, structures, blanks, and erasures from high-detail crops. Use during phase 5 after master_ready to produce SubmissionMappingDecisionSet and SubmissionTranscriptionDecisionSet without grading, correcting, translating, or completing the student's work.
---

# Exam Submission Transcription

Produce model decisions for phase 5. Leave redaction, crop rendering, coordinate transforms, source priority, review gates, hashing, and final `submission.json` commits to Python.

## Mapping workflow

1. Run `submission-context` and read only its `submission_structure_manifest.json`. Do not open `exam_master.json`; the structure manifest intentionally excludes reference answers, rubrics, and solutions.
2. Inspect each listed navigation page at low or medium detail. Use accepted `answer_area` regions and question-version targets as fixed containers.
3. Create one formal mapping for every root objective question or listed part. When one part has multiple visible answer slots, create multiple mappings with distinct `slot_label` and `slot_order` values.
4. Include a bbox even for an apparently blank slot. Keep exclusive `right` and `bottom` coordinates within the logical page and accepted answer area.
5. Set `source_role=answer_sheet` for formal responses. Map question-booklet or scratch-sheet work separately and never let it replace a formal blank or response.
6. Record all visible content roles. Reference every overlapping known annotation in `excluded_annotation_refs`; if a new possible teacher mark appears, exclude it from student content and require review.
7. Preserve uncertain boundaries or targets with `requires_review`; do not use answer plausibility to choose a mapping.
8. Validate against `submission_mapping_decision.schema.json`, then run `submission-inputs` to create high-detail crops. Do not edit its manifest or crops.

## Transcription workflow

1. Read `submission_input_manifest.json` and inspect only its listed original-detail crops. Use the mapping only to identify the response unit, not to infer expected content.
2. Transcribe exactly what is visible in the original language and notation. Preserve spelling, signs, coefficients, subscripts, superscripts, charges, arrows, bond placement, and surviving revisions.
3. Use `observed_content` for the visible response. Describe a hand-drawn structure faithfully when plain text cannot reproduce its form; the crop remains the primary evidence.
4. Use `normalized_answer` only for lossless formatting, such as `b` to `B` or `Fe3+` to `Fe^{3+}`. Do not translate Chinese, expand an option letter to option text, add operands around an isolated symbol, balance an equation, or repair chemistry.
5. Put actual alternate readings in `alternatives`. Put explanations of ambiguity in `uncertainty_notes`.
6. Record an untouched empty slot with `is_blank=true`. Initially record fully cancelled but illegible writing with `is_blank=false`, `has_erasure=true`, no invented content, and review. A human reviewer may subsequently classify it as an effective blank while preserving the erasure fact by setting `is_blank=true`, `has_erasure=true`, `human_confirmed=true`, and `blank_after_erasure_review=true` with a review note.
7. Require review for confidence below `0.8`, multiple readings, unclear final revisions, uncertain content roles, newly suspected annotations, or any representation that loses material visual information.
8. Validate against `submission_transcription_decision.schema.json`, then invoke `submission`. Never write `submission.json` by hand.

## Boundaries

- Do not read or compare reference answers, rubrics, solution summaries, scores, or correctness labels.
- Do not treat semantic or chemical plausibility as transcription evidence.
- Do not copy printed labels or teacher annotations into student `observed_content`.
- Do not promote scratch work when the answer sheet is blank, erased, or different.
- Use optional subject Skills only for notation recognition and lossless rendering in this phase, never for solving or correction.

Use provenance `skill_version=exam-submission-transcription-v1.0.0`; use `prompt_version=submission-mapping-v1.0.0` for mapping and `prompt_version=submission-transcription-v1.0.0` for transcription.
