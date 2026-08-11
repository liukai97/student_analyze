# Submission transcription prompt — version 1.0.0

Use the repository's `exam-submission-transcription` Skill in its transcription workflow. Read the
generated `submission_input_manifest.json` and inspect only its listed high-detail crops. Return one
`SubmissionTranscriptionDecisionSet` that validates against
`submission_transcription_decision.schema.json`.

Record exactly what is visible, format-only normalization, real alternate readings, blank state,
erasure state, confidence, and review reasons. Do not grade, translate, expand option meanings,
repair notation, infer erased content, or use expected-answer plausibility.
