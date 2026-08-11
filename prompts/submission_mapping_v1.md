# Submission mapping prompt — version 1.0.0

Use the repository's `exam-submission-transcription` Skill in its mapping workflow. Read only the
generated `submission_structure_manifest.json` and its listed navigation pages. Return one
`SubmissionMappingDecisionSet` that validates against
`submission_mapping_decision.schema.json`.

Cover every formal response target with an answer-sheet bbox, including blank slots. Keep scratch
separate, exclude all overlapping known annotations, preserve uncertain targets, and do not inspect
reference answers, rubrics, solutions, or correctness labels.
