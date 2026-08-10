# Question reconstruction prompt — version 1.0.0

Use the repository's `exam-question-reconstruction` Skill to inspect every effective printed-question
region in the active document graph. Return one `QuestionReconstructionDecisionSet` that validates
against `question_reconstruction_decision.schema.json`.

Reconstruct only printed prompt text, options, subparts, diagrams, points, and knowledge areas. Omit
all student/teacher marks and answer-derived hints. Record contaminated sources, human-confirmation
state, uncertainty, and the visual region IDs for which a separate clean crop remains necessary. Do
not solve any question.
