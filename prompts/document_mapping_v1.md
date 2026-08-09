# Document mapping prompt — version 1.0.0

Use the repository's `exam-analysis` Skill to inspect every logical page referenced by the
active `page_manifest.json`. Return one `DocumentGraphDecisionSet` JSON document that validates
against `document_graph_decision.schema.json`.

Classify document roles and page order, reconstruct question and subquestion hierarchy, locate
evidence regions, and propose semantic relations. Treat `supersedes` as optional: create it only
from visible replacement or errata evidence, never from a fixed question number, filename, or
student handwriting. Map answer areas without transcribing or judging their contents. Preserve
ambiguous relationships as candidates with `requires_review=true`; do not guess.
