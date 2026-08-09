---
name: exam-analysis
description: Classify exam images and logical pages, reconstruct question and subquestion structure, identify page order and evidence regions, and propose evidence-backed document relationships such as continued_by, answers, scratch_evidence_for, and optional supersedes. Use when mapping scanned or photographed question booklets, answer sheets, replacement or errata pages, supplemental materials, or scratch work into a DocumentGraphDecisionSet before solving, transcribing, or grading.
---

# Exam Analysis

Produce semantic decisions for phase 3. Leave hashing, stable final IDs, graph validation, effective-version calculation, and artifact commits to Python.

## Workflow

1. Read the active `page_manifest.json` and inspect every referenced logical-page image. Treat filenames and input order only as locators, never as semantic evidence.
2. Inspect all pages at navigation detail. Group pages into documents and classify each document as `question_booklet`, `answer_sheet`, `replacement_or_errata`, `supplemental_material`, `scratch`, `other`, or `unknown`.
3. Infer each document's internal page order from printed page numbers, titles, question sequences, answer frames, and cross-page continuity. Preserve uncertainty instead of forcing a total order.
4. Locate printed questions, answer areas, scratch work, and replacement notices in logical-page coordinates. Use exclusive `right` and `bottom` bbox coordinates. Reinspect only conflicting pages or necessary high-detail crops and record why.
5. Reconstruct question and subquestion hierarchy. Give every conceptual question one decision `ref`, and give each visible version a separate version `ref`. Attach one or more `printed_question` regions to each version.
6. Propose semantic relations with concrete visual evidence. Mark an unambiguous conclusion `accepted` with `requires_review=false`; mark an unresolved alternative `candidate` with `requires_review=true`.
7. Validate the decision JSON against `schemas/document_graph_decision.schema.json`, then invoke the repository's `map` command. Do not edit `document_graph.json` by hand.

## Relationship rules

- Use `continued_by` only between printed regions belonging to the same question version when content visibly continues across regions or pages.
- Use `answers` from an `answer_area` region to a concrete question-version ref. Map the area without transcribing or judging the student's writing.
- Use `scratch_evidence_for` from a `scratch` region to a concrete question-version ref. Never let scratch override an answer-sheet region.
- Do not output `contains_page` or `derived_from`; Python derives them from validated assignments and bboxes.

## Optional version and replacement relations

Treat the absence of `supersedes` as the normal case. Never create one merely because two pages share a question number.

Propose `new-version supersedes old-version` only when visual semantics support replacement or correction. Prefer evidence in this order:

1. An explicit printed replacement or errata directive that identifies the affected exam and question.
2. A matching original question version in the same exam materials.
3. Compatible printed question number, points, instructions, or answer-sheet layout.

Distinguish replacement from parallel A/B papers, duplicate scans, alternate-language editions, answer keys, and ordinary supplements. If the notice is missing, the target is ambiguous, or multiple newer versions remain possible, emit candidate relations and require review. Do not use student handwriting as the primary evidence for choosing the effective version.

Do not set an `effective` field. Python computes the effective version only after validating accepted `supersedes` edges. The current chemistry question 15 replacement is a regression example, not a general rule or fixed path.

## Evidence and review

- Cite visible text, layout, page number, question sequence, or region coordinates; avoid unsupported summaries such as "looks like a replacement."
- Use confidence only as a routing signal, not as a calibrated probability.
- Set `requires_review=true` for unknown roles, ambiguous page order, uncertain boundaries, conflicting versions, or non-unique mappings.
- Do not inspect student answers to solve questions during this phase. Preserve the later blind-solver boundary.

Use provenance values `prompt_version=document-mapping-v1.0.0` and `skill_version=exam-analysis-v1.0.0` for this version of the workflow.
