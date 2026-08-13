---
name: chemistry-exam-grading
description: Apply chemistry-specific equivalence, notation, equation, structure, calculation, and error-diagnosis conventions while grading reviewed exam responses. Use together with `exam-grading` for chemistry targets routed as `llm_rubric` in phase 6, especially formulas, equations, mechanisms, structures, crystal calculations, experimental explanations, and multi-step reasoning.
---

# Chemistry Exam Grading

Apply these chemistry-specific rules while following `exam-grading`. Do not
change its routing, evidence, point, or review requirements.

## Equivalence and notation

- Accept equivalent chemical and mathematical notation only when identity,
  charge, stoichiometry, state, direction, and requested conditions are
  preserved.
- Distinguish harmless typography from chemical changes. Missing or wrong
  subscripts, superscripts, ionic charges, coefficients, equilibrium arrows, or
  bonding positions are material when they change meaning or the rubric requires
  the notation explicitly.
- Do not silently balance, repair, or complete the student's formula or equation.
- For a reaction equation, assess species, direction, conservation, conditions,
  and required state symbols separately when the rubric distinguishes them.
- For a numerical or crystal expression, check the particle count, molar mass,
  powers of ten, unit conversion, denominator or volume expression, and final
  unit. Accept algebraically equivalent forms.

## Structures and diagrams

- Use the reviewed transcription for textual facts. Inspect its listed crop when
  bond placement, connectivity, stereochemistry, ring closure, electron flow, or
  spatial arrangement cannot be represented losslessly in text.
- Set `visual_judgment_required=true` and require human review whenever awarded
  credit depends on direct interpretation of a drawn structure, mechanism,
  orbital, crystal cell, apparatus, or other diagram.
- Do not infer an obscured bond, atom, coefficient, or label from the expected
  answer.

## Academic error classification

- Use `concept_error` for a wrong chemical principle, species identity,
  structure, causal relationship, or interpretation.
- Use `calculation_error` when the chemical setup is sound but arithmetic,
  algebra, particle count, exponent, or unit conversion is wrong.
- Use `notation_or_equation_error` for a chemically material formula, charge,
  coefficient, arrow, state, bond, or equation-format defect.
- Use `reasoning_omission` when the conclusion may be correct but a rubric-required
  explanation, comparison, condition, experimental step, or justification is
  absent.
- Cite only the response item and rubric criterion that expose the error. Do not
  infer a stable knowledge weakness from one response; phase 7 handles evidence
  aggregation.

If chemistry equivalence or partial credit remains genuinely debatable, return
`undetermined` and require review rather than choosing by confidence alone.
