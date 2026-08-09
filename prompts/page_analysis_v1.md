# Page analysis prompt — version 1.0.0

Inspect each immutable source image with Codex standard image input. Combine visible
text direction, page borders, center seam, printed page numbers, and EXIF metadata.
Return a `PageDecisionSet` JSON document; do not modify pixels.

For every source asset:

1. Decide `single_page` or `double_page` without using filename-specific production rules.
2. Provide source-pixel crop boxes with exclusive `right` and `bottom` coordinates.
3. Choose only 0, 90, 180, or 270 degrees clockwise. EXIF is evidence, not authority.
4. Preserve a small center overlap when it prevents loss of gutter content.
5. Record concrete visual evidence, confidence, warnings, and `requires_review`.
6. If orientation or boundaries are not unique, set `requires_review=true`; do not guess.

Stage 2 determines only source-local `single`, `left`, and `right` positions. It must
not assign document roles, whole-exam semantic order, question numbers, or answer mappings.
