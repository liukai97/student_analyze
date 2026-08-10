"""Deterministic checks used while compiling an Exam Master."""

from __future__ import annotations

from collections import Counter
import re

from student_analyze.exam_master_models import (
    AnswerEntryDecision,
    QuestionMasterDecision,
    QuestionType,
    RubricCriterionDecision,
    VerificationMethod,
    VerificationResult,
    VerificationStatus,
)


_EQUATION_ARROW = re.compile(r"(?:<=>|⇌|↔|->|→|=)")
_ELEMENT = re.compile(r"[A-Z][a-z]?")


def run_deterministic_verifications(
    question: QuestionMasterDecision,
    answers: list[AnswerEntryDecision],
    rubric: list[RubricCriterionDecision],
) -> list[VerificationResult]:
    methods = list(dict.fromkeys(question.verification_requests))
    if question.question_type in {
        QuestionType.OBJECTIVE_SINGLE,
        QuestionType.OBJECTIVE_MULTIPLE,
    } and VerificationMethod.OPTION_MEMBERSHIP not in methods:
        methods.append(VerificationMethod.OPTION_MEMBERSHIP)
    if VerificationMethod.RUBRIC_POINTS not in methods:
        methods.append(VerificationMethod.RUBRIC_POINTS)

    results: list[VerificationResult] = []
    for method in methods:
        if method == VerificationMethod.OPTION_MEMBERSHIP:
            results.append(_verify_option_membership(question, answers))
        elif method == VerificationMethod.RUBRIC_POINTS:
            results.append(_verify_rubric_points(question, rubric))
        elif method == VerificationMethod.CHEMICAL_EQUATION_BALANCE:
            results.append(_verify_chemical_equations(answers))
        else:
            results.append(
                VerificationResult(
                    method=method,
                    status=VerificationStatus.NOT_APPLICABLE,
                    details=(
                        "No deterministic implementation is registered for this method; "
                        "the independent review remains authoritative."
                    ),
                )
            )
    return results


def _verify_option_membership(
    question: QuestionMasterDecision,
    answers: list[AnswerEntryDecision],
) -> VerificationResult:
    if question.question_type not in {
        QuestionType.OBJECTIVE_SINGLE,
        QuestionType.OBJECTIVE_MULTIPLE,
    }:
        return VerificationResult(
            method=VerificationMethod.OPTION_MEMBERSHIP,
            status=VerificationStatus.NOT_APPLICABLE,
            details="Question is not objective.",
        )
    allowed = {option.label.strip() for option in question.options}
    if not allowed:
        return VerificationResult(
            method=VerificationMethod.OPTION_MEMBERSHIP,
            status=VerificationStatus.FAILED,
            details="Objective question has no reconstructed options.",
        )
    if len(answers) != 1 or answers[0].part_ref is not None:
        return VerificationResult(
            method=VerificationMethod.OPTION_MEMBERSHIP,
            status=VerificationStatus.FAILED,
            details="Objective question must have one question-level answer entry.",
        )

    selected = _split_option_labels(answers[0].answer)
    valid = bool(selected) and selected <= allowed
    if question.question_type == QuestionType.OBJECTIVE_SINGLE:
        valid = valid and len(selected) == 1
    return VerificationResult(
        method=VerificationMethod.OPTION_MEMBERSHIP,
        status=VerificationStatus.PASSED if valid else VerificationStatus.FAILED,
        details=(
            f"Selected labels {sorted(selected)} are valid reconstructed options."
            if valid
            else f"Selected labels {sorted(selected)} do not satisfy options {sorted(allowed)}."
        ),
    )


def _split_option_labels(answer: str) -> set[str]:
    stripped = answer.strip()
    if not stripped:
        return set()
    return {
        token
        for token in re.split(r"[\s,，;/+]+", stripped)
        if token
    }


def _verify_rubric_points(
    question: QuestionMasterDecision,
    rubric: list[RubricCriterionDecision],
) -> VerificationResult:
    if question.points is None:
        return VerificationResult(
            method=VerificationMethod.RUBRIC_POINTS,
            status=VerificationStatus.NOT_APPLICABLE,
            details="Printed total points are unavailable.",
        )
    if any(criterion.points is None for criterion in rubric):
        return VerificationResult(
            method=VerificationMethod.RUBRIC_POINTS,
            status=VerificationStatus.NOT_APPLICABLE,
            details="At least one rubric criterion has no explicit points.",
        )
    total = sum(criterion.points or 0 for criterion in rubric)
    passed = abs(total - question.points) < 1e-9
    return VerificationResult(
        method=VerificationMethod.RUBRIC_POINTS,
        status=VerificationStatus.PASSED if passed else VerificationStatus.FAILED,
        details=(
            f"Rubric points sum to the printed total {question.points:g}."
            if passed
            else f"Rubric points sum to {total:g}, expected {question.points:g}."
        ),
    )


def _verify_chemical_equations(
    answers: list[AnswerEntryDecision],
) -> VerificationResult:
    equations = [answer.answer for answer in answers if _EQUATION_ARROW.search(answer.answer)]
    if not equations:
        return VerificationResult(
            method=VerificationMethod.CHEMICAL_EQUATION_BALANCE,
            status=VerificationStatus.FAILED,
            details="No parseable equation arrow was found in the reference answers.",
        )
    failures: list[str] = []
    for equation in equations:
        try:
            left, right = _equation_counts(equation)
        except ValueError as exc:
            failures.append(f"{equation}: {exc}")
            continue
        if left != right:
            failures.append(f"{equation}: reactants={dict(left)}, products={dict(right)}")
    return VerificationResult(
        method=VerificationMethod.CHEMICAL_EQUATION_BALANCE,
        status=VerificationStatus.FAILED if failures else VerificationStatus.PASSED,
        details=(
            "; ".join(failures)
            if failures
            else f"Atom counts balance in {len(equations)} equation(s)."
        ),
    )


def _equation_counts(equation: str) -> tuple[Counter[str], Counter[str]]:
    sides = _EQUATION_ARROW.split(equation, maxsplit=1)
    if len(sides) != 2:
        raise ValueError("equation must contain one supported arrow")
    return _side_counts(sides[0]), _side_counts(sides[1])


def _side_counts(side: str) -> Counter[str]:
    terms = [term.strip() for term in re.split(r"\s+\+\s+", side.strip())]
    if not terms or any(not term for term in terms):
        raise ValueError("equation side contains an empty term")
    result: Counter[str] = Counter()
    for term in terms:
        match = re.fullmatch(r"(?:(\d+(?:\.\d+)?)\s*)?(.+)", term)
        if match is None:
            raise ValueError(f"cannot parse term {term!r}")
        coefficient = float(match.group(1) or 1)
        if not coefficient.is_integer():
            raise ValueError("fractional coefficients are not supported")
        formula = re.sub(r"\((?:aq|s|l|g)\)$", "", match.group(2).strip())
        counts = _formula_counts(formula)
        for element, count in counts.items():
            result[element] += int(coefficient) * count
    return result


def _formula_counts(formula: str) -> Counter[str]:
    formula = formula.replace("[", "(").replace("]", ")")
    stack: list[Counter[str]] = [Counter()]
    index = 0
    while index < len(formula):
        char = formula[index]
        if char == "(":
            stack.append(Counter())
            index += 1
            continue
        if char == ")":
            if len(stack) == 1:
                raise ValueError("unmatched closing parenthesis")
            group = stack.pop()
            multiplier, index = _read_number(formula, index + 1)
            for element, count in group.items():
                stack[-1][element] += count * multiplier
            continue
        match = _ELEMENT.match(formula, index)
        if match is None:
            raise ValueError(f"unsupported formula token at {formula[index:]!r}")
        element = match.group(0)
        multiplier, index = _read_number(formula, match.end())
        stack[-1][element] += multiplier
    if len(stack) != 1:
        raise ValueError("unclosed parenthesis")
    return stack[0]


def _read_number(value: str, index: int) -> tuple[int, int]:
    end = index
    while end < len(value) and value[end].isdigit():
        end += 1
    return (int(value[index:end]) if end > index else 1), end
