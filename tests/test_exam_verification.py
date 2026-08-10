from student_analyze.exam_master_models import (
    AnswerEntryDecision,
    QuestionMasterDecision,
    QuestionType,
    RubricCriterionDecision,
    VerificationMethod,
    VerificationStatus,
)
from student_analyze.exam_verification import run_deterministic_verifications


def test_balanced_chemical_equation_passes_atom_check() -> None:
    question = QuestionMasterDecision(
        question_id="question-1",
        version_id="version-1",
        prompt_text="Write the equation.",
        question_type=QuestionType.FILL_BLANK,
        points=2,
        solver_performed=True,
        reference_answers=[
            AnswerEntryDecision(answer="Se + H2O + CO -> H2Se + CO2")
        ],
        rubric=[
            RubricCriterionDecision(
                ref="balanced-equation",
                description="Balanced equation.",
                points=2,
            )
        ],
        solution_summary="Balance atoms on both sides.",
        verification_requests=[VerificationMethod.CHEMICAL_EQUATION_BALANCE],
    )
    results = run_deterministic_verifications(
        question,
        question.reference_answers,
        question.rubric,
    )
    balance = next(
        result
        for result in results
        if result.method == VerificationMethod.CHEMICAL_EQUATION_BALANCE
    )
    assert balance.status == VerificationStatus.PASSED


def test_unbalanced_chemical_equation_fails_atom_check() -> None:
    question = QuestionMasterDecision(
        question_id="question-1",
        version_id="version-1",
        prompt_text="Write the equation.",
        question_type=QuestionType.FILL_BLANK,
        solver_performed=True,
        reference_answers=[AnswerEntryDecision(answer="H2 + O2 -> H2O")],
        rubric=[
            RubricCriterionDecision(
                ref="balanced-equation",
                description="Balanced equation.",
            )
        ],
        solution_summary="Balance atoms on both sides.",
        verification_requests=[VerificationMethod.CHEMICAL_EQUATION_BALANCE],
    )
    results = run_deterministic_verifications(
        question,
        question.reference_answers,
        question.rubric,
    )
    balance = next(
        result
        for result in results
        if result.method == VerificationMethod.CHEMICAL_EQUATION_BALANCE
    )
    assert balance.status == VerificationStatus.FAILED
