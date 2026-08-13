"""Phase 6 contracts for rubric grading and human score review."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import Field, model_validator

from student_analyze.document_models import (
    DecisionMethod,
    DocumentDecisionProvenance,
)
from student_analyze.exam_master_models import (
    AnswerEntryDecision,
    ChoiceOptionDecision,
    QuestionType,
    RubricCriterionDecision,
)
from student_analyze.models import Sha256, StableId, StrictModel
from student_analyze.page_models import Confidence
from student_analyze.submission_models import SubmissionCropAsset


GRADING_SCHEMA_VERSION = "1.0.0"
LOW_GRADING_CONFIDENCE = 0.8


class GradingRoute(str, Enum):
    AUTO_OBJECTIVE = "auto_objective"
    AUTO_BLANK = "auto_blank"
    LLM_RUBRIC = "llm_rubric"


class RubricEvaluationStatus(str, Enum):
    MET = "met"
    PARTIALLY_MET = "partially_met"
    NOT_MET = "not_met"
    UNDETERMINED = "undetermined"


class GradingMethod(str, Enum):
    AUTO_OBJECTIVE = "auto_objective"
    AUTO_BLANK = "auto_blank"
    LLM_RUBRIC = "llm_rubric"
    HUMAN_REVIEW = "human_review"


class AcademicErrorType(str, Enum):
    UNANSWERED = "unanswered"
    INCORRECT_OBJECTIVE = "incorrect_objective"
    CONCEPT_ERROR = "concept_error"
    CALCULATION_ERROR = "calculation_error"
    NOTATION_OR_EQUATION_ERROR = "notation_or_equation_error"
    REASONING_OMISSION = "reasoning_omission"
    OTHER = "other"


class GradingPhase(str, Enum):
    GRADED = "graded"
    REVIEWED = "reviewed"


class GradingReviewStatus(str, Enum):
    NOT_REQUIRED = "not_required"
    REQUIRED = "required"
    RESOLVED = "resolved"


class GradingResponseEvidence(StrictModel):
    submission_item_id: StableId
    slot_label: str = Field(min_length=1)
    observed_content: str | None = Field(default=None, min_length=1)
    normalized_answer: str | None = Field(default=None, min_length=1)
    is_blank: bool
    crop: SubmissionCropAsset

    @model_validator(mode="after")
    def validate_response(self) -> GradingResponseEvidence:
        if self.is_blank and (
            self.observed_content is not None or self.normalized_answer is not None
        ):
            raise ValueError("blank grading evidence cannot contain answer text")
        if not self.is_blank and self.observed_content is None:
            raise ValueError("resolved non-blank grading evidence requires observed content")
        return self


class GradingTargetInput(StrictModel):
    target_id: StableId
    question_id: StableId
    version_id: StableId
    part_id: StableId | None = None
    printed_label: str = Field(min_length=1)
    prompt_text: str = Field(min_length=1)
    question_type: QuestionType
    max_points: float = Field(ge=0)
    options: list[ChoiceOptionDecision] = Field(default_factory=list)
    reference_answers: list[AnswerEntryDecision] = Field(min_length=1)
    rubric: list[RubricCriterionDecision] = Field(min_length=1)
    solution_summary: str | None = Field(default=None, min_length=1)
    assumptions: list[str] = Field(default_factory=list)
    knowledge_points: list[str] = Field(default_factory=list)
    responses: list[GradingResponseEvidence] = Field(min_length=1)
    route: GradingRoute
    mandatory_review: bool = False

    @model_validator(mode="after")
    def validate_target(self) -> GradingTargetInput:
        response_ids = [item.submission_item_id for item in self.responses]
        if len(response_ids) != len(set(response_ids)):
            raise ValueError("grading target response evidence must be unique")
        rubric_refs = [item.ref for item in self.rubric]
        if len(rubric_refs) != len(set(rubric_refs)):
            raise ValueError("grading target rubric refs must be unique")
        if any(item.points is None for item in self.rubric):
            raise ValueError("grading requires explicit points for every rubric criterion")
        rubric_total = sum(item.points or 0 for item in self.rubric)
        if abs(rubric_total - self.max_points) > 1e-9:
            raise ValueError("grading target max_points must equal its rubric total")
        if any(
            answer.part_ref is not None and answer.part_ref != self.part_id
            for answer in self.reference_answers
        ):
            raise ValueError("grading target contains an answer for another part")
        if any(
            criterion.part_ref is not None and criterion.part_ref != self.part_id
            for criterion in self.rubric
        ):
            raise ValueError("grading target contains a rubric for another part")
        all_blank = all(item.is_blank for item in self.responses)
        if self.route == GradingRoute.AUTO_BLANK and not all_blank:
            raise ValueError("auto-blank grading requires every formal response to be blank")
        if self.route != GradingRoute.AUTO_BLANK and all_blank:
            raise ValueError("blank grading targets must use auto_blank")
        objective = self.question_type in {
            QuestionType.OBJECTIVE_SINGLE,
            QuestionType.OBJECTIVE_MULTIPLE,
        }
        if self.route == GradingRoute.AUTO_OBJECTIVE and not objective:
            raise ValueError("only objective questions may use auto_objective")
        if self.route == GradingRoute.LLM_RUBRIC and objective:
            raise ValueError("non-blank objective questions must be graded automatically")
        return self


class GradingInputManifest(StrictModel):
    schema_version: Literal[GRADING_SCHEMA_VERSION] = GRADING_SCHEMA_VERSION
    case_id: StableId
    exam_master_sha256: Sha256
    submission_sha256: Sha256
    manifest_fingerprint: Sha256
    created_at: datetime
    targets: list[GradingTargetInput] = Field(min_length=1)
    llm_target_ids: list[StableId] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_targets(self) -> GradingInputManifest:
        target_ids = [item.target_id for item in self.targets]
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("grading input targets must be unique")
        expected = [
            item.target_id
            for item in self.targets
            if item.route == GradingRoute.LLM_RUBRIC
        ]
        if self.llm_target_ids != expected:
            raise ValueError("llm_target_ids must preserve the routed target order")
        return self


class RubricEvaluation(StrictModel):
    rubric_ref: StableId
    status: RubricEvaluationStatus
    awarded_points: float | None = Field(default=None, ge=0)
    evidence_item_ids: list[StableId] = Field(default_factory=list)
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_points(self) -> RubricEvaluation:
        if self.status == RubricEvaluationStatus.UNDETERMINED:
            if self.awarded_points is not None:
                raise ValueError("undetermined rubric evaluations cannot award points")
        elif self.awarded_points is None:
            raise ValueError("determined rubric evaluations require awarded_points")
        if self.status == RubricEvaluationStatus.NOT_MET and self.awarded_points != 0:
            raise ValueError("not-met rubric evaluations must award zero points")
        return self


class AcademicErrorDiagnosis(StrictModel):
    error_type: AcademicErrorType
    rubric_refs: list[StableId] = Field(min_length=1)
    evidence_item_ids: list[StableId] = Field(min_length=1)
    diagnosis: str = Field(min_length=1)


class GradingTargetDecision(StrictModel):
    target_id: StableId
    rubric_evaluations: list[RubricEvaluation] = Field(min_length=1)
    error_diagnoses: list[AcademicErrorDiagnosis] = Field(default_factory=list)
    confidence: Confidence
    visual_judgment_required: bool = False
    requires_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_decision(self) -> GradingTargetDecision:
        rubric_refs = [item.rubric_ref for item in self.rubric_evaluations]
        if len(rubric_refs) != len(set(rubric_refs)):
            raise ValueError("grading decisions must evaluate each rubric at most once")
        if self.confidence < LOW_GRADING_CONFIDENCE and not self.requires_review:
            raise ValueError("low-confidence grading decisions must require review")
        if any(
            item.status == RubricEvaluationStatus.UNDETERMINED
            for item in self.rubric_evaluations
        ) and not self.requires_review:
            raise ValueError("undetermined rubric evaluations must require review")
        if self.visual_judgment_required and not self.requires_review:
            raise ValueError("visual grading judgments must require review")
        if self.requires_review != bool(self.review_reasons):
            raise ValueError("review_reasons must reflect requires_review")
        return self


class GradingDecisionSet(StrictModel):
    schema_version: Literal[GRADING_SCHEMA_VERSION] = GRADING_SCHEMA_VERSION
    case_id: StableId
    grading_input_manifest_sha256: Sha256
    provenance: DocumentDecisionProvenance
    decisions: list[GradingTargetDecision] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_decisions(self) -> GradingDecisionSet:
        target_ids = [item.target_id for item in self.decisions]
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("grading decisions must contain unique targets")
        return self


class GradingTargetResult(StrictModel):
    target_id: StableId
    question_id: StableId
    version_id: StableId
    part_id: StableId | None = None
    submission_item_ids: list[StableId] = Field(min_length=1)
    method: GradingMethod
    rubric_evaluations: list[RubricEvaluation] = Field(min_length=1)
    error_diagnoses: list[AcademicErrorDiagnosis] = Field(default_factory=list)
    proposed_score: float | None = Field(default=None, ge=0)
    final_score: float | None = Field(default=None, ge=0)
    max_points: float = Field(ge=0)
    confidence: Confidence
    review_status: GradingReviewStatus
    review_reason: str | None = Field(default=None, min_length=1)
    review_note: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_result(self) -> GradingTargetResult:
        if self.proposed_score is not None and self.proposed_score > self.max_points:
            raise ValueError("proposed score cannot exceed max_points")
        if self.final_score is not None and self.final_score > self.max_points:
            raise ValueError("final score cannot exceed max_points")
        if self.review_status == GradingReviewStatus.REQUIRED:
            if (
                self.review_reason is None
                or self.review_note is not None
                or self.final_score is not None
            ):
                raise ValueError("required review needs a reason and no final score")
        elif self.review_reason is not None:
            raise ValueError("only required review may retain a review reason")
        if self.review_status == GradingReviewStatus.RESOLVED:
            if (
                self.method != GradingMethod.HUMAN_REVIEW
                or self.final_score is None
                or self.review_note is None
            ):
                raise ValueError("resolved grading results require a human final score")
        elif self.review_note is not None:
            raise ValueError("only resolved review may retain a review note")
        return self


class GradingReviewItem(StrictModel):
    target_id: StableId
    reasons: list[str] = Field(min_length=1)


class QuestionScoreSummary(StrictModel):
    question_id: StableId
    target_ids: list[StableId] = Field(min_length=1)
    proposed_score: float | None = Field(default=None, ge=0)
    final_score: float | None = Field(default=None, ge=0)
    max_score: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_score(self) -> QuestionScoreSummary:
        if len(self.target_ids) != len(set(self.target_ids)):
            raise ValueError("question score target IDs must be unique")
        if self.proposed_score is not None and self.proposed_score > self.max_score:
            raise ValueError("question proposed score cannot exceed max_score")
        if self.final_score is not None and self.final_score > self.max_score:
            raise ValueError("question final score cannot exceed max_score")
        return self


class Grading(StrictModel):
    schema_version: Literal[GRADING_SCHEMA_VERSION] = GRADING_SCHEMA_VERSION
    phase: GradingPhase
    case_id: StableId
    exam_master_sha256: Sha256
    submission_sha256: Sha256
    grading_input_manifest_fingerprint: Sha256
    grading_input_manifest_sha256: Sha256
    grading_decision_sha256: Sha256
    source_grading_sha256: Sha256 | None = None
    review_decision_sha256: Sha256 | None = None
    stage_fingerprint: Sha256
    created_at: datetime
    grading_provenance: DocumentDecisionProvenance
    review_provenance: DocumentDecisionProvenance | None = None
    input_manifest: GradingInputManifest
    targets: list[GradingTargetResult] = Field(min_length=1)
    questions: list[QuestionScoreSummary] = Field(min_length=1)
    provisional_score: float | None = Field(default=None, ge=0)
    final_score: float | None = Field(default=None, ge=0)
    max_score: float = Field(ge=0)
    review_items: list[GradingReviewItem] = Field(default_factory=list)
    requires_review: bool = False
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_grading(self) -> Grading:
        target_ids = [item.target_id for item in self.targets]
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("grading targets must be unique")
        question_ids = [item.question_id for item in self.questions]
        if len(question_ids) != len(set(question_ids)):
            raise ValueError("grading question summaries must be unique")
        summarized = [target_id for item in self.questions for target_id in item.target_ids]
        if len(summarized) != len(set(summarized)) or set(summarized) != set(target_ids):
            raise ValueError("question summaries must cover every grading target once")
        targets_by_id = {item.target_id: item for item in self.targets}
        for question in self.questions:
            grouped = [targets_by_id[target_id] for target_id in question.target_ids]
            if any(item.question_id != question.question_id for item in grouped):
                raise ValueError("question summary contains a target from another question")
            expected_max = sum(item.max_points for item in grouped)
            if abs(expected_max - question.max_score) > 1e-9:
                raise ValueError("question max_score must equal its target maximum total")
            proposed = [item.proposed_score for item in grouped]
            expected_proposed = (
                None if any(item is None for item in proposed) else sum(proposed)
            )
            if question.proposed_score != expected_proposed:
                raise ValueError("question proposed_score must equal its target total")
            final = [item.final_score for item in grouped]
            expected_final = None if any(item is None for item in final) else sum(final)
            if question.final_score != expected_final:
                raise ValueError("question final_score must equal its target total")
        review_ids = [item.target_id for item in self.review_items]
        if len(review_ids) != len(set(review_ids)):
            raise ValueError("grading review items must be unique")
        flagged = {
            item.target_id
            for item in self.targets
            if item.review_status == GradingReviewStatus.REQUIRED
        }
        if set(review_ids) != flagged:
            raise ValueError("review items must cover every required grading target")
        if self.requires_review != bool(self.review_items):
            raise ValueError("requires_review must reflect grading review items")
        expected_max = sum(item.max_points for item in self.targets)
        if abs(expected_max - self.max_score) > 1e-9:
            raise ValueError("max_score must equal the target maximum total")
        proposed = [item.proposed_score for item in self.targets]
        expected_provisional = None if any(item is None for item in proposed) else sum(proposed)
        if self.provisional_score != expected_provisional:
            raise ValueError("provisional_score must equal determined target proposals")

        if self.phase == GradingPhase.GRADED:
            if any(
                value is not None
                for value in (
                    self.source_grading_sha256,
                    self.review_decision_sha256,
                    self.review_provenance,
                    self.final_score,
                )
            ):
                raise ValueError("graded artifacts cannot contain final review fields")
            if any(item.review_status == GradingReviewStatus.RESOLVED for item in self.targets):
                raise ValueError("graded artifacts cannot contain resolved review results")
            if any(item.final_score is not None for item in self.targets):
                raise ValueError("graded artifacts cannot expose target final scores")
        else:
            if (
                self.source_grading_sha256 is None
                or self.review_decision_sha256 is None
                or self.review_provenance is None
            ):
                raise ValueError("reviewed artifacts require review provenance and hashes")
            if self.requires_review:
                raise ValueError("reviewed artifacts cannot retain review items")
            if any(item.final_score is None for item in self.targets):
                raise ValueError("reviewed artifacts require every final target score")
            expected_final = sum(item.final_score or 0 for item in self.targets)
            if self.final_score != expected_final:
                raise ValueError("final_score must equal the reviewed target total")
        return self


class ReviewedTargetDecision(StrictModel):
    target_id: StableId
    rubric_evaluations: list[RubricEvaluation] = Field(min_length=1)
    error_diagnoses: list[AcademicErrorDiagnosis] = Field(default_factory=list)
    decision_reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_evaluations(self) -> ReviewedTargetDecision:
        refs = [item.rubric_ref for item in self.rubric_evaluations]
        if len(refs) != len(set(refs)):
            raise ValueError("human review must evaluate each rubric at most once")
        if any(
            item.status == RubricEvaluationStatus.UNDETERMINED
            for item in self.rubric_evaluations
        ):
            raise ValueError("human review cannot leave rubric evaluations undetermined")
        return self


class GradingReviewDecisionSet(StrictModel):
    schema_version: Literal[GRADING_SCHEMA_VERSION] = GRADING_SCHEMA_VERSION
    case_id: StableId
    source_grading_sha256: Sha256
    provenance: DocumentDecisionProvenance
    decisions: list[ReviewedTargetDecision] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_review(self) -> GradingReviewDecisionSet:
        if self.provenance.method != DecisionMethod.HUMAN_REVIEW:
            raise ValueError("grading review decisions must come from human_review")
        target_ids = [item.target_id for item in self.decisions]
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("grading review decisions must contain unique targets")
        return self


class GradingReviewContextItem(StrictModel):
    target: GradingTargetInput
    proposed_result: GradingTargetResult
    reasons: list[str] = Field(min_length=1)


class GradingReviewManifest(StrictModel):
    schema_version: Literal[GRADING_SCHEMA_VERSION] = GRADING_SCHEMA_VERSION
    case_id: StableId
    source_grading_sha256: Sha256
    created_at: datetime
    items: list[GradingReviewContextItem] = Field(default_factory=list)
