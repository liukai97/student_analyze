"""Phase 4 contracts for answer evidence, blind solving, and Exam Master."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import Field, model_validator

from student_analyze.document_models import DocumentDecisionProvenance
from student_analyze.models import Sha256, StableId, StrictModel
from student_analyze.page_models import Confidence, ImageSize, PixelBox


EXAM_MASTER_SCHEMA_VERSION = "1.0.0"


class EvidenceSourceKind(str, Enum):
    OFFICIAL_ANSWER = "official_answer"
    TEACHER_ANNOTATION = "teacher_annotation"


class EvidenceDecisionStatus(str, Enum):
    ACCEPTED = "accepted"
    CANDIDATE = "candidate"


class AnnotationActor(str, Enum):
    PRINTED = "printed"
    TEACHER = "teacher"
    STUDENT = "student"
    UNKNOWN = "unknown"


class EndorsementKind(str, Enum):
    OFFICIAL_KEY = "official_key"
    CORRECT = "correct"
    INCORRECT = "incorrect"
    FULL_SCORE = "full_score"
    ALL_CORRECT = "all_correct"
    CORRECTED_ANSWER = "corrected_answer"
    SCORE = "score"
    REVIEWED_ONLY = "reviewed_only"
    UNKNOWN = "unknown"


class ScopeKind(str, Enum):
    QUESTION = "question"
    QUESTIONS = "questions"
    SECTION = "section"
    EXAM = "exam"


class AnswerSupport(str, Enum):
    NONE = "none"
    ANSWER_KEY = "answer_key"
    ACCEPTED_EXAMPLE = "accepted_example"


class RubricSupport(str, Enum):
    NONE = "none"
    EXACT_MATCH = "exact_match"
    COMPLETE = "complete"
    ACCEPTED_EXAMPLE = "accepted_example"


class EvidenceAnswerDecision(StrictModel):
    question_id: StableId
    printed_part_label: str | None = None
    answer: str = Field(min_length=1)
    acceptable_alternatives: list[str] = Field(default_factory=list)


class EvidenceRubricDecision(StrictModel):
    ref: StableId
    question_id: StableId
    printed_part_label: str | None = None
    description: str = Field(min_length=1)
    points: float | None = Field(default=None, ge=0)


class AnswerEvidenceDecision(StrictModel):
    ref: StableId
    source_kind: EvidenceSourceKind
    page_id: StableId
    bbox: PixelBox
    observed_content: str = Field(min_length=1)
    actor: AnnotationActor
    endorsement_kind: EndorsementKind
    scope_kind: ScopeKind
    question_ids: list[StableId] = Field(min_length=1)
    answer_support: AnswerSupport
    rubric_support: RubricSupport
    answers: list[EvidenceAnswerDecision] = Field(default_factory=list)
    rubric: list[EvidenceRubricDecision] = Field(default_factory=list)
    score: float | None = Field(default=None, ge=0)
    max_score: float | None = Field(default=None, gt=0)
    status: EvidenceDecisionStatus
    confidence: Confidence
    evidence: list[str] = Field(min_length=1)
    human_confirmed: bool = False
    requires_review: bool
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_evidence_decision(self) -> AnswerEvidenceDecision:
        if len(self.question_ids) != len(set(self.question_ids)):
            raise ValueError("answer evidence question_ids must be unique")
        if self.scope_kind == ScopeKind.QUESTION and len(self.question_ids) != 1:
            raise ValueError("question scope must target exactly one question")

        answer_keys = [
            (answer.question_id, answer.printed_part_label) for answer in self.answers
        ]
        if len(answer_keys) != len(set(answer_keys)):
            raise ValueError("answer evidence answers must target unique question parts")
        if any(answer.question_id not in self.question_ids for answer in self.answers):
            raise ValueError("answer evidence answers must stay inside source scope")
        rubric_refs = [criterion.ref for criterion in self.rubric]
        if len(rubric_refs) != len(set(rubric_refs)):
            raise ValueError("answer evidence rubric refs must be unique")
        if any(criterion.question_id not in self.question_ids for criterion in self.rubric):
            raise ValueError("answer evidence rubric must stay inside source scope")

        if self.status == EvidenceDecisionStatus.ACCEPTED and self.requires_review:
            raise ValueError("accepted evidence cannot require review")
        if self.status == EvidenceDecisionStatus.CANDIDATE and not self.requires_review:
            raise ValueError("candidate evidence must require review")
        if self.actor == AnnotationActor.UNKNOWN and not self.requires_review:
            raise ValueError("unknown annotation actors must require review")

        if self.answer_support == AnswerSupport.NONE and self.answers:
            raise ValueError("evidence without answer support cannot contain answers")
        if self.answer_support != AnswerSupport.NONE and not self.answers:
            raise ValueError("answer-supporting evidence must contain answers")
        if (
            self.rubric_support in {RubricSupport.EXACT_MATCH, RubricSupport.COMPLETE}
            and self.answer_support != AnswerSupport.ANSWER_KEY
        ):
            raise ValueError("exact or complete rubric support requires an answer key")
        if (
            self.rubric_support == RubricSupport.ACCEPTED_EXAMPLE
            and self.answer_support != AnswerSupport.ACCEPTED_EXAMPLE
        ):
            raise ValueError("accepted-example rubric support requires an accepted example")
        if self.rubric_support == RubricSupport.COMPLETE and not self.rubric:
            raise ValueError("complete rubric support requires rubric criteria")
        if self.rubric_support != RubricSupport.COMPLETE and self.rubric:
            raise ValueError("only complete rubric support may contain rubric criteria")

        if (self.score is None) != (self.max_score is None):
            raise ValueError("score and max_score must be provided together")
        if self.score is not None and self.max_score is not None and self.score > self.max_score:
            raise ValueError("score cannot exceed max_score")

        if self.source_kind == EvidenceSourceKind.OFFICIAL_ANSWER:
            if self.actor != AnnotationActor.PRINTED:
                raise ValueError("official answers must use the printed actor")
            if self.endorsement_kind != EndorsementKind.OFFICIAL_KEY:
                raise ValueError("official answers must use official_key endorsement")
        if (
            self.source_kind == EvidenceSourceKind.TEACHER_ANNOTATION
            and self.status == EvidenceDecisionStatus.ACCEPTED
        ):
            if self.actor != AnnotationActor.TEACHER or not self.human_confirmed:
                raise ValueError(
                    "accepted teacher evidence requires confirmed teacher authorship"
                )
        if (
            self.source_kind == EvidenceSourceKind.TEACHER_ANNOTATION
            and self.endorsement_kind
            in {
                EndorsementKind.INCORRECT,
                EndorsementKind.SCORE,
                EndorsementKind.REVIEWED_ONLY,
                EndorsementKind.UNKNOWN,
            }
            and self.answer_support != AnswerSupport.NONE
        ):
            raise ValueError(
                "incorrect, score-only, reviewed-only, or unknown teacher marks "
                "cannot support an answer key"
            )
        if self.human_confirmed and self.source_kind != EvidenceSourceKind.TEACHER_ANNOTATION:
            raise ValueError("human_confirmed is reserved for teacher-annotation semantics")
        return self


class AnswerEvidenceDecisionSet(StrictModel):
    schema_version: Literal[EXAM_MASTER_SCHEMA_VERSION] = EXAM_MASTER_SCHEMA_VERSION
    case_id: StableId
    document_graph_sha256: Sha256
    provenance: DocumentDecisionProvenance
    items: list[AnswerEvidenceDecision] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_items(self) -> AnswerEvidenceDecisionSet:
        refs = [item.ref for item in self.items]
        if len(refs) != len(set(refs)):
            raise ValueError("answer evidence refs must be unique")
        return self


class QuestionType(str, Enum):
    OBJECTIVE_SINGLE = "objective_single"
    OBJECTIVE_MULTIPLE = "objective_multiple"
    FILL_BLANK = "fill_blank"
    SHORT_ANSWER = "short_answer"
    CALCULATION = "calculation"
    ESSAY = "essay"
    DIAGRAM = "diagram"
    OTHER = "other"


class ChoiceOptionDecision(StrictModel):
    label: str = Field(min_length=1)
    text: str = Field(min_length=1)


class SubpartDecision(StrictModel):
    ref: StableId
    printed_label: str = Field(min_length=1)
    prompt_text: str = Field(min_length=1)
    points: float | None = Field(default=None, ge=0)


class QuestionReconstructionDecision(StrictModel):
    question_id: StableId
    version_id: StableId
    prompt_text: str = Field(min_length=1)
    question_type: QuestionType
    points: float | None = Field(default=None, ge=0)
    options: list[ChoiceOptionDecision] = Field(default_factory=list)
    subparts: list[SubpartDecision] = Field(default_factory=list)
    knowledge_points: list[str] = Field(default_factory=list)
    visual_region_ids: list[StableId] = Field(default_factory=list)
    source_contains_student_content: bool
    source_contains_teacher_annotation: bool
    human_confirmed: bool = False
    confidence: Confidence
    evidence: list[str] = Field(min_length=1)
    uncertainty: list[str] = Field(default_factory=list)
    requires_review: bool
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_reconstruction(self) -> QuestionReconstructionDecision:
        option_labels = [option.label for option in self.options]
        if len(option_labels) != len(set(option_labels)):
            raise ValueError("reconstructed choice option labels must be unique")
        part_refs = [part.ref for part in self.subparts]
        if len(part_refs) != len(set(part_refs)):
            raise ValueError("reconstructed subpart refs must be unique")
        if len(self.visual_region_ids) != len(set(self.visual_region_ids)):
            raise ValueError("visual_region_ids must be unique")
        if self.question_type in {
            QuestionType.OBJECTIVE_SINGLE,
            QuestionType.OBJECTIVE_MULTIPLE,
        } and not self.options:
            raise ValueError("reconstructed objective questions must include options")
        contaminated = (
            self.source_contains_student_content
            or self.source_contains_teacher_annotation
        )
        if contaminated and not self.human_confirmed and not self.requires_review:
            raise ValueError(
                "reconstructions from marked sources require human confirmation or review"
            )
        if self.uncertainty and not self.requires_review:
            raise ValueError("uncertain reconstructions must require review")
        return self


class QuestionReconstructionDecisionSet(StrictModel):
    schema_version: Literal[EXAM_MASTER_SCHEMA_VERSION] = EXAM_MASTER_SCHEMA_VERSION
    case_id: StableId
    document_graph_sha256: Sha256
    provenance: DocumentDecisionProvenance
    questions: list[QuestionReconstructionDecision] = Field(min_length=1)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_questions(self) -> QuestionReconstructionDecisionSet:
        question_ids = [question.question_id for question in self.questions]
        if len(question_ids) != len(set(question_ids)):
            raise ValueError("question reconstructions must be unique")
        return self


class CropReviewStatus(str, Enum):
    VISUALLY_REVIEWED = "visually_reviewed"
    HUMAN_REVIEWED = "human_reviewed"
    DETERMINISTIC_TEST_FIXTURE = "deterministic_test_fixture"


class SolverCropDecision(StrictModel):
    ref: StableId
    question_id: StableId
    version_id: StableId
    source_region_id: StableId
    page_id: StableId
    bbox: PixelBox
    review_status: CropReviewStatus
    contains_student_handwriting: bool
    contains_teacher_annotation: bool
    contains_answer_content: bool
    confidence: Confidence
    evidence: list[str] = Field(min_length=1)
    requires_review: bool = False
    warnings: list[str] = Field(default_factory=list)


class SolverInputDecisionSet(StrictModel):
    schema_version: Literal[EXAM_MASTER_SCHEMA_VERSION] = EXAM_MASTER_SCHEMA_VERSION
    case_id: StableId
    document_graph_sha256: Sha256
    provenance: DocumentDecisionProvenance
    crops: list[SolverCropDecision] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_crops(self) -> SolverInputDecisionSet:
        refs = [crop.ref for crop in self.crops]
        if len(refs) != len(set(refs)):
            raise ValueError("solver crop refs must be unique")
        return self


class SolverTask(str, Enum):
    RECONSTRUCT_ONLY = "reconstruct_only"
    RECONSTRUCT_AND_SOLVE = "reconstruct_and_solve"


class SolverRouteReason(str, Enum):
    RELIABLE_EVIDENCE = "reliable_evidence"
    NO_RELIABLE_EVIDENCE = "no_reliable_evidence"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    INCOMPLETE_RUBRIC = "incomplete_rubric"


class SolverCropAsset(StrictModel):
    asset_id: StableId
    crop_decision_ref: StableId
    question_id: StableId
    version_id: StableId
    source_region_id: StableId
    page_id: StableId
    bbox: PixelBox
    relative_path: str = Field(min_length=1)
    sha256: Sha256
    size_bytes: int = Field(gt=0)
    media_type: Literal["image/jpeg"] = "image/jpeg"
    image_size: ImageSize

    @model_validator(mode="after")
    def validate_relative_path(self) -> SolverCropAsset:
        normalized = self.relative_path.replace("\\", "/")
        parts = normalized.split("/")
        if normalized.startswith("/") or any(part in {"", ".", ".."} for part in parts):
            raise ValueError("solver crop path must stay below the case directory")
        if normalized != self.relative_path:
            raise ValueError("solver crop path must use forward slashes")
        return self


class SolverQuestionInput(StrictModel):
    question_id: StableId
    version_id: StableId
    printed_label: str = Field(min_length=1)
    task: SolverTask
    route_reason: SolverRouteReason
    reconstruction: QuestionReconstructionDecision
    assets: list[SolverCropAsset] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_route(self) -> SolverQuestionInput:
        if any(asset.question_id != self.question_id for asset in self.assets):
            raise ValueError("solver assets must belong to the containing question")
        if any(asset.version_id != self.version_id for asset in self.assets):
            raise ValueError("solver assets must belong to the effective version")
        if (
            self.reconstruction.question_id != self.question_id
            or self.reconstruction.version_id != self.version_id
        ):
            raise ValueError("solver reconstruction must belong to the containing question")
        requested_regions = set(self.reconstruction.visual_region_ids)
        actual_regions = {asset.source_region_id for asset in self.assets}
        if actual_regions != requested_regions:
            raise ValueError(
                "solver assets must cover exactly the requested clean visual regions"
            )
        if (
            self.task == SolverTask.RECONSTRUCT_ONLY
            and self.route_reason != SolverRouteReason.RELIABLE_EVIDENCE
        ):
            raise ValueError("reconstruct-only questions require reliable evidence")
        if (
            self.task == SolverTask.RECONSTRUCT_AND_SOLVE
            and self.route_reason == SolverRouteReason.RELIABLE_EVIDENCE
        ):
            raise ValueError("reliable evidence must not be routed to mandatory solving")
        return self


class SolverInputManifest(StrictModel):
    schema_version: Literal[EXAM_MASTER_SCHEMA_VERSION] = EXAM_MASTER_SCHEMA_VERSION
    case_id: StableId
    document_graph_sha256: Sha256
    answer_evidence_decision_sha256: Sha256
    question_reconstruction_decision_sha256: Sha256
    crop_decision_sha256: Sha256
    manifest_fingerprint: Sha256
    created_at: datetime
    question_reconstruction_provenance: DocumentDecisionProvenance
    denied_document_roles: list[str] = Field(
        default_factory=lambda: ["answer_sheet", "scratch"]
    )
    questions: list[SolverQuestionInput] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_questions(self) -> SolverInputManifest:
        question_ids = [question.question_id for question in self.questions]
        if len(question_ids) != len(set(question_ids)):
            raise ValueError("solver manifest questions must be unique")
        asset_ids = [asset.asset_id for question in self.questions for asset in question.assets]
        paths = [asset.relative_path for question in self.questions for asset in question.assets]
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("solver manifest asset IDs must be unique")
        if len(paths) != len(set(paths)):
            raise ValueError("solver manifest asset paths must be unique")
        return self


class AnswerEntryDecision(StrictModel):
    part_ref: StableId | None = None
    answer: str = Field(min_length=1)
    acceptable_alternatives: list[str] = Field(default_factory=list)


class RubricCriterionDecision(StrictModel):
    ref: StableId
    part_ref: StableId | None = None
    description: str = Field(min_length=1)
    points: float | None = Field(default=None, ge=0)


class VerificationMethod(str, Enum):
    OPTION_MEMBERSHIP = "option_membership"
    RUBRIC_POINTS = "rubric_points"
    CHEMICAL_EQUATION_BALANCE = "chemical_equation_balance"
    NUMERIC_CONSISTENCY = "numeric_consistency"
    UNIT_CONSISTENCY = "unit_consistency"
    LOGIC_REVIEW = "logic_review"


class QuestionMasterDecision(StrictModel):
    question_id: StableId
    version_id: StableId
    prompt_text: str = Field(min_length=1)
    question_type: QuestionType
    points: float | None = Field(default=None, ge=0)
    options: list[ChoiceOptionDecision] = Field(default_factory=list)
    subparts: list[SubpartDecision] = Field(default_factory=list)
    knowledge_points: list[str] = Field(default_factory=list)
    solver_performed: bool
    reference_answers: list[AnswerEntryDecision] = Field(default_factory=list)
    rubric: list[RubricCriterionDecision] = Field(default_factory=list)
    solution_summary: str | None = None
    assumptions: list[str] = Field(default_factory=list)
    uncertainty: list[str] = Field(default_factory=list)
    verification_requests: list[VerificationMethod] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_question_decision(self) -> QuestionMasterDecision:
        option_labels = [option.label for option in self.options]
        if len(option_labels) != len(set(option_labels)):
            raise ValueError("choice option labels must be unique")
        part_refs = [part.ref for part in self.subparts]
        if len(part_refs) != len(set(part_refs)):
            raise ValueError("subpart refs must be unique")
        part_ref_set = set(part_refs)
        if any(
            answer.part_ref is not None and answer.part_ref not in part_ref_set
            for answer in self.reference_answers
        ):
            raise ValueError("reference answers contain an unknown part_ref")
        if any(
            criterion.part_ref is not None and criterion.part_ref not in part_ref_set
            for criterion in self.rubric
        ):
            raise ValueError("rubric criteria contain an unknown part_ref")
        rubric_refs = [criterion.ref for criterion in self.rubric]
        if len(rubric_refs) != len(set(rubric_refs)):
            raise ValueError("rubric criterion refs must be unique")
        if self.question_type in {
            QuestionType.OBJECTIVE_SINGLE,
            QuestionType.OBJECTIVE_MULTIPLE,
        } and not self.options:
            raise ValueError("objective questions must include options")

        if self.solver_performed:
            if not self.reference_answers or not self.rubric or not self.solution_summary:
                raise ValueError(
                    "solved questions require answers, rubric, and solution summary"
                )
        elif self.reference_answers or self.rubric or self.solution_summary is not None:
            raise ValueError(
                "reconstruct-only questions must not emit answers, rubric, or solution"
            )
        return self


class ExamMasterDecisionSet(StrictModel):
    schema_version: Literal[EXAM_MASTER_SCHEMA_VERSION] = EXAM_MASTER_SCHEMA_VERSION
    case_id: StableId
    document_graph_sha256: Sha256
    solver_input_manifest_sha256: Sha256
    provenance: DocumentDecisionProvenance
    questions: list[QuestionMasterDecision] = Field(min_length=1)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_questions(self) -> ExamMasterDecisionSet:
        question_ids = [question.question_id for question in self.questions]
        if len(question_ids) != len(set(question_ids)):
            raise ValueError("Exam Master decisions must contain unique questions")
        return self


class IndependentReviewStatus(str, Enum):
    CONFIRMED = "confirmed"
    CONFLICT = "conflict"
    UNCERTAIN = "uncertain"


class QuestionReviewDecision(StrictModel):
    question_id: StableId
    version_id: StableId
    status: IndependentReviewStatus
    independently_derived_answer: str = Field(min_length=1)
    evidence: list[str] = Field(min_length=1)
    warnings: list[str] = Field(default_factory=list)


class ExamReviewDecisionSet(StrictModel):
    schema_version: Literal[EXAM_MASTER_SCHEMA_VERSION] = EXAM_MASTER_SCHEMA_VERSION
    case_id: StableId
    solver_input_manifest_sha256: Sha256
    exam_master_decision_sha256: Sha256
    provenance: DocumentDecisionProvenance
    reviews: list[QuestionReviewDecision] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_reviews(self) -> ExamReviewDecisionSet:
        question_ids = [review.question_id for review in self.reviews]
        if len(question_ids) != len(set(question_ids)):
            raise ValueError("independent reviews must contain unique questions")
        return self


class AnswerSource(str, Enum):
    OFFICIAL_ANSWER = "official_answer"
    TEACHER_ENDORSED_SUBMISSION = "teacher_endorsed_submission"
    TEACHER_CORRECTION = "teacher_correction"
    INDEPENDENT_SOLUTION = "independent_solution"


class VerificationLevel(str, Enum):
    SOURCE_VALIDATED = "source_validated"
    HUMAN_CONFIRMED = "human_confirmed"
    INDEPENDENTLY_REVIEWED = "independently_reviewed"


class VerificationStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_APPLICABLE = "not_applicable"
    REQUIRES_REVIEW = "requires_review"


class ApprovalStatus(str, Enum):
    APPROVED = "approved"
    PROVISIONAL = "provisional"
    REQUIRES_REVIEW = "requires_review"


class VerificationResult(StrictModel):
    method: VerificationMethod
    status: VerificationStatus
    details: str = Field(min_length=1)


class AnswerProvenance(StrictModel):
    source: AnswerSource
    evidence_refs: list[StableId] = Field(default_factory=list)
    source_scope: list[StableId] = Field(min_length=1)
    verification_level: VerificationLevel


class MasterSubpart(StrictModel):
    part_id: StableId
    decision_ref: StableId
    printed_label: str = Field(min_length=1)
    prompt_text: str = Field(min_length=1)
    points: float | None = Field(default=None, ge=0)


class MasterQuestion(StrictModel):
    question_id: StableId
    version_id: StableId
    printed_label: str = Field(min_length=1)
    prompt_text: str = Field(min_length=1)
    question_type: QuestionType
    points: float | None = Field(default=None, ge=0)
    options: list[ChoiceOptionDecision] = Field(default_factory=list)
    subparts: list[MasterSubpart] = Field(default_factory=list)
    knowledge_points: list[str] = Field(default_factory=list)
    solver_required: bool
    route_reason: SolverRouteReason
    reference_answers: list[AnswerEntryDecision] = Field(min_length=1)
    rubric: list[RubricCriterionDecision] = Field(min_length=1)
    solution_summary: str | None = None
    assumptions: list[str] = Field(default_factory=list)
    answer_provenance: list[AnswerProvenance] = Field(min_length=1)
    verification_results: list[VerificationResult] = Field(min_length=1)
    approval_status: ApprovalStatus
    requires_review: bool
    warnings: list[str] = Field(default_factory=list)


class MasterReviewItem(StrictModel):
    question_id: StableId
    reason: str = Field(min_length=1)


class ExamMaster(StrictModel):
    schema_version: Literal[EXAM_MASTER_SCHEMA_VERSION] = EXAM_MASTER_SCHEMA_VERSION
    case_id: StableId
    document_graph_sha256: Sha256
    answer_evidence_decision_sha256: Sha256
    question_reconstruction_decision_sha256: Sha256
    crop_decision_sha256: Sha256
    solver_input_manifest_sha256: Sha256
    exam_master_decision_sha256: Sha256
    exam_review_decision_sha256: Sha256
    stage_fingerprint: Sha256
    created_at: datetime
    evidence_provenance: DocumentDecisionProvenance
    reconstruction_provenance: DocumentDecisionProvenance
    solver_provenance: DocumentDecisionProvenance
    review_provenance: DocumentDecisionProvenance
    answer_evidence: list[AnswerEvidenceDecision] = Field(default_factory=list)
    solver_input_manifest: SolverInputManifest
    questions: list[MasterQuestion] = Field(min_length=1)
    review_items: list[MasterReviewItem] = Field(default_factory=list)
    requires_review: bool = False
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_master(self) -> ExamMaster:
        question_ids = [question.question_id for question in self.questions]
        if len(question_ids) != len(set(question_ids)):
            raise ValueError("Exam Master questions must be unique")
        manifest_ids = {
            question.question_id for question in self.solver_input_manifest.questions
        }
        if set(question_ids) != manifest_ids:
            raise ValueError("Exam Master questions must match the solver manifest")
        review_ids = [item.question_id for item in self.review_items]
        if len(review_ids) != len(set(review_ids)):
            raise ValueError("Exam Master review items must be unique")
        if self.requires_review != bool(self.review_items):
            raise ValueError("requires_review must reflect review_items")
        if any(question.requires_review for question in self.questions) != self.requires_review:
            raise ValueError("question review flags must reflect Exam Master review state")
        return self
