"""Domain errors with messages suitable for the command-line interface."""


class StudentAnalyzeError(Exception):
    """Base class for expected pipeline failures."""


class ConfigurationError(StudentAnalyzeError):
    """Configuration is missing or invalid."""


class SourceIntegrityError(StudentAnalyzeError):
    """A source asset changed or cannot be safely read."""


class CaseValidationError(StudentAnalyzeError):
    """A persisted case or artifact does not satisfy its contract."""


class InvalidTransitionError(StudentAnalyzeError):
    """A pipeline stage transition would skip or invalidate a stage."""


class AtomicCommitError(StudentAnalyzeError):
    """A validated artifact could not be committed atomically."""


class ReviewRequiredError(StudentAnalyzeError):
    """A visual decision is unresolved and must not advance the pipeline."""
