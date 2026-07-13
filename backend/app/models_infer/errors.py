class PlateInferenceError(RuntimeError):
    """Base class for plate inference failures."""


class InferenceConfigurationError(PlateInferenceError):
    """Raised when the local model configuration is invalid."""


class InferenceDependencyError(PlateInferenceError):
    """Raised when an inference dependency is missing."""
