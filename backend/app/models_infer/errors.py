class PlateInferenceError(RuntimeError):
    """Base class for plate inference failures."""


class InferenceConfigurationError(PlateInferenceError):
    """Raised when the local model configuration is invalid."""


class InferenceDependencyError(PlateInferenceError):
    """Raised when an inference dependency is missing."""


class InferenceTimeoutError(PlateInferenceError):
    """Raised when inference exceeds the configured timeout."""
