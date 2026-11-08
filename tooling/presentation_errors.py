"""Shared failures for presentation compilation and output validation."""


class PresentationError(RuntimeError):
    """A presentation or its locked compiler cannot be verified."""
