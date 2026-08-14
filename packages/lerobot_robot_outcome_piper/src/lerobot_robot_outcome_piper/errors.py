"""Explicit PiPER plugin failures."""


class OutcomePiperError(RuntimeError):
    """Base plugin error."""


class OutcomePiperStateError(OutcomePiperError):
    """Invalid lifecycle state."""


class OutcomePiperValidationError(OutcomePiperError):
    """Invalid input, feedback, or frozen safety data."""
