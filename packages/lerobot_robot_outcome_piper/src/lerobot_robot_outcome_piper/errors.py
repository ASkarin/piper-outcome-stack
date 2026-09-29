"""Explicit PiPER plugin failures."""


class OutcomePiperError(RuntimeError):
    """Base plugin error."""


class OutcomePiperStateError(OutcomePiperError):
    """Invalid lifecycle state."""


class OutcomePiperValidationError(OutcomePiperError):
    """Invalid input, feedback, or frozen safety data."""


class OutcomePiperInputDisconnected(OutcomePiperStateError):
    """A positively detected loss of the selected input device."""


class OutcomePiperIntentRejected(OutcomePiperValidationError):
    """A valid operator input has no admissible target; pause without clipping."""


class OutcomePiperControlTimeout(OutcomePiperValidationError):
    """Input processing exceeded its budget; hold a healthy arm and end the session."""


class OutcomePiperCameraError(OutcomePiperStateError):
    """Camera availability or image time quality failed; healthy control may hold."""


class OutcomePiperLogError(OutcomePiperStateError):
    """Operator output cannot be delivered; hold before continuing Xbox motion."""
