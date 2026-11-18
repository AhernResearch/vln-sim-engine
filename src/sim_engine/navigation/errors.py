"""Known geometric rejections. Both are raised before any actual movement."""


class RejectedCandidate(ValueError):
    """Known geometry rejection before any actual movement."""


class ClearanceError(ValueError):
    """Candidate cannot satisfy the frozen preparation profile; retain evidence."""
