from dataclasses import dataclass


@dataclass
class Entity:
    """An extracted named entity."""

    text: str
    label: str  # PERSON, ORG, DATE, MONEY, etc.
    start: int
    end: int

    def normalized_key(self) -> tuple[str, str]:
        """Return (normalized_text, label) for matching."""
        from ragcheck.normalize import normalize_text
        return (normalize_text(self.text), self.label)


@dataclass
class VerificationResult:
    """Result of verifying an answer against context."""

    grounding_score: float | None  # 0-1, None if no entities
    grounded_entities: list[str]
    ungrounded_entities: list[str]
    confidence: str  # "high", "medium", "low", "insufficient", "undefined"
    details: dict

    @staticmethod
    def confidence_from_score(score: float | None) -> str:
        """Map grounding score to confidence level."""
        if score is None:
            return "undefined"
        if score >= 0.9:
            return "high"
        if score >= 0.7:
            return "medium"
        if score >= 0.5:
            return "low"
        return "insufficient"
