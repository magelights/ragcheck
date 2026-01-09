"""Text normalization for entity matching."""


def normalize_text(text: str) -> str:
    """
    Normalize text for entity comparison.

    TODO: Implement normalization:
    - Lowercase
    - Strip whitespace
    - Normalize unicode (optional)

    Example:
        normalize_text("  Jane Doe  ") -> "jane doe"
    """
    return text.lower().strip()
