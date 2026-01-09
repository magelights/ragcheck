"""Integration tests for verify() function."""

import pytest
from ragcheck import verify


@pytest.fixture(autouse=True)
def skip_if_no_spacy():
    """Skip tests if spaCy model not installed."""
    try:
        import spacy
        spacy.load("en_core_web_sm")
    except OSError:
        pytest.skip("spaCy model en_core_web_sm not installed")


def test_grounded_answer():
    """Answer with entities from context should score high."""
    result = verify(
        answer="Jane Doe signed the contract on March 15, 2024.",
        context=["The agreement was signed by Jane Doe on March 15, 2024."]
    )

    assert result.grounding_score is not None
    assert result.grounding_score > 0.8
    assert result.confidence in ("high", "medium")


def test_hallucinated_answer():
    """Answer with fabricated entities should score low."""
    result = verify(
        answer="John Smith signed the contract.",
        context=["The agreement was signed by Jane Doe."]
    )

    assert result.grounding_score is not None
    assert "John Smith" in result.ungrounded_entities


def test_empty_context():
    """Empty context means nothing can be grounded."""
    result = verify(
        answer="Jane Doe is the CEO.",
        context=[]
    )

    assert result.grounding_score == 0.0
