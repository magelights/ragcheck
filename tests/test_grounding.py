"""Tests for grounding score calculation."""

from ragcheck.grounding import compute_grounding
from ragcheck.types import Entity


def test_all_grounded():
    """All answer entities appear in context."""
    answer = [Entity("Jane Doe", "PERSON", 0, 8)]
    context = [Entity("Jane Doe", "PERSON", 0, 8)]

    score, grounded, ungrounded = compute_grounding(answer, context)

    assert score == 1.0
    assert grounded == ["Jane Doe"]
    assert ungrounded == []


def test_none_grounded():
    """No answer entities appear in context."""
    answer = [Entity("John Smith", "PERSON", 0, 10)]
    context = [Entity("Jane Doe", "PERSON", 0, 8)]

    score, grounded, ungrounded = compute_grounding(answer, context)

    assert score == 0.0
    assert grounded == []
    assert ungrounded == ["John Smith"]


def test_partial_grounding():
    """Some entities grounded, some not."""
    answer = [
        Entity("Acme Corp", "ORG", 0, 9),
        Entity("John Smith", "PERSON", 20, 30),
    ]
    context = [Entity("Acme Corp", "ORG", 0, 9)]

    score, grounded, ungrounded = compute_grounding(answer, context)

    assert score == 0.5
    assert grounded == ["Acme Corp"]
    assert ungrounded == ["John Smith"]


def test_empty_answer():
    """No entities in answer returns None score."""
    answer = []
    context = [Entity("Jane Doe", "PERSON", 0, 8)]

    score, grounded, ungrounded = compute_grounding(answer, context)

    assert score is None
    assert grounded == []
    assert ungrounded == []


def test_type_must_match():
    """Same text but different type should not match."""
    answer = [Entity("Apple", "ORG", 0, 5)]
    context = [Entity("Apple", "PRODUCT", 0, 5)]

    score, grounded, ungrounded = compute_grounding(answer, context)

    assert score == 0.0
    assert ungrounded == ["Apple"]
