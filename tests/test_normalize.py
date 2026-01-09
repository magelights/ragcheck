"""Tests for text normalization."""

from ragcheck.normalize import normalize_text


def test_lowercase():
    assert normalize_text("Jane Doe") == "jane doe"


def test_strip_whitespace():
    assert normalize_text("  Jane Doe  ") == "jane doe"


def test_empty_string():
    assert normalize_text("") == ""
