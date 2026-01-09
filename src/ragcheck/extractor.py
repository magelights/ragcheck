"""Entity extraction using spaCy."""

import spacy

from ragcheck.types import Entity


class SpacyExtractor:
    """Extract named entities using spaCy NER."""

    def __init__(self, model: str = "en_core_web_sm"):
        """Initialize with a spaCy model."""
        self.nlp = spacy.load(model)

    def extract(self, text: str) -> list[Entity]:
        """Extract entities from text."""
        doc = self.nlp(text)
        return [
            Entity(ent.text, ent.label_, ent.start_char, ent.end_char)
            for ent in doc.ents
        ]
