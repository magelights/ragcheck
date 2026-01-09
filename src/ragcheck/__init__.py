"""ragcheck - Hallucination detection for RAG systems."""

from ragcheck.extractor import SpacyExtractor
from ragcheck.grounding import compute_grounding
from ragcheck.types import VerificationResult

__all__ = ["verify", "VerificationResult"]


def verify(
    answer: str,
    context: list[str],
    model: str = "en_core_web_sm",
) -> VerificationResult:
    """
    Verify an LLM answer against retrieved context.

    Args:
        answer: The LLM-generated answer to verify
        context: List of retrieved context chunks
        model: spaCy model name (default: en_core_web_sm)

    Returns:
        VerificationResult with grounding score and entity details
    """
    # 1. Create extractor
    extractor = SpacyExtractor(model)

    # 2. Extract entities from answer
    answer_entities = extractor.extract(answer)

    # 3. Extract entities from all context chunks
    context_entities = []
    for chunk in context:
        context_entities.extend(extractor.extract(chunk))

    # 4. Compute grounding score
    score, grounded, ungrounded = compute_grounding(answer_entities, context_entities)

    # 5. Build result
    return VerificationResult(
        grounding_score=score,
        grounded_entities=grounded,
        ungrounded_entities=ungrounded,
        confidence=VerificationResult.confidence_from_score(score),
        details={
            "answer_entity_count": len(answer_entities),
            "context_entity_count": len(context_entities),
            "answer_entities": [e.text for e in answer_entities],
            "context_entities": [e.text for e in context_entities],
        },
    )
