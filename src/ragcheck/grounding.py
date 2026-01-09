"""Grounding score calculation."""

from ragcheck.types import Entity


def compute_grounding(
    answer_entities: list[Entity],
    context_entities: list[Entity],
) -> tuple[float | None, list[str], list[str]]:
    """
    Compute entity grounding score.

    Formula: EG = |answer ∩ context| / |answer|
    """
    if not answer_entities:
        return (None, [], [])

    # Build set of normalized keys from context for fast lookup
    context_keys = {ent.normalized_key() for ent in context_entities}

    grounded = []
    ungrounded = []

    for ent in answer_entities:
        if ent.normalized_key() in context_keys:
            grounded.append(ent.text)
        else:
            ungrounded.append(ent.text)

    score = len(grounded) / len(answer_entities)
    return (score, grounded, ungrounded)
