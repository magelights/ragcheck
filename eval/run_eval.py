"""
Evaluate ragcheck on FinQABench dataset.

Usage:
    uv run python eval/run_eval.py
"""

import random
from datasets import load_dataset
from ragcheck import verify


def load_finqabench():
    """Load FinQABench dataset from HuggingFace."""
    print("Loading FinQABench dataset...")
    ds = load_dataset("lighthouzai/finqabench", split="train")
    print(f"Loaded {len(ds)} examples")
    return ds


def inject_hallucination(text: str, entities: list[str]) -> str | None:
    """
    Inject a hallucination by replacing an entity with a fake one.
    Returns None if no entities to replace.
    """
    if not entities:
        return None

    # Pick a random entity to replace
    original = random.choice(entities)

    # Generate fake replacements based on entity type
    fake_replacements = {
        # If it looks like money, replace with different amount
        "$": "$999.9 billion",
        # If it looks like a percentage, replace with different percentage
        "%": "87.3%",
        # If it looks like a year, replace with different year
        "20": "2019",
        "19": "1995",
        # Default: replace with fake company name
        "default": "FakeCorp Inc."
    }

    # Find appropriate replacement
    replacement = fake_replacements["default"]
    for prefix, fake in fake_replacements.items():
        if original.startswith(prefix):
            replacement = fake
            break

    # Replace in text
    if original in text:
        return text.replace(original, replacement, 1)
    return None


def evaluate(ds, num_samples: int = 50, threshold: float = 0.8):
    """
    Run evaluation on dataset.

    For each sample:
    1. Verify the correct answer (should score high)
    2. Verify a hallucinated answer (should score low)

    Calculate precision/recall based on threshold.
    """
    print(f"\nRunning evaluation on {num_samples} samples...")
    print(f"Threshold: {threshold}")
    print("-" * 50)

    results = {
        "true_positive": 0,   # Hallucinated answer correctly flagged (score < threshold)
        "false_positive": 0,  # Correct answer incorrectly flagged (score < threshold)
        "true_negative": 0,   # Correct answer correctly passed (score >= threshold)
        "false_negative": 0,  # Hallucinated answer incorrectly passed (score >= threshold)
        "skipped": 0,         # Could not create hallucination
    }

    for i, row in enumerate(ds.select(range(min(num_samples, len(ds))))):
        context = row["Context"]
        correct_answer = row["Response"]

        # Verify correct answer
        correct_result = verify(correct_answer, [context])

        if correct_result.grounding_score is None:
            results["skipped"] += 1
            continue

        # Try to create hallucinated answer
        hallucinated_answer = inject_hallucination(
            correct_answer,
            correct_result.details.get("answer_entities", [])
        )

        if hallucinated_answer is None:
            results["skipped"] += 1
            continue

        # Verify hallucinated answer
        hallucinated_result = verify(hallucinated_answer, [context])

        if hallucinated_result.grounding_score is None:
            results["skipped"] += 1
            continue

        # Evaluate correct answer
        if correct_result.grounding_score >= threshold:
            results["true_negative"] += 1
        else:
            results["false_positive"] += 1

        # Evaluate hallucinated answer
        if hallucinated_result.grounding_score < threshold:
            results["true_positive"] += 1
        else:
            results["false_negative"] += 1

        # Progress update
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{num_samples} samples...")

    return results


def print_results(results: dict):
    """Print evaluation metrics."""
    tp = results["true_positive"]
    fp = results["false_positive"]
    tn = results["true_negative"]
    fn = results["false_negative"]
    skipped = results["skipped"]

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)

    print(f"\nConfusion Matrix:")
    print(f"  True Positives (hallucination detected):  {tp}")
    print(f"  False Positives (correct flagged):        {fp}")
    print(f"  True Negatives (correct passed):          {tn}")
    print(f"  False Negatives (hallucination missed):   {fn}")
    print(f"  Skipped (no entities):                    {skipped}")

    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

    print(f"\nMetrics:")
    print(f"  Precision: {precision:.2%}")
    print(f"  Recall:    {recall:.2%}")
    print(f"  F1 Score:  {f1:.2%}")
    print(f"  Accuracy:  {accuracy:.2%}")

    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


def main():
    # Set random seed for reproducibility
    random.seed(42)

    # Load dataset
    ds = load_finqabench()

    # Run evaluation
    results = evaluate(ds, num_samples=50, threshold=0.8)

    # Print results
    metrics = print_results(results)

    # Check if we met our targets
    print("\n" + "=" * 50)
    print("TARGET CHECK")
    print("=" * 50)
    targets = {"precision": 0.8, "recall": 0.8, "f1": 0.8}
    for metric, target in targets.items():
        actual = metrics[metric]
        status = "PASS" if actual >= target else "FAIL"
        print(f"  {metric}: {actual:.2%} (target: {target:.0%}) [{status}]")


if __name__ == "__main__":
    main()
