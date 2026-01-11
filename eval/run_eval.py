"""
Evaluate ragcheck on FinQABench dataset.

Usage:
    uv run python eval/run_eval.py
"""

import random
from datasets import load_dataset
from hallugen import mutate
from ragcheck import verify

STRATEGIES = ["entity_swap", "numeric_drift", "temporal_shift"]


def load_finqabench():
    """Load FinQABench dataset from HuggingFace."""
    print("Loading FinQABench dataset...")
    ds = load_dataset("lighthouzai/finqabench", split="train")
    print(f"Loaded {len(ds)} examples")
    return ds


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

        # Create hallucinated answer using hallugen
        strategy = random.choice(STRATEGIES)
        mutation_result = mutate(correct_answer, context=context, strategy=strategy)

        # Skip if no mutation was made
        if mutation_result.mutated_text == correct_answer:
            results["skipped"] += 1
            continue

        hallucinated_answer = mutation_result.mutated_text

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
