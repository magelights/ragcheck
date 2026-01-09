# ragcheck

Hallucination detection for RAG systems. Verify LLM answers against retrieved context.

## Installation

```bash
pip install ragcheck
```

## Usage

```python
from ragcheck import verify

result = verify(
    answer="The contract was signed on March 15, 2024 by John Smith.",
    context=["The agreement dated March 15, 2024...", "Signed by Jane Doe, CEO..."]
)

print(result.grounding_score)      # 0.67
print(result.ungrounded_entities)  # ["John Smith"]
print(result.confidence)           # "medium"
```

## Eval

```bash
uv run python eval/run_eval.py
```

Runs evaluation on the [FinQABench](https://huggingface.co/datasets/lighthouzai/finqabench) dataset.

## References

Loosely based on [HalluGraph](https://arxiv.org/abs/2512.01659).

## License

MIT
