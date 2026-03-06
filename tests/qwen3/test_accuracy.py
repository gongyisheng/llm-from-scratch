"""Token-level accuracy: greedy decode must match HuggingFace transformers."""

import pytest
import torch.distributed as dist

from qwen3.main import load_model
from tests.accuracy_runner import run_accuracy_test

PROMPTS = [
    "The capital of France is",
    "What is 4453 + 6073?",
    "def fibonacci(n):",
    "The largest planet in our solar system is",
    "Once upon a time, there was a",
    "Write a one-sentence story about a calm ninja in the ocean.",
    "Reverse the following security code: KRONNFW ->",
]


@pytest.mark.slow
def test_accuracy(model_name, device):
    """Greedy generation must match HuggingFace transformers token-for-token."""
    run_accuracy_test(
        model_name, device, load_model, PROMPTS,
        mismatch_expected=dist.is_initialized(),
    )
