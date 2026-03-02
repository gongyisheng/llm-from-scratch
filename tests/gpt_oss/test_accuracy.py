"""Token-level accuracy: greedy decode must match HuggingFace transformers."""

import pytest

from gpt_oss.main import load_model

from tests.accuracy_runner import run_accuracy_test

PROMPTS = [
    "The capital of France is",
    "1 + 1 =",
    "def fibonacci(n):",
    "The largest planet in our solar system is",
    "Once upon a time, there was a",
]


@pytest.mark.slow
def test_accuracy(model_name, device):
    """Greedy generation must match HuggingFace transformers token-for-token."""
    run_accuracy_test(model_name, device, load_model, PROMPTS, mismatch_expected=True)
