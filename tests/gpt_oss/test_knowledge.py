import pytest

from gpt_oss.main import load_model, run_inference
from gpt_oss.generate import generate_batch

from tests.knowledge_runner import run_single_prompt_test, run_batch_test

MODEL_ARCH = "gpt_oss"

slow = pytest.mark.slow


@slow
def test_math(model_name, device):
    run_single_prompt_test(
        model_name, device,
        "What is 2+2? Reply with just the number.", "4",
        load_model, run_inference, model_arch=MODEL_ARCH,
    )


@slow
def test_knowledge(model_name, device):
    run_single_prompt_test(
        model_name, device,
        "What is the capital of France? Reply with just the city name.", "Paris",
        load_model, run_inference, model_arch=MODEL_ARCH,
    )


@slow
def test_comparison(model_name, device):
    run_single_prompt_test(
        model_name, device,
        "Which is bigger, 9.11 or 9.9? Reply with just the number.", "9.9",
        load_model, run_inference, model_arch=MODEL_ARCH,
    )


@slow
def test_batch(model_name, device):
    """Batch generation with left padding should produce correct results.

    Note: MoE models don't guarantee exact match between batch and single-
    sequence generation because the router's top-k expert selection is
    discontinuous — tiny floating-point differences from padding masks can
    flip which experts are selected.  We therefore check semantic correctness
    instead of exact token equality.
    """
    run_batch_test(
        model_name, device,
        ["What is 2+2? Reply with just the number.",
         "What is the capital of France? Reply with just the city name."],
        ["4", "Paris"],
        load_model, generate_batch, model_arch=MODEL_ARCH,
    )
