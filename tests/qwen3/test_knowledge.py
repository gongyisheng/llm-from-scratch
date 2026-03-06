import pytest
import torch.distributed as dist

from qwen3.main import run_inference
from qwen3.generate import generate_batch
from tests.knowledge_runner import run_single_prompt_test, run_thinking_test, run_batch_test

if dist.is_initialized():
    from qwen3.main import load_parallel_model as load_model
else:
    from qwen3.main import load_model

slow = pytest.mark.slow


@slow
def test_math(model_name, device):
    run_single_prompt_test(
        model_name, device,
        "What is 2+2? Reply with just the number.", "4",
        load_model, run_inference,
    )


@slow
def test_knowledge(model_name, device):
    run_single_prompt_test(
        model_name, device,
        "What is the capital of France? Reply with just the city name.", "Paris",
        load_model, run_inference,
    )


@slow
def test_comparison(model_name, device):
    run_single_prompt_test(
        model_name, device,
        "Which is bigger, 9.11 or 9.9? Reply with just the number.", "9.9",
        load_model, run_inference,
    )


@slow
def test_thinking_mode(model_name, device):
    run_thinking_test(
        model_name, device,
        "What is 2+2? Reply with just the number.", "4",
        load_model, run_inference,
    )


@slow
def test_batch(model_name, device):
    """Batch generation with left padding should produce correct results.

    Note: bfloat16 numerics mean that padding masks can introduce tiny
    floating-point differences which may accumulate and cause divergent
    outputs compared to single-sequence generation.  We therefore check
    semantic correctness instead of exact token equality.
    """
    run_batch_test(
        model_name, device,
        ["What is 2+2? Reply with just the number.",
         "What is the capital of France? Reply with just the city name."],
        ["4", "Paris"],
        load_model, generate_batch,
    )
