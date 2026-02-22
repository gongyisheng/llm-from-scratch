import gc
from pathlib import Path

import pytest
import torch

from qwen3_moe.main import load_model, run_inference
from qwen3_moe.generate import generate, generate_batch

CHECKPOINTS_DIR = Path(__file__).resolve().parent.parent.parent / "checkpoints"

MODELS = ["Qwen3-30B-A3B"]

DEVICE = "cpu"


def checkpoint_available(model_name: str) -> bool:
    return (CHECKPOINTS_DIR / model_name / "config.json").exists()


# Cache one model at a time to avoid OOM when testing multiple models
_model_cache: dict[str, tuple] = {}


def get_model(model_name: str):
    if model_name in _model_cache:
        return _model_cache[model_name]

    # Evict previous model to free memory
    for key in list(_model_cache):
        del _model_cache[key]
    gc.collect()

    model_dir = CHECKPOINTS_DIR / model_name
    result = load_model(model_dir, device=DEVICE)

    _model_cache[model_name] = result
    return result


def infer(model_name: str, prompt: str, **kwargs):
    model, tokenizer, config = get_model(model_name)
    return run_inference(
        model, tokenizer, config, prompt,
        max_tokens=128, temperature=0, enable_thinking=True, **kwargs,
    )


# --- Tests ---

slow = pytest.mark.slow


@slow
@pytest.mark.parametrize("model_name", MODELS)
def test_math(model_name):
    if not checkpoint_available(model_name):
        pytest.skip(f"Checkpoint {model_name} not downloaded")

    output = infer(model_name, "What is 2+2? Reply with just the number.")
    assert "4" in output, f"Expected '4' in output, got: {output}"


@slow
@pytest.mark.parametrize("model_name", MODELS)
def test_knowledge(model_name):
    if not checkpoint_available(model_name):
        pytest.skip(f"Checkpoint {model_name} not downloaded")

    output = infer(
        model_name, "What is the capital of France? Reply with just the city name."
    )
    assert "Paris" in output, f"Expected 'Paris' in output, got: {output}"


@slow
@pytest.mark.parametrize("model_name", MODELS)
def test_thinking_mode(model_name):
    if not checkpoint_available(model_name):
        pytest.skip(f"Checkpoint {model_name} not downloaded")

    output = infer(
        model_name,
        "What is 2+2? Reply with just the number.",
    )
    assert "<think>" in output, f"Expected '<think>' in output, got: {output}"
    assert "4" in output, f"Expected '4' in output, got: {output}"


@slow
@pytest.mark.parametrize("model_name", MODELS)
def test_batch(model_name):
    """Batch generation with left padding should produce correct results.

    Note: MoE models don't guarantee exact match between batch and single-
    sequence generation because the router's top-k expert selection is
    discontinuous — tiny floating-point differences from padding masks can
    flip which experts are selected.  We therefore check semantic correctness
    instead of exact token equality.
    """
    if not checkpoint_available(model_name):
        pytest.skip(f"Checkpoint {model_name} not downloaded")

    model, tokenizer, config = get_model(model_name)

    prompts = [
        "What is 2+2? Reply with just the number.",
        "What is the capital of France? Reply with just the city name.",
    ]
    expected = ["4", "Paris"]

    max_tokens = 64

    # --- batch output ---
    all_token_ids = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, enable_thinking=True)
        all_token_ids.append(tokenizer.encode(formatted))

    with torch.no_grad():
        batch_outputs = generate_batch(
            model,
            all_token_ids,
            max_new_tokens=max_tokens,
            temperature=0,
            eos_token_id=config.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    # --- verify batch outputs are correct ---
    for i, prompt in enumerate(prompts):
        text = tokenizer.decode(batch_outputs[i])
        assert expected[i] in text, (
            f"Prompt {i}: expected '{expected[i]}' in batch output, got: {text}"
        )
