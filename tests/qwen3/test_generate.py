import gc
from pathlib import Path

import pytest
import torch

from qwen3.main import load_model, run_inference
from qwen3.generate import generate_batch

CHECKPOINTS_DIR = Path(__file__).resolve().parent.parent.parent / "checkpoints"


def checkpoint_available(model_name: str) -> bool:
    return (CHECKPOINTS_DIR / model_name / "config.json").exists()


# Cache one model at a time to avoid GPU OOM when testing multiple models
_model_cache: dict[str, tuple] = {}


def get_model(model_name: str, device: str):
    cache_key = f"{model_name}@{device}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    # Evict previous model to free GPU memory
    for key in list(_model_cache):
        model_ref = _model_cache.pop(key)[0]
        model_ref.cpu()
        del model_ref
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model_dir = CHECKPOINTS_DIR / model_name
    result = load_model(model_dir, device=device)

    _model_cache[cache_key] = result
    return result


def infer(model_name: str, device: str, prompt: str, **kwargs):
    model, tokenizer, config = get_model(model_name, device)
    return run_inference(
        model, tokenizer, config, prompt,
        max_tokens=4096, temperature=0, enable_thinking=True, **kwargs,
    )


# --- Tests ---

slow = pytest.mark.slow


@slow
def test_math(model_name, device):
    if not checkpoint_available(model_name):
        pytest.skip(f"Checkpoint {model_name} not downloaded")

    output = infer(model_name, device, "What is 2+2? Reply with just the number.")
    assert "4" in output, f"Expected '4' in output, got: {output}"


@slow
def test_knowledge(model_name, device):
    if not checkpoint_available(model_name):
        pytest.skip(f"Checkpoint {model_name} not downloaded")

    output = infer(
        model_name, device, "What is the capital of France? Reply with just the city name."
    )
    assert "Paris" in output, f"Expected 'Paris' in output, got: {output}"


@slow
def test_comparison(model_name, device):
    if not checkpoint_available(model_name):
        pytest.skip(f"Checkpoint {model_name} not downloaded")

    output = infer(model_name, device, "Which is bigger, 9.11 or 9.9? Reply with just the number.")
    assert "9.9" in output, f"Expected '9.9' in output, got: {output}"


@slow
def test_thinking_mode(model_name, device):
    if not checkpoint_available(model_name):
        pytest.skip(f"Checkpoint {model_name} not downloaded")

    output = infer(
        model_name, device,
        "What is 2+2? Reply with just the number.",
    )
    assert "<think>" in output, f"Expected '<think>' in output, got: {output}"
    assert "4" in output, f"Expected '4' in output, got: {output}"


@slow
def test_batch(model_name, device):
    """Batch generation with left padding should produce correct results.

    Note: bfloat16 numerics mean that padding masks can introduce tiny
    floating-point differences which may accumulate and cause divergent
    outputs compared to single-sequence generation.  We therefore check
    semantic correctness instead of exact token equality.
    """
    if not checkpoint_available(model_name):
        pytest.skip(f"Checkpoint {model_name} not downloaded")

    model, tokenizer, config = get_model(model_name, device)

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
