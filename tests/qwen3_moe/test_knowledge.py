import gc
import re
import time
from pathlib import Path

import pytest
import torch

from qwen3_moe.main import load_model, run_inference
from qwen3_moe.generate import generate, generate_batch

CHECKPOINTS_DIR = Path(__file__).resolve().parent.parent.parent / "checkpoints"


def extract_answer(text: str) -> str:
    # extract assistant response after last <|im_start|>assistant, strip thinking
    m = re.search(r"<\|im_start\|>assistant\n(.*?)(?:<\|im_end\|>|$)", text, flags=re.DOTALL)
    content = m.group(1) if m else text
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()


def checkpoint_available(model_name: str) -> bool:
    return (CHECKPOINTS_DIR / model_name / "config.json").exists()


# Cache one model at a time to avoid OOM when testing multiple models
_model_cache: dict[str, tuple] = {}


def get_model(model_name: str, device: str):
    cache_key = f"{model_name}@{device}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    # Evict previous model to free memory
    for key in list(_model_cache):
        del _model_cache[key]
    gc.collect()

    model_dir = CHECKPOINTS_DIR / model_name
    t0 = time.perf_counter()
    result = load_model(model_dir, device=device)
    print(f"\n  Load Model: {time.perf_counter() - t0:.2f}s")

    _model_cache[cache_key] = result
    return result


def infer(model_name: str, device: str, prompt: str, **kwargs):
    model, tokenizer, config = get_model(model_name, device)
    t0 = time.perf_counter()
    result = run_inference(
        model, tokenizer, config, prompt,
        max_tokens=4096, temperature=0, enable_thinking=True, **kwargs,
    )
    print(f"\n  Inference: {time.perf_counter() - t0:.2f}s")
    return result


# --- Tests ---

slow = pytest.mark.slow


@slow
def test_math(model_name, device):
    if not checkpoint_available(model_name):
        pytest.skip(f"Checkpoint {model_name} not downloaded")

    prompt = "What is 2+2? Reply with just the number."
    output = infer(model_name, device, prompt)
    answer = extract_answer(output)
    print(f"\n  Prompt: {prompt!r}")
    print(f"  Output: {output!r}")
    print(f"  Answer: {answer!r}")
    assert "4" in answer, f"Expected '4' in answer, got: {answer}"


@slow
def test_knowledge(model_name, device):
    if not checkpoint_available(model_name):
        pytest.skip(f"Checkpoint {model_name} not downloaded")

    prompt = "What is the capital of France? Reply with just the city name."
    output = infer(model_name, device, prompt)
    answer = extract_answer(output)
    print(f"\n  Prompt: {prompt!r}")
    print(f"  Output: {output!r}")
    print(f"  Answer: {answer!r}")
    assert "Paris" in answer, f"Expected 'Paris' in answer, got: {answer}"


@slow
def test_comparison(model_name, device):
    if not checkpoint_available(model_name):
        pytest.skip(f"Checkpoint {model_name} not downloaded")

    prompt = "Which is bigger, 9.11 or 9.9? Reply with just the number."
    output = infer(model_name, device, prompt)
    answer = extract_answer(output)
    print(f"\n  Prompt: {prompt!r}")
    print(f"  Output: {output!r}")
    print(f"  Answer: {answer!r}")
    assert "9.9" in answer, f"Expected '9.9' in answer, got: {answer}"


@slow
def test_thinking_mode(model_name, device):
    if not checkpoint_available(model_name):
        pytest.skip(f"Checkpoint {model_name} not downloaded")

    prompt = "What is 2+2? Reply with just the number."
    output = infer(model_name, device, prompt)
    answer = extract_answer(output)
    print(f"\n  Prompt: {prompt!r}")
    print(f"  Output: {output!r}")
    print(f"  Answer: {answer!r}")
    assert "<think>" in output, f"Expected '<think>' in output, got: {output}"
    assert "4" in answer, f"Expected '4' in answer, got: {answer}"


@slow
def test_batch(model_name, device):
    """Batch generation with left padding should produce correct results.

    Note: MoE models don't guarantee exact match between batch and single-
    sequence generation because the router's top-k expert selection is
    discontinuous — tiny floating-point differences from padding masks can
    flip which experts are selected.  We therefore check semantic correctness
    instead of exact token equality.
    """
    if not checkpoint_available(model_name):
        pytest.skip(f"Checkpoint {model_name} not downloaded")

    model, tokenizer, config = get_model(model_name, device)

    prompts = [
        "What is 2+2? Reply with just the number.",
        "What is the capital of France? Reply with just the city name.",
    ]
    expected = ["4", "Paris"]

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
            max_new_tokens=4096,
            temperature=0,
            eos_token_id=config.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    # --- verify batch outputs are correct ---
    for i, prompt in enumerate(prompts):
        text = tokenizer.decode(batch_outputs[i])
        answer = extract_answer(text)
        print(f"\n  Prompt: {prompt!r}")
        print(f"  Output: {text!r}")
        print(f"  Answer: {answer!r}")
        assert expected[i] in answer, (
            f"Prompt {i}: expected '{expected[i]}' in batch answer, got: {answer}"
        )
