import pytest
import torch

from tests.utils import _print, checkpoint_available, extract_answer, get_model, infer


def run_single_prompt_test(model_name, device, prompt, expected, load_model_fn, run_inference_fn, model_arch="qwen3"):
    if not checkpoint_available(model_name):
        pytest.skip(f"Checkpoint {model_name} not downloaded")

    output = infer(model_name, device, prompt, load_model_fn, run_inference_fn)
    answer = extract_answer(output, model_arch=model_arch)
    _print(f"\n  Prompt: {prompt!r}")
    _print(f"  Output: {output!r}")
    _print(f"  Answer: {answer!r}")
    assert expected in answer, f"Expected {expected!r} in answer, got: {answer}"
    return output, answer


def run_thinking_test(model_name, device, prompt, expected, load_model_fn, run_inference_fn, model_arch="qwen3"):
    output, answer = run_single_prompt_test(
        model_name, device, prompt, expected, load_model_fn, run_inference_fn,
        model_arch=model_arch,
    )
    assert "<think>" in output, f"Expected '<think>' in output, got: {output}"


def run_batch_test(model_name, device, prompts, expected, load_model_fn, generate_batch_fn, model_arch="qwen3"):
    if not checkpoint_available(model_name):
        pytest.skip(f"Checkpoint {model_name} not downloaded")

    model, tokenizer, config = get_model(model_name, device, load_model_fn)

    all_token_ids = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, enable_thinking=True)
        all_token_ids.append(tokenizer.encode(formatted))

    with torch.no_grad():
        batch_outputs = generate_batch_fn(
            model,
            all_token_ids,
            max_new_tokens=1024,
            temperature=0,
            eos_token_id=config.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    for i, prompt in enumerate(prompts):
        text = tokenizer.decode(batch_outputs[i])
        answer = extract_answer(text, model_arch=model_arch)
        _print(f"\n  Prompt: {prompt!r}")
        _print(f"  Output: {text!r}")
        _print(f"  Answer: {answer!r}")
        assert expected[i] in answer, (
            f"Prompt {i}: expected {expected[i]!r} in batch answer, got: {answer}"
        )
