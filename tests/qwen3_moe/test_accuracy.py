"""Token-level accuracy: greedy decode must match HuggingFace transformers."""

import gc
from pathlib import Path

import pytest
import torch

CHECKPOINTS_DIR = Path(__file__).resolve().parent.parent.parent / "checkpoints"

PROMPTS = [
    "The capital of France is",
    "1 + 1 =",
    "def fibonacci(n):",
    "The largest planet in our solar system is",
    "Once upon a time, there was a",
]
MAX_NEW_TOKENS = 20
LOGPROB_ATOL = 0.02


def checkpoint_available(model_name):
    return (CHECKPOINTS_DIR / model_name / "config.json").exists()


def hf_greedy_decode(model, prompt_ids, max_new_tokens):
    """Greedy decode with HF transformers model."""
    device = next(model.parameters()).device
    input_ids = torch.tensor([prompt_ids], device=device)

    out = model(input_ids, use_cache=True)
    last = out.logits[0, -1, :]
    cache = out.past_key_values
    nid = torch.argmax(last).item()
    tokens = [nid]
    lps = [torch.log_softmax(last.float(), dim=-1)[nid].item()]

    for _ in range(max_new_tokens - 1):
        inp = torch.tensor([[tokens[-1]]], device=device)
        out = model(inp, past_key_values=cache, use_cache=True)
        last = out.logits[0, -1, :]
        cache = out.past_key_values
        nid = torch.argmax(last).item()
        tokens.append(nid)
        lps.append(torch.log_softmax(last.float(), dim=-1)[nid].item())

    return tokens, lps


def scratch_greedy_decode(model, prompt_ids, max_new_tokens):
    """Greedy decode with scratch from-scratch model."""
    device = next(model.parameters()).device
    input_ids = torch.tensor([prompt_ids], device=device)
    position_ids = torch.arange(len(prompt_ids), device=device).unsqueeze(0)

    logits, cache = model(input_ids, position_ids)
    last = logits[0, -1, :]
    nid = torch.argmax(last).item()
    tokens = [nid]
    lps = [torch.log_softmax(last.float(), dim=-1)[nid].item()]

    for step in range(max_new_tokens - 1):
        inp = torch.tensor([[tokens[-1]]], device=device)
        pos = torch.tensor([[len(prompt_ids) + step]], device=device)
        logits, cache = model(inp, pos, cache)
        last = logits[0, -1, :]
        nid = torch.argmax(last).item()
        tokens.append(nid)
        lps.append(torch.log_softmax(last.float(), dim=-1)[nid].item())

    return tokens, lps


@pytest.mark.slow
def test_accuracy(model_name, device):
    """Greedy generation must match HuggingFace transformers token-for-token."""
    if not checkpoint_available(model_name):
        pytest.skip(f"Checkpoint {model_name} not downloaded")
    transformers = pytest.importorskip("transformers")

    model_dir = CHECKPOINTS_DIR / model_name

    # Phase 1: HF baseline (generate, then free memory)
    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(str(model_dir))
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        str(model_dir), dtype=torch.bfloat16, attn_implementation="eager",
    ).to(device)
    hf_model.requires_grad_(False)

    hf_results = []
    with torch.no_grad():
        for prompt in PROMPTS:
            prompt_ids = hf_tokenizer.encode(prompt)
            tokens, lps = hf_greedy_decode(hf_model, prompt_ids, MAX_NEW_TOKENS)
            hf_results.append({"prompt_ids": prompt_ids, "tokens": tokens, "logprobs": lps})

    del hf_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Phase 2: scratch model
    from qwen3_moe.main import load_model

    scratch_model, _, _ = load_model(model_dir, device=device)

    mismatches = []
    with torch.no_grad():
        for i, prompt in enumerate(PROMPTS):
            hf = hf_results[i]
            scratch_tokens, scratch_lps = scratch_greedy_decode(
                scratch_model, hf["prompt_ids"], MAX_NEW_TOKENS,
            )

            diffs = [abs(hf["logprobs"][s] - scratch_lps[s]) for s in range(MAX_NEW_TOKENS)]
            string_match = hf["tokens"] == scratch_tokens

            print(f"\n===Prompt[{i}]: {prompt!r}===")
            print(f"  Max LogProb Diff: {max(diffs):.6f}")
            print(f"  Mean LogProb Diff: {sum(diffs)/len(diffs):.6f}")
            print(f"  String Match: {'YES' if string_match else 'NO'}")
            print(f"  HF:      {hf_tokenizer.decode(hf['tokens'])!r}")
            print(f"  Scratch: {hf_tokenizer.decode(scratch_tokens)!r}")

            if not string_match:
                mismatches.append(
                    f"Token mismatch for '{prompt}':\n"
                    f"  HF:      {hf_tokenizer.decode(hf['tokens'])!r}\n"
                    f"  Scratch: {hf_tokenizer.decode(scratch_tokens)!r}"
                )
            elif max(diffs) >= LOGPROB_ATOL:
                worst = max(range(MAX_NEW_TOKENS), key=lambda s: diffs[s])
                mismatches.append(
                    f"Logprob diff at step {worst} for '{prompt}': "
                    f"HF={hf['logprobs'][worst]:.6f}, Scratch={scratch_lps[worst]:.6f}, "
                    f"diff={diffs[worst]:.6f}"
                )

    assert not mismatches, "\n".join(mismatches)
