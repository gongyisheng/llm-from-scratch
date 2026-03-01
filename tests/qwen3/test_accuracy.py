"""Token-level accuracy: greedy decode must match HuggingFace transformers."""

import gc
import time
from pathlib import Path

import pytest
import torch
from tqdm import tqdm

CHECKPOINTS_DIR = Path(__file__).resolve().parent.parent.parent / "checkpoints"

PROMPTS = [
    "The capital of France is",
    "What is 4453 + 6073?",
    "def fibonacci(n):",
    "The largest planet in our solar system is",
    "Once upon a time, there was a",
    "Write a one-sentence story about a calm ninja in the ocean.",
    "Reverse the following security code: KRONNFW ->",
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

    # Step 1: HF baseline
    print(f"\n[Step 1/3] HF model ({model_name}, {device})")

    t0 = time.perf_counter()
    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(str(model_dir))
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        str(model_dir), dtype=torch.bfloat16, attn_implementation="eager",
    ).to(device)
    hf_model.requires_grad_(False)
    t_hf_load = time.perf_counter() - t0
    print(f"  Load: {t_hf_load:.2f}s")

    hf_results = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for prompt in tqdm(PROMPTS, desc="  HF inference"):
            prompt_ids = hf_tokenizer.encode(prompt)
            tokens, lps = hf_greedy_decode(hf_model, prompt_ids, MAX_NEW_TOKENS)
            hf_results.append({"prompt_ids": prompt_ids, "tokens": tokens, "logprobs": lps})
    t_hf_infer = time.perf_counter() - t0
    print(f"  Inference: {t_hf_infer:.2f}s")

    del hf_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Step 2: Scratch model
    print(f"\n[Step 2/3] Scratch model ({model_name}, {device})")

    from qwen3.main import load_model

    t0 = time.perf_counter()
    scratch_model, _, _ = load_model(model_dir, device=device)
    t_scratch_load = time.perf_counter() - t0
    print(f"  Load: {t_scratch_load:.2f}s")

    scratch_results = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for hf in tqdm(hf_results, desc="  Scratch inference"):
            tokens, lps = scratch_greedy_decode(
                scratch_model, hf["prompt_ids"], MAX_NEW_TOKENS,
            )
            scratch_results.append({"tokens": tokens, "logprobs": lps})
    t_scratch_infer = time.perf_counter() - t0
    print(f"  Inference: {t_scratch_infer:.2f}s")

    # Step 3: Compare
    print(f"\n[Step 3/3] Comparing results")

    mismatches = []
    for i, prompt in enumerate(PROMPTS):
        hf = hf_results[i]
        scratch = scratch_results[i]

        diffs = [abs(hf["logprobs"][s] - scratch["logprobs"][s]) for s in range(MAX_NEW_TOKENS)]
        string_match = hf["tokens"] == scratch["tokens"]

        print(f"\n  Prompt[{i}]: {prompt!r}")
        print(f"    Max LogProb Diff: {max(diffs):.6f}")
        print(f"    Mean LogProb Diff: {sum(diffs)/len(diffs):.6f}")
        print(f"    String Match: {'YES' if string_match else 'NO'}")
        print(f"    HF:      {hf_tokenizer.decode(hf['tokens'])!r}")
        print(f"    Scratch: {hf_tokenizer.decode(scratch['tokens'])!r}")

        if not string_match:
            mismatches.append(
                f"Token mismatch for '{prompt}':\n"
                f"  HF:      {hf_tokenizer.decode(hf['tokens'])!r}\n"
                f"  Scratch: {hf_tokenizer.decode(scratch['tokens'])!r}"
            )
        elif max(diffs) >= LOGPROB_ATOL:
            worst = max(range(MAX_NEW_TOKENS), key=lambda s: diffs[s])
            mismatches.append(
                f"Logprob diff at step {worst} for '{prompt}': "
                f"HF={hf['logprobs'][worst]:.6f}, Scratch={scratch['logprobs'][worst]:.6f}, "
                f"diff={diffs[worst]:.6f}"
            )

    # Timing summary
    print(f"\n  Timing summary:")
    print(f"    HF      — load: {t_hf_load:.2f}s, inference: {t_hf_infer:.2f}s")
    print(f"    Scratch — load: {t_scratch_load:.2f}s, inference: {t_scratch_infer:.2f}s")

    assert not mismatches, "\n".join(mismatches)
