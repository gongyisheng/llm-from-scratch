"""Shared accuracy test runner: greedy decode must match HuggingFace transformers."""

import gc
import time

import pytest
import torch
from tqdm import tqdm

from tests.utils import CHECKPOINTS_DIR, checkpoint_available


def hf_greedy_decode(model, prompt_ids, max_new_tokens):
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
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    input_ids = torch.tensor([prompt_ids], device=device)
    prompt_len = len(prompt_ids)
    position_ids = torch.arange(prompt_len, device=device).unsqueeze(0)

    causal_mask = torch.triu(
        torch.ones(prompt_len, prompt_len, device=device, dtype=torch.bool), diagonal=1
    )
    attn_mask = torch.where(causal_mask, float("-inf"), 0.0)[None, None].to(dtype)

    logits, cache = model(input_ids, position_ids, attn_mask=attn_mask)
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


def run_accuracy_test(model_name, device, load_model_fn, prompts, max_new_tokens=20, logprob_atol=0.02, mismatch_expected=False):
    """Full HF-vs-scratch comparison: load HF, load scratch, compare token-by-token."""
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
        device_map=device,
    )
    hf_model.requires_grad_(False)
    # Force loop-based expert forward (not grouped_mm) to match scratch implementation
    if hasattr(hf_model.config, "_experts_implementation"):
        hf_model.config._experts_implementation = None
    t_hf_load = time.perf_counter() - t0
    print(f"\n  Load Model: {t_hf_load:.2f}s")

    hf_results = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for prompt in tqdm(prompts, desc="  HF inference"):
            prompt_ids = hf_tokenizer.encode(prompt)
            tokens, lps = hf_greedy_decode(hf_model, prompt_ids, max_new_tokens)
            hf_results.append({"prompt_ids": prompt_ids, "tokens": tokens, "logprobs": lps})
    t_hf_infer = time.perf_counter() - t0
    print(f"\n  Inference: {t_hf_infer:.2f}s")

    del hf_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Step 2: Scratch model
    print(f"\n[Step 2/3] Scratch model ({model_name}, {device})")

    t0 = time.perf_counter()
    scratch_model, _, _ = load_model_fn(model_dir, device=device)
    t_scratch_load = time.perf_counter() - t0
    print(f"  Load: {t_scratch_load:.2f}s")

    scratch_results = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for hf in tqdm(hf_results, desc="  Scratch inference"):
            tokens, lps = scratch_greedy_decode(
                scratch_model, hf["prompt_ids"], max_new_tokens,
            )
            scratch_results.append({"tokens": tokens, "logprobs": lps})
    t_scratch_infer = time.perf_counter() - t0
    print(f"  Inference: {t_scratch_infer:.2f}s")

    # Step 3: Compare
    print("\n[Step 3/3] Comparing results")

    mismatches = []
    for i, prompt in enumerate(prompts):
        hf = hf_results[i]
        scratch = scratch_results[i]

        diffs = [abs(hf["logprobs"][s] - scratch["logprobs"][s]) for s in range(max_new_tokens)]
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
        elif max(diffs) >= logprob_atol:
            worst = max(range(max_new_tokens), key=lambda s: diffs[s])
            mismatches.append(
                f"Logprob diff at step {worst} for '{prompt}': "
                f"HF={hf['logprobs'][worst]:.6f}, Scratch={scratch['logprobs'][worst]:.6f}, "
                f"diff={diffs[worst]:.6f}"
            )

    # Timing summary
    print("\n  Timing summary:")
    print(f"    HF      — load: {t_hf_load:.2f}s, inference: {t_hf_infer:.2f}s")
    print(f"    Scratch — load: {t_scratch_load:.2f}s, inference: {t_scratch_infer:.2f}s")

    if mismatch_expected:
        if mismatches:
            print(f"\n  [{len(mismatches)} mismatch(es) found, expected due to kernel differences]")
        return

    assert not mismatches, "\n".join(mismatches)
