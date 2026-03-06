"""Shared accuracy test runner: greedy decode must match HuggingFace transformers."""

import gc
import os
import sys
import time

import pytest
import torch
import torch.distributed as dist
from tqdm import tqdm

from parallel.comm import get_rank
from tests.utils import CHECKPOINTS_DIR, _print, checkpoint_available


def _nccl_available() -> bool:
    """True when distributed is initialized with the NCCL backend."""
    return dist.is_initialized() and dist.get_backend() == "nccl"


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
    rank = get_rank()

    # Step 1: HF baseline (rank 0 only — avoids OOM when ranks share a GPU)
    _print(f"\n[Step 1/3] HF model ({model_name}, {device})")

    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(str(model_dir))
    all_prompt_ids = [
        hf_tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True, enable_thinking=False, return_dict=False,
        ) for p in prompts
    ]

    hf_results = []
    if _nccl_available():
        # NCCL: all ranks load HF with TP (same sharding as scratch → exact match)
        mismatch_expected = False
        load_hf = True
        load_kwargs = dict(torch_dtype=torch.bfloat16, attn_implementation="eager", tp_plan="auto")
        load_label = "Load Model (TP)"
    elif rank == 0:
        # gloo or non-distributed: rank 0 only
        load_hf = True
        load_kwargs = dict(dtype=torch.bfloat16, attn_implementation="eager", device_map=device)
        load_label = "Load Model"
    else:
        load_hf = False

    if load_hf:
        t0 = time.perf_counter()
        # Suppress duplicate tqdm progress on non-rank-0 (tqdm writes to stderr);
        # also restore after from_pretrained (HF TP redirects to /dev/null)
        saved_stdout, saved_stderr = sys.stdout, sys.stderr
        if rank != 0:
            sys.stderr = open(os.devnull, "w")
        hf_model = transformers.AutoModelForCausalLM.from_pretrained(
            str(model_dir), **load_kwargs,
        )
        sys.stdout, sys.stderr = saved_stdout, saved_stderr
        hf_model.requires_grad_(False)
        if hasattr(hf_model.config, "_experts_implementation"):
            hf_model.config._experts_implementation = None
        t_hf_load = time.perf_counter() - t0
        _print(f"\n  {load_label}: {t_hf_load:.2f}s")

        t0 = time.perf_counter()
        with torch.no_grad():
            for i, prompt in enumerate(tqdm(prompts, desc="  HF inference", disable=rank != 0)):
                tokens, lps = hf_greedy_decode(hf_model, all_prompt_ids[i], max_new_tokens)
                hf_results.append({"prompt_ids": all_prompt_ids[i], "tokens": tokens, "logprobs": lps})
        t_hf_infer = time.perf_counter() - t0
        _print(f"\n  Inference: {t_hf_infer:.2f}s")

        del hf_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        t_hf_load = t_hf_infer = 0.0
        hf_results = [{"prompt_ids": pid} for pid in all_prompt_ids]

    # Step 2: Scratch model
    _print(f"\n[Step 2/3] Scratch model ({model_name}, {device})")

    t0 = time.perf_counter()
    scratch_model, scratch_tokenizer, _ = load_model_fn(model_dir, device=device)
    t_scratch_load = time.perf_counter() - t0
    _print(f"  Load: {t_scratch_load:.2f}s")

    scratch_prompt_ids = [
        scratch_tokenizer.encode(
            scratch_tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
            )
        ) for p in prompts
    ]

    scratch_results = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for pid in tqdm(scratch_prompt_ids, desc="  Scratch inference", disable=rank != 0):
            tokens, lps = scratch_greedy_decode(
                scratch_model, pid, max_new_tokens,
            )
            scratch_results.append({"tokens": tokens, "logprobs": lps})
    t_scratch_infer = time.perf_counter() - t0
    _print(f"  Inference: {t_scratch_infer:.2f}s")

    # Step 3: Compare (rank 0 only — it has the HF reference results)
    _print("\n[Step 3/3] Comparing results")

    if rank != 0:
        return

    mismatches = []
    for i, prompt in enumerate(prompts):
        hf = hf_results[i]
        scratch = scratch_results[i]

        diffs = [abs(hf["logprobs"][s] - scratch["logprobs"][s]) for s in range(max_new_tokens)]
        string_match = hf["tokens"] == scratch["tokens"]

        _print(f"\n  Prompt[{i}]: {prompt!r}")
        _print(f"    Max LogProb Diff: {max(diffs):.6f}")
        _print(f"    Mean LogProb Diff: {sum(diffs)/len(diffs):.6f}")
        _print(f"    String Match: {'YES' if string_match else 'NO'}")
        _print(f"    HF:      {hf_tokenizer.decode(hf['tokens'])!r}")
        _print(f"    Scratch: {hf_tokenizer.decode(scratch['tokens'])!r}")

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
    _print("\n  Timing summary:")
    _print(f"    HF      — load: {t_hf_load:.2f}s, inference: {t_hf_infer:.2f}s")
    _print(f"    Scratch — load: {t_scratch_load:.2f}s, inference: {t_scratch_infer:.2f}s")

    if mismatch_expected:
        if mismatches:
            _print(f"\n  [{len(mismatches)} mismatch(es) found, expected due to kernel differences]")
        return

    assert not mismatches, "\n".join(mismatches)
