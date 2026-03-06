import gc
import re
import time
from pathlib import Path

import torch

from parallel.comm import get_rank

CHECKPOINTS_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


def _print(*args, **kwargs):
    """Print only on rank 0."""
    if get_rank() == 0:
        print(*args, **kwargs)


def checkpoint_available(model_name: str) -> bool:
    return (CHECKPOINTS_DIR / model_name / "config.json").exists()


def extract_answer(text: str, model_arch: str = "qwen3") -> str:
    if model_arch in ("qwen3", "qwen3_moe"):
        # Qwen3 format: <|im_start|>assistant\n...<|im_end|>
        m = re.search(r"<\|im_start\|>assistant\n(.*?)(?:<\|im_end\|>|$)", text, flags=re.DOTALL)
        if m:
            content = m.group(1)
            return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    elif model_arch == "gpt_oss":
        # GPT-OSS format: <|channel|>final<|message|>...<|return|>
        # Analysis (thinking) is in a separate <|channel|>analysis block, not in the final block
        m = re.search(r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)", text, flags=re.DOTALL)
        if m:
            return m.group(1).strip()

    # Fallback: return full text
    return text.strip()


# Cache one model at a time to avoid GPU OOM when testing multiple models
_model_cache: dict[str, tuple] = {}


def get_model(model_name: str, device: str, load_model_fn):
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
    t0 = time.perf_counter()
    result = load_model_fn(model_dir, device=device)
    _print(f"\n  Load Model: {time.perf_counter() - t0:.2f}s")

    _model_cache[cache_key] = result
    return result


def infer(model_name: str, device: str, prompt: str, load_model_fn, run_inference_fn, **kwargs):
    model, tokenizer, config = get_model(model_name, device, load_model_fn)
    t0 = time.perf_counter()
    result = run_inference_fn(
        model, tokenizer, config, prompt,
        max_tokens=1024, temperature=0, enable_thinking=True, **kwargs,
    )
    _print(f"\n  Inference: {time.perf_counter() - t0:.2f}s")
    return result
