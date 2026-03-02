import gc
import re
import time
from pathlib import Path

import torch

CHECKPOINTS_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


def checkpoint_available(model_name: str) -> bool:
    return (CHECKPOINTS_DIR / model_name / "config.json").exists()


def extract_answer(text: str) -> str:
    # extract assistant response after last <|im_start|>assistant, strip thinking
    m = re.search(r"<\|im_start\|>assistant\n(.*?)(?:<\|im_end\|>|$)", text, flags=re.DOTALL)
    content = m.group(1) if m else text
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()


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
    print(f"\n  Load Model: {time.perf_counter() - t0:.2f}s")

    _model_cache[cache_key] = result
    return result


def infer(model_name: str, device: str, prompt: str, load_model_fn, run_inference_fn, **kwargs):
    model, tokenizer, config = get_model(model_name, device, load_model_fn)
    t0 = time.perf_counter()
    result = run_inference_fn(
        model, tokenizer, config, prompt,
        max_tokens=4096, temperature=0, enable_thinking=True, **kwargs,
    )
    print(f"\n  Inference: {time.perf_counter() - t0:.2f}s")
    return result
