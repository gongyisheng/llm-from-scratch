import argparse
import sys
import time
from pathlib import Path

import torch

from qwen3_moe.config import Qwen3MoEConfig
from qwen3_moe.tokenizer import Qwen3Tokenizer
from qwen3_moe.model import Qwen3MoEModel
from qwen3_moe.weights import load_weights
from qwen3_moe.generate import generate

SUPPORTED_MODELS = {
    "Qwen3-30B-A3B": "Qwen/Qwen3-30B-A3B",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen3-MoE inference from scratch",
    )
    parser.add_argument(
        "-m",
        "--model",
        choices=list(SUPPORTED_MODELS.keys()),
        default="Qwen3-30B-A3B",
        help="model to use (default: Qwen3-30B-A3B)",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        default="Which is bigger, 9.9 or 9.11?",
        help="user prompt text (default: 'Hello, who are you?')",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=1.0,
        help="sampling temperature (default: 1.0, use 0 for greedy)",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=-1,
        help="top-k filtering, -1 to disable (default: -1)",
    )
    parser.add_argument(
        "-n",
        "--max-tokens",
        type=int,
        default=4096,
        help="max new tokens to generate (default: 4096)",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="enable thinking mode (default: off)",
    )
    parser.add_argument(
        "-d",
        "--device",
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="device to run on (default: auto)",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def ensure_checkpoint(model_name: str, model_dir: Path):
    """Check if checkpoint exists; offer to download if missing."""
    if (model_dir / "config.json").exists():
        return

    hf_repo = SUPPORTED_MODELS[model_name]
    print(f"Checkpoint not found at {model_dir}")
    answer = input(f"Download {hf_repo}? [Y/n] ").strip().lower()

    if answer in ("", "y", "yes"):
        try:
            from huggingface_hub import snapshot_download

            print(f"Downloading {hf_repo} to {model_dir} ...")
            snapshot_download(hf_repo, local_dir=str(model_dir))
            print("Download complete.")
        except ImportError:
            print("huggingface_hub is not installed.")
            print("  pip install huggingface_hub")
            print(f"  # or: bash scripts/download.sh {hf_repo}")
            sys.exit(1)
    else:
        print("To download manually:")
        print(f"  bash scripts/download.sh {hf_repo}")
        sys.exit(0)


def load_model(model_dir, device="auto"):
    """Load model, tokenizer, and config from a checkpoint directory."""
    config = Qwen3MoEConfig.from_model_dir(model_dir)
    tokenizer = Qwen3Tokenizer.from_model_dir(model_dir)

    with torch.device("meta"):
        model = Qwen3MoEModel(config)
    load_weights(model, model_dir, dtype=config.dtype)
    # Recompute RoPE buffers (they were meta tensors during init)
    for module in model.modules():
        if hasattr(module, "_build_buffers"):
            module._build_buffers()
    model.eval()

    device = resolve_device(device)
    model = model.to(device=device)

    return model, tokenizer, config


def run_inference(
    model,
    tokenizer,
    config,
    prompt,
    max_tokens=1024,
    temperature=1.0,
    top_k=-1,
    enable_thinking=False,
):
    """Run inference and return decoded output text."""
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, enable_thinking=enable_thinking
    )
    token_ids = tokenizer.encode(formatted)

    with torch.no_grad():
        output_ids = generate(
            model,
            token_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=config.eos_token_id,
        )

    return tokenizer.decode(output_ids)


def main():
    args = parse_args()

    model_dir = Path(__file__).resolve().parent.parent / "checkpoints" / args.model
    ensure_checkpoint(args.model, model_dir)

    model, tokenizer, config = load_model(model_dir, device=args.device)

    print(f"Model: {args.model}")
    print(f"Device: {next(model.parameters()).device}")
    print(f"Thinking: {'on' if args.thinking else 'off'}")
    print("Generating...\n")

    start = time.time()
    output_text = run_inference(
        model,
        tokenizer,
        config,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        enable_thinking=args.thinking,
    )
    elapsed = time.time() - start

    # decode and print
    token_count = len(tokenizer.encode(output_text))
    prompt_count = len(tokenizer.encode(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            enable_thinking=args.thinking,
        )
    ))
    new_tokens = token_count - prompt_count
    print(output_text)
    print(f"\n--- {new_tokens} tokens in {elapsed:.2f}s ({new_tokens/elapsed:.1f} tok/s) ---")


if __name__ == "__main__":
    main()
