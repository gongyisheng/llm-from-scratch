import argparse
import sys
import time
from pathlib import Path

import torch

from gpt_oss.config import GPTOSSConfig
from gpt_oss.tokenizer import GPTOSSTokenizer
from gpt_oss.model import GPTOSSModel
from gpt_oss.weights import load_weights
from gpt_oss.generate import generate

def parse_args():
    parser = argparse.ArgumentParser(
        description="GPT-OSS inference from scratch",
    )
    parser.add_argument(
        "-m",
        "--model",
        choices=["gpt-oss-20b", "gpt-oss-120b"],
        default="gpt-oss-20b",
        help="model to use (default: gpt-oss-20b)",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        default="Which is bigger, 9.9 or 9.11?",
        help="user prompt text",
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

    hf_repo = f"openai/{model_name}"
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
    config = GPTOSSConfig.from_model_dir(model_dir)
    tokenizer = GPTOSSTokenizer.from_model_dir(model_dir)

    with torch.device("meta"):
        model = GPTOSSModel(config)
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
    **kwargs,
):
    """Run inference and return decoded output text."""
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, **kwargs)
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
    )
    elapsed = time.time() - start

    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}]
    )
    prompt_count = len(tokenizer.encode(formatted_prompt))
    token_count = len(tokenizer.encode(output_text))
    new_tokens = token_count - prompt_count
    print(output_text)
    print(f"\n--- {new_tokens} tokens in {elapsed:.2f}s ({new_tokens/elapsed:.1f} tok/s) ---")


if __name__ == "__main__":
    main()
