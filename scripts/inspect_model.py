"""Inspect model structure: print each tensor's name, shape, and parameter count.

Reads directly from safetensors files — no model code or GPU required.

Usage:
    uv run python scripts/inspect_model.py -m Qwen3-0.6B
    uv run python scripts/inspect_model.py -m Qwen3-30B-A3B
    uv run python scripts/inspect_model.py -m Qwen3-0.6B --filter layers.0
"""

import argparse
import json
import math
import sys
from pathlib import Path

from safetensors import safe_open

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = ROOT / "checkpoints"


def parse_args():
    p = argparse.ArgumentParser(description="Inspect model structure")
    p.add_argument("-m", "--model", required=True, help="model name under checkpoints/")
    p.add_argument("--filter", default=None, help="only show tensors whose name contains this substring")
    return p.parse_args()


def fmt_size(n):
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.2f}K"
    return str(n)


def get_quant_method(model_dir):
    """Read quantization method from config.json if present."""
    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        qconfig = config.get("quantization_config", {})
        return qconfig.get("quant_method")
    return None


def effective_params(name, numel, quant_method):
    """Return the effective parameter count, adjusting for quantization packing.

    MxFP4 stores 2 FP4 values per U8 byte in _blocks tensors, so the raw
    element count is half the actual parameter count.  _scales tensors are
    quantization metadata and do not represent original model parameters.
    """
    if quant_method == "mxfp4":
        if name.endswith("_blocks"):
            return numel * 2
        if name.endswith("_scales"):
            return 0
    return numel


def main():
    args = parse_args()
    model_dir = CHECKPOINTS_DIR / args.model

    safetensor_files = sorted(model_dir.glob("*.safetensors"))
    if not safetensor_files:
        print(f"No .safetensors files found in {model_dir}")
        sys.exit(1)

    quant_method = get_quant_method(model_dir)

    print(f"Inspecting {len(safetensor_files)} safetensors file(s) in {model_dir}")
    if quant_method:
        print(f"Quantization: {quant_method}")
    print()

    # Collect all tensors across shards
    tensors = {}
    for sf_path in safetensor_files:
        with safe_open(sf_path, framework="pt") as f:
            for name in f.keys():
                tensors[name] = (f.get_slice(name).get_shape(), f.get_slice(name).get_dtype())

    # Print
    print(f"{'Name':<60s} {'Shape':<16s} {'#Params':>16s} {'Dtype'}")
    print("-" * 115)

    total_params = 0
    count = 0
    for name in sorted(tensors.keys()):
        if args.filter and args.filter not in name:
            continue
        shape, dtype = tensors[name]
        numel = math.prod(shape)
        params = effective_params(name, numel, quant_method)
        total_params += params
        count += 1
        print(f"{name:<60s} {str(list(shape)):<16s} {fmt_size(params):>16s} {dtype}")

    print("-" * 115)
    print(f"{'Total':<60s} {'':<16s} {fmt_size(total_params):>16s}")
    print(f"\n{count} tensors")


if __name__ == "__main__":
    main()
