"""Layer-by-layer accuracy comparison: scratch implementation vs HuggingFace eager.

Runs both models on the same input and compares intermediate tensors at each stage.
Useful for diagnosing precision mismatches.

Usage:
    uv run python scripts/compare_hf.py -a qwen3 -m Qwen3-0.6B
    uv run python scripts/compare_hf.py -a qwen3 -m Qwen3-4B -d cpu
    uv run python scripts/compare_hf.py -a qwen3_moe -m Qwen3-30B-A3B
    uv run python scripts/compare_hf.py -a qwen3 -m Qwen3-0.6B --layer 5  # layers 0-5
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CHECKPOINTS_DIR = ROOT / "checkpoints"


def parse_args():
    p = argparse.ArgumentParser(description="Layer-by-layer accuracy comparison")
    p.add_argument("-a", "--arch", required=True, choices=["qwen3", "qwen3_moe"], help="model architecture")
    p.add_argument("-m", "--model", required=True, help="model name under checkpoints/")
    p.add_argument("-d", "--device", default="auto", choices=["cuda", "cpu", "auto"])
    p.add_argument("-p", "--prompt", default="The capital of France is")
    p.add_argument("--layer", type=int, default=-1, help="compare layers 0..N, -1 = all layers (default: -1)")
    return p.parse_args()


def resolve_device(d):
    if d == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return d


def max_diff(a, b):
    return (a.float() - b.float()).abs().max().item()


def compare_tensors(name, scratch, hf):
    diff = max_diff(scratch, hf)
    status = f"PASS {diff:.6f}" if diff == 0.0 else f"DIFF {diff:.6f}"
    print(f"  {name:30s}  {status}")
    return diff


def compare_layer_detail(layer_idx, hf_model, scratch_model, hf_hidden, scratch_hidden, position_ids, seq_len):
    """Deep comparison of a single layer's internals (projections, norms, RoPE)."""
    hf_layer = hf_model.model.layers[layer_idx]
    scratch_layer = scratch_model.layers[layer_idx]

    # RMSNorm (attention)
    hf_normed = hf_layer.input_layernorm(hf_hidden)
    scratch_normed = scratch_layer.norm1(scratch_hidden)
    compare_tensors("input_layernorm", scratch_normed, hf_normed)

    # Q/K/V projections
    hf_attn = hf_layer.self_attn
    scratch_attn = scratch_layer.attn

    hf_Q = hf_attn.q_proj(hf_normed)
    hf_K = hf_attn.k_proj(hf_normed)
    hf_V = hf_attn.v_proj(hf_normed)

    scratch_Q = scratch_attn.q_proj(scratch_normed) if hasattr(scratch_attn, 'q_proj') else scratch_attn.W_q(scratch_normed)
    scratch_K = scratch_attn.k_proj(scratch_normed) if hasattr(scratch_attn, 'k_proj') else scratch_attn.W_k(scratch_normed)
    scratch_V = scratch_attn.v_proj(scratch_normed) if hasattr(scratch_attn, 'v_proj') else scratch_attn.W_v(scratch_normed)

    compare_tensors("q_proj", scratch_Q, hf_Q)
    compare_tensors("k_proj", scratch_K, hf_K)
    compare_tensors("v_proj", scratch_V, hf_V)

    # QK norm
    n_heads = scratch_attn.n_heads
    n_kv = scratch_attn.n_kv_groups
    head_dim = scratch_attn.head_dim
    batch = 1

    hf_Q_r = hf_Q.view(batch, seq_len, n_heads, head_dim).transpose(1, 2)
    hf_K_r = hf_K.view(batch, seq_len, n_kv, head_dim).transpose(1, 2)
    scratch_Q_r = scratch_Q.view(batch, seq_len, n_heads, head_dim).transpose(1, 2)
    scratch_K_r = scratch_K.view(batch, seq_len, n_kv, head_dim).transpose(1, 2)

    hf_Q_normed = hf_attn.q_norm(hf_Q_r)
    hf_K_normed = hf_attn.k_norm(hf_K_r)
    scratch_Q_normed = scratch_attn.q_norm(scratch_Q_r.reshape(-1, seq_len, head_dim)).view(batch, n_heads, seq_len, head_dim)
    scratch_K_normed = scratch_attn.k_norm(scratch_K_r.reshape(-1, seq_len, head_dim)).view(batch, n_kv, seq_len, head_dim)

    compare_tensors("q_norm", scratch_Q_normed, hf_Q_normed)
    compare_tensors("k_norm", scratch_K_normed, hf_K_normed)

    # RoPE
    rope = scratch_model.rope
    hf_rope = hf_model.model.rotary_emb
    hf_cos, hf_sin = hf_rope(hf_V, position_ids)
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    hf_Q_roped, hf_K_roped = apply_rotary_pos_emb(hf_Q_normed, hf_K_normed, hf_cos, hf_sin)

    scratch_Q_roped = rope(scratch_Q_normed, position_ids)
    scratch_K_roped = rope(scratch_K_normed, position_ids)

    compare_tensors("rope_Q", scratch_Q_roped, hf_Q_roped)
    compare_tensors("rope_K", scratch_K_roped, hf_K_roped)


def main():
    args = parse_args()
    device = resolve_device(args.device)
    model_dir = CHECKPOINTS_DIR / args.model

    if not (model_dir / "config.json").exists():
        print(f"Checkpoint not found: {model_dir}")
        sys.exit(1)

    try:
        import transformers
    except ImportError:
        print("transformers is required: pip install transformers")
        sys.exit(1)

    # -- Load HF model --
    print(f"Loading HF model from {model_dir} ...")
    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(str(model_dir))
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        str(model_dir), dtype=torch.bfloat16, attn_implementation="eager",
    ).to(device)
    hf_model.requires_grad_(False)

    # -- Load scratch model --
    print(f"Loading scratch model ({args.arch}) ...")
    if args.arch == "qwen3_moe":
        from qwen3_moe.main import load_model
    else:
        from qwen3.main import load_model
    scratch_model, _, _ = load_model(model_dir, device=device)

    # -- Prepare input --
    prompt_ids = hf_tokenizer.encode(args.prompt)
    input_ids = torch.tensor([prompt_ids], device=device)
    seq_len = len(prompt_ids)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    n_layers = len(scratch_model.layers)
    max_layer = n_layers - 1 if args.layer == -1 else min(args.layer, n_layers - 1)

    print(f"\nPrompt: {args.prompt!r}  ({seq_len} tokens)")
    print(f"Comparing layers 0..{max_layer} (total {n_layers} layers)")

    with torch.no_grad():
        # -- HF: single forward pass to get all hidden states --
        hf_out = hf_model(input_ids, output_hidden_states=True)
        # hidden_states[0] = embedding, hidden_states[i+1] = output of layer i
        hf_hidden_states = hf_out.hidden_states
        hf_logits = hf_out.logits

        # -- Scratch: layer-by-layer --
        scratch_emb = scratch_model.tok_emb(input_ids)
        rope = scratch_model.rope

        # -- Embedding --
        print("\n--- Embedding ---")
        compare_tensors("embed_tokens", scratch_emb, hf_hidden_states[0])

        # -- Layer-by-layer comparison --
        # hf_hidden_states[i+1] = output of layer i (for i < n_layers-1)
        # hf_hidden_states[n_layers] = final_norm(output of last layer)
        scratch_hidden = scratch_emb
        for i in range(max_layer + 1):
            print(f"\n--- Layer {i} ---")
            compare_tensors("input hidden_states", scratch_hidden, hf_hidden_states[i])

            compare_layer_detail(i, hf_model, scratch_model, hf_hidden_states[i], scratch_hidden, position_ids, seq_len)

            scratch_hidden, _ = scratch_model.layers[i](scratch_hidden, rope, position_ids)
            if i < n_layers - 1:
                compare_tensors("layer_output", scratch_hidden, hf_hidden_states[i + 1])
            else:
                # last entry has final RMSNorm applied — compare normed outputs
                scratch_normed = scratch_model.final_norm(scratch_hidden)
                compare_tensors("layer_output + final_norm", scratch_normed, hf_hidden_states[i + 1])

        print("\n--- Final ---")
        scratch_logits, _ = scratch_model(input_ids, position_ids)

        diff = max_diff(scratch_logits, hf_logits)
        print(f"  {'max logit diff':30s}  {diff:.6f}")

        hf_top = torch.argmax(hf_logits[0, -1, :]).item()
        scratch_top = torch.argmax(scratch_logits[0, -1, :]).item()
        match = "MATCH" if hf_top == scratch_top else "MISMATCH"
        print(f"  {'top token':30s}  HF={hf_top} Scratch={scratch_top}  {match}")

        hf_text = hf_tokenizer.decode([hf_top])
        scratch_text = hf_tokenizer.decode([scratch_top])
        print(f"  {'decoded':30s}  HF={hf_text!r} Scratch={scratch_text!r}")


if __name__ == "__main__":
    main()
