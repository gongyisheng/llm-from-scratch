import torch

from qwen3_moe.model import Qwen3MoEModel


def sample(logits, temperature=1.0, top_k=-1):
    if temperature == 0 or top_k == 1:
        next_token = torch.argmax(logits, dim=-1)
        return next_token

    # topk filter
    if top_k > 0:
        values, indices = torch.topk(logits, k=top_k)
        logits = torch.full_like(logits, -float("inf"))
        logits.scatter_(1, indices, values)

    # softmax, temperature sampling
    probs = torch.softmax(logits / temperature, dim=-1)

    # sample
    next_token = torch.multinomial(probs, num_samples=1)

    return next_token


def generate(
    model: Qwen3MoEModel,
    prompt_token_ids: list[int],
    max_new_tokens,
    temperature=1.0,
    top_k=-1,
    eos_token_id=None,
):
    device = next(model.parameters()).device
    prompt_len = len(prompt_token_ids)

    # add batch dim: [seq] -> [1, seq]
    input_ids = torch.tensor([prompt_token_ids], device=device)

    # prefill
    position_ids = torch.arange(prompt_len, device=device).unsqueeze(0)
    logits, kv_cache = model(input_ids, position_ids)
    next_token = sample(logits[:, -1, :], temperature=temperature, top_k=top_k)
    generated = [next_token.item()]

    # decode loop
    for _ in range(max_new_tokens - 1):
        offset = prompt_len + len(generated) - 1
        position_ids = torch.tensor([[offset]], device=device)
        logits, kv_cache = model(next_token.view(1, 1), position_ids, kv_cache)
        next_token = sample(logits[:, -1, :], temperature=temperature, top_k=top_k)
        generated.append(next_token.item())
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    return prompt_token_ids + generated


def generate_batch(
    model: Qwen3MoEModel,
    list_of_token_ids: list[list[int]],
    max_new_tokens,
    temperature=1.0,
    top_k=-1,
    eos_token_id=None,
    pad_token_id=0,
):
    """Generate tokens for a batch of prompts using left padding.

    Args:
        list_of_token_ids: list of token id lists, one per prompt.
        pad_token_id: token id used for left-padding shorter sequences.
    Returns:
        list of token id lists (prompt + generated) for each sequence.
    """
    device = next(model.parameters()).device
    batch_size = len(list_of_token_ids)
    dtype = next(model.parameters()).dtype

    # --- left-pad all sequences to the same length ---
    lengths = [len(ids) for ids in list_of_token_ids]
    max_len = max(lengths)

    padded = torch.full(
        (batch_size, max_len), pad_token_id, device=device, dtype=torch.long
    )
    position_ids = torch.zeros(
        (batch_size, max_len), device=device, dtype=torch.long
    )
    # True = padding position (will be masked)
    padding_mask = torch.ones(
        (batch_size, max_len), device=device, dtype=torch.bool
    )

    for i, ids in enumerate(list_of_token_ids):
        pad_len = max_len - lengths[i]
        padded[i, pad_len:] = torch.tensor(ids, device=device)
        position_ids[i, pad_len:] = torch.arange(lengths[i], device=device)
        padding_mask[i, pad_len:] = False

    # --- prefill attention mask: causal + padding ---
    # causal: (max_len, max_len), True above diagonal
    causal_mask = torch.triu(
        torch.ones(max_len, max_len, device=device, dtype=torch.bool), diagonal=1
    )
    # combined: masked if causal-future OR padding-key
    # (batch, max_len, max_len)
    combined = causal_mask[None, :, :] | padding_mask[:, None, :]
    # ensure every position can attend to itself — prevents softmax(all -inf) = NaN
    # for padding positions whose entire row would otherwise be masked
    diag = torch.arange(max_len, device=device)
    combined[:, diag, diag] = False
    # additive mask: 0 = attend, -inf = block
    attention_mask = torch.where(combined, float("-inf"), 0.0)
    attention_mask = attention_mask.unsqueeze(1).to(dtype)  # (batch, 1, q, kv)

    # --- prefill ---
    logits, kv_cache = model(padded, position_ids, attention_mask=attention_mask)
    next_tokens = sample(logits[:, -1, :], temperature=temperature, top_k=top_k)
    if next_tokens.dim() == 1:
        next_tokens = next_tokens.unsqueeze(1)

    generated = [[next_tokens[i].item()] for i in range(batch_size)]
    finished = [False] * batch_size

    if eos_token_id is not None:
        for i in range(batch_size):
            if next_tokens[i].item() == eos_token_id:
                finished[i] = True

    # --- decode loop ---
    for _ in range(max_new_tokens - 1):
        if all(finished):
            break

        # position for each sequence's new token
        step_position_ids = torch.tensor(
            [[lengths[i] + len(generated[i]) - 1] for i in range(batch_size)],
            device=device,
        )

        # extend padding mask by one non-padding column for the new token
        padding_mask = torch.cat(
            [padding_mask, torch.zeros((batch_size, 1), device=device, dtype=torch.bool)],
            dim=1,
        )

        # decode attention mask: (batch, 1, 1, kv_len)
        decode_attn_mask = torch.where(
            padding_mask, float("-inf"), 0.0
        )[:, None, None, :].to(dtype)

        logits, kv_cache = model(
            next_tokens, step_position_ids, kv_cache, decode_attn_mask
        )
        next_tokens = sample(logits[:, -1, :], temperature=temperature, top_k=top_k)
        if next_tokens.dim() == 1:
            next_tokens = next_tokens.unsqueeze(1)

        for i in range(batch_size):
            if not finished[i]:
                generated[i].append(next_tokens[i].item())
                if eos_token_id is not None and next_tokens[i].item() == eos_token_id:
                    finished[i] = True

    # return prompt + generated for each sequence
    results = []
    for i in range(batch_size):
        results.append(list_of_token_ids[i] + generated[i])
    return results
