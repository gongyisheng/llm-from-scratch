import json
from pathlib import Path

from jinja2 import Template
from tokenizers import Tokenizer


class Qwen3Tokenizer:

    def __init__(self, tokenizer: Tokenizer, chat_template: Template, pad_token_id: int):
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.pad_token_id = pad_token_id

    @classmethod
    def from_model_dir(cls, path: str | Path) -> "Qwen3Tokenizer":
        path = Path(path)
        tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        with open(path / "tokenizer_config.json") as f:
            tok_cfg = json.load(f)
        chat_template = Template(tok_cfg["chat_template"])
        pad_token = tok_cfg.get("pad_token", "<|endoftext|>")
        pad_token_id = tokenizer.token_to_id(pad_token)
        return cls(tokenizer, chat_template, pad_token_id)

    def apply_chat_template(
        self, messages: list[dict], enable_thinking: bool = False
    ) -> str:
        return self.chat_template.render(
            messages=messages,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)


if __name__ == "__main__":
    tokenizer = Qwen3Tokenizer.from_model_dir("../checkpoints/Qwen3-0.6B")

    # single turn
    messages = [{"role": "user", "content": "What is 2+2?"}]
    formatted = tokenizer.apply_chat_template(messages)
    print("=== Single Turn ===")
    print(formatted)
    print("Token IDs:", tokenizer.encode(formatted)[:20], "...")

    # multi turn
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "What is 2+2?"},
    ]
    formatted = tokenizer.apply_chat_template(messages)
    print("\n=== Multi Turn ===")
    print(formatted)
