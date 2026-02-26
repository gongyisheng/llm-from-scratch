import json
from datetime import datetime
from pathlib import Path

import tiktoken
from jinja2 import BaseLoader, Environment


class GPTOSSTokenizer:
    def __init__(self, encoding: tiktoken.Encoding, eos_token_id: int, pad_token_id: int, chat_template=None):
        self.encoding = encoding
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.chat_template = chat_template

    @classmethod
    def from_model_dir(cls, path: str | Path) -> "GPTOSSTokenizer":
        path = Path(path)

        # load special tokens from tokenizer.json
        with open(path / "tokenizer.json") as f:
            tok_json = json.load(f)
        base = tiktoken.get_encoding("o200k_base")
        special_tokens = {**base._special_tokens}
        for token in tok_json["added_tokens"]:
            special_tokens[token["content"]] = token["id"]
        encoding = tiktoken.Encoding(
            name="o200k_harmony",
            pat_str=base._pat_str,
            mergeable_ranks=base._mergeable_ranks,
            special_tokens=special_tokens,
        )

        # load chat template
        template_file = path / "chat_template.jinja"
        chat_template = None
        if template_file.exists():
            env = Environment(loader=BaseLoader())
            env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
            chat_template = env.from_string(template_file.read_text())

        # load eos/pad token ids from tokenizer_config.json
        with open(path / "tokenizer_config.json") as f:
            tok_cfg = json.load(f)
        eos_token_id = special_tokens[tok_cfg["eos_token"]]
        pad_token_id = special_tokens[tok_cfg["pad_token"]]

        return cls(encoding, eos_token_id, pad_token_id, chat_template=chat_template)

    def apply_chat_template(self, messages: list[dict], **kwargs) -> str:
        if self.chat_template is None:
            raise RuntimeError("No chat template loaded. Use from_model_dir() with a checkpoint that has chat_template.jinja")
        return self.chat_template.render(
            messages=messages,
            add_generation_prompt=True,
            **kwargs,
        )

    def encode(self, text: str) -> list[int]:
        return self.encoding.encode(text, allowed_special="all")

    def decode(self, token_ids: list[int]) -> str:
        return self.encoding.decode(token_ids)


if __name__ == "__main__":
    _root = Path(__file__).resolve().parent.parent
    tokenizer = GPTOSSTokenizer.from_model_dir(_root / "checkpoints/gpt-oss-20b")

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
