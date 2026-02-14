import json
from pathlib import Path

from tokenizers import Tokenizer

SPECIAL_TOKENS = [
    # special tokens excluding bos
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
]


class BaseTokenizerWrapper:
    bos_token_id: int

    def encode(self, text: str, **kwargs) -> list[int]:
        raise NotImplementedError

    def decode(self, token_ids: list[int], **kwargs) -> str:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class HFTokenizerWrapper(BaseTokenizerWrapper):
    def __init__(
        self,
        tokenizer_path: Path,
    ):
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path / "tokenizer.json"))
        with (tokenizer_path / "tokenizer_config.json").open("r") as f:
            config = json.load(f)

        bos_token = self._get_tokens_from_config(config.get("bos_token", None))
        if bos_token is None:
            raise ValueError("BOS token must be specified in the tokenizer config.")

        # check if tokenizer adds bos token automatically
        test_encoding = self.tokenizer.encode("test").ids
        self.bos_token_id = self.tokenizer.token_to_id(bos_token)
        self.need_bos = self.bos_token_id not in test_encoding

        self.tokenizer.add_special_tokens(SPECIAL_TOKENS)

    def _get_tokens_from_config(
        self,
        token: dict[str, str] | str | None,
    ) -> str | None:
        if isinstance(token, dict):
            token = token["content"]
        return token

    def encode(
        self,
        text: str,
        **kwargs,
    ) -> list[int]:
        token_ids = self.tokenizer.encode(text, **kwargs).ids

        if self.need_bos:
            token_ids = [self.bos_token_id] + token_ids

        return token_ids

    def decode(
        self,
        token_ids: list[int],
        **kwargs,
    ) -> str:
        return self.tokenizer.decode(token_ids, **kwargs)

    def __len__(self) -> int:
        return self.tokenizer.get_vocab_size()

    def render_messages(
        self,
        messages: list[dict[str, str]],
        max_length: int,
    ) -> tuple[list[int], list[int]]:
        """Render messages into ids and mask.
        We assume each `messages` has the following structure:

        [
            {"role": "user" | "assistant", "content": "..."},
            ...,
        ]

        """

        ids, mask = [self.bos_token_id], [0]

        for i, message in enumerate(messages):
            role = message["role"]
            content = message["content"]

            assert role in ["user", "assistant"], f"Unknown role: {role}"
            assert isinstance(content, str), "Message content should be a string."

            if role == "user":
                _ids = self.encode(f"<|user_start|>{content}<|user_end|>")[1:]  # remove bos
                ids.extend(_ids)
                mask.extend([0] * len(_ids))

            else:  # assistant
                _ids = self.encode(f"<|assistant_start|>{content}<|assistant_end|>")[1:]  # remove bos
                ids.extend(_ids)
                mask.extend([1] * len(_ids))

        # Truncate from the left if exceeds max_length
        ids = ids[:max_length]
        mask = mask[:max_length]

        return ids, mask
