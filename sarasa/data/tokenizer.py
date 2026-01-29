import json
from pathlib import Path

from tokenizers import Tokenizer


class BaseTokenizerWrapper:
    def encode(self, *args, **kwargs) -> list[int]:
        raise NotImplementedError

    def decode(self, *args, **kwargs) -> str:
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
    ) -> list[int]:
        token_ids = self.tokenizer.encode(text).ids

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
