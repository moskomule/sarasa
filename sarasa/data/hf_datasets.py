import enum
from typing import Any, Callable

import torch
from datasets import disable_progress_bars, load_dataset
from datasets.distributed import split_dataset_by_node
from loguru import logger
from torch.utils.data import IterableDataset

from sarasa.utils import rank, world_size


class Datasets(enum.StrEnum):
    c4 = enum.auto()
    fineweb_edu = enum.auto()
    fineweb_edu_100b = enum.auto()
    fineweb_edu_dedup = enum.auto()

    def load(
        self,
        cache_dir: str | None,
    ) -> Any:
        match self:
            case Datasets.c4:
                return load_dataset(
                    "allenai/c4",
                    name="en",
                    split="train",
                    streaming=True,
                    cache_dir=cache_dir,
                )
            case Datasets.fineweb_edu:
                return load_dataset(
                    "HuggingFaceFW/fineweb-edu",
                    name="default",
                    split="train",
                    streaming=True,
                    cache_dir=cache_dir,
                )
            case Datasets.fineweb_edu_100b:
                return load_dataset(
                    "HuggingFaceFW/fineweb-edu",
                    name="sample-100BT",
                    split="train",
                    streaming=True,
                    cache_dir=cache_dir,
                )
            case Datasets.fineweb_edu_dedup:
                return load_dataset(
                    "HuggingFaceTB/smollm-corpus",
                    "fineweb-edu-dedup",
                    split="train",
                    streaming=True,
                    cache_dir=cache_dir,
                )


class HFTextDataset(IterableDataset):
    def __init__(
        self,
        dataset_name: Datasets | str,
        split: str,
        tokenizer: Callable[[str], list[int]],
        seq_len: int,
        infinite: bool = True,
        cache_dir: str | None = None,
    ):
        if rank() != 0:
            disable_progress_bars()
        self.dataset_name = dataset_name
        if dataset_name in Datasets:
            ds = Datasets(dataset_name).load(cache_dir=cache_dir)

        else:
            logger.warning(f"Unknown dataset: {dataset_name}. Trying to use `load_dataset` directly.")
            ds = load_dataset(dataset_name, split=split, streaming=True, cache_dir=cache_dir)

        self.data = split_dataset_by_node(ds, rank=rank(), world_size=world_size())
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.token_buffer: list[int] = []

    def _text_processor(
        self,
        sample: dict,
    ) -> str:
        # Default text processor: extract 'text' field
        return sample["text"]

    def __iter__(self):
        max_buffer_token_len = 1 + self.seq_len

        while True:
            for sample in iter(self.data):
                # Use the dataset-specific text processor
                sample_text = self._text_processor(sample)
                sample_tokens = self.tokenizer.encode(sample_text)
                self.token_buffer.extend(sample_tokens)

                while len(self.token_buffer) >= max_buffer_token_len:
                    x = torch.LongTensor(self.token_buffer[:max_buffer_token_len])
                    # update tokens to the remaining tokens
                    self.token_buffer = self.token_buffer[max_buffer_token_len:]
                    input = x[:-1]
                    label = x[1:]
                    yield {"input": input}, label

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break
            else:
                # Reset offset for the next iteration
                logger.warning(f"Dataset {self.dataset_name} is being re-looped")
                if hasattr(self.data, "set_epoch") and hasattr(self.data, "epoch"):
                    self.data.set_epoch(self.data.epoch + 1)
