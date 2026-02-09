import enum
from typing import Callable

import torch
from datasets import IterableDataset as HFIterableDataset
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
        val_size: int,
    ) -> tuple[HFIterableDataset, HFIterableDataset]:
        match self:
            case Datasets.c4:
                ds = load_dataset(
                    "allenai/c4",
                    name="en",
                    split="train",
                    streaming=True,
                    cache_dir=cache_dir,
                )
            case Datasets.fineweb_edu:
                ds = load_dataset(
                    "HuggingFaceFW/fineweb-edu",
                    name="default",
                    split="train",
                    streaming=True,
                    cache_dir=cache_dir,
                )
            case Datasets.fineweb_edu_100b:
                ds = load_dataset(
                    "HuggingFaceFW/fineweb-edu",
                    name="sample-100BT",
                    split="train",
                    streaming=True,
                    cache_dir=cache_dir,
                )
            case Datasets.fineweb_edu_dedup:
                ds = load_dataset(
                    "HuggingFaceTB/smollm-corpus",
                    "fineweb-edu-dedup",
                    split="train",
                    streaming=True,
                    cache_dir=cache_dir,
                )

        val_ds = ds.take(val_size)
        train_ds = ds.skip(val_size)
        return train_ds, val_ds


class HFTextDataset(IterableDataset):
    def __init__(
        self,
        dataset: HFIterableDataset,
        tokenizer: Callable[[str], list[int]],
        seq_len: int,
        infinite: bool = True,
    ):
        if rank() != 0:
            disable_progress_bars()

        self.data = split_dataset_by_node(dataset, rank=rank(), world_size=world_size())
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.token_buffer: list[int] = []
        self.infinite = infinite

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
                break
            else:
                # Reset offset for the next iteration
                logger.warning("Dataset is being re-looped")
                if hasattr(self.data, "set_epoch") and hasattr(self.data, "epoch"):
                    self.data.set_epoch(self.data.epoch + 1)
