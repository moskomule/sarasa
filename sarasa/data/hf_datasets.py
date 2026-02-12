import enum
from collections.abc import Iterable
from typing import Any, Callable, Literal

import torch
from datasets import IterableDataset as HFIterableDataset
from datasets import disable_progress_bars, load_dataset
from datasets.distributed import split_dataset_by_node
from loguru import logger
from torch.utils.data import IterableDataset

from sarasa.utils import IGNORE_INDEX, rank, world_size


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

        val_ds = None
        if val_size > 0:
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
        strategy: Literal["default", "bos_aligned_crop", "bos_aligned_pad"] = "default",
        buffer_size: int = 1_000,
    ):
        if rank() != 0:
            disable_progress_bars()

        self.data = split_dataset_by_node(dataset, rank=rank(), world_size=world_size())
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.buffer: list[int] | list[list[int]] = []
        self.infinite = infinite
        self.strategy = strategy
        self.buffer_size = buffer_size

    def _text_processor(
        self,
        sample: dict,
    ) -> str:
        # Default text processor: extract 'text' field
        return sample["text"]

    def _default_iter(
        self,
        data_iter: Iterable[dict[str, Any]],
    ):
        max_buffer_token_len = 1 + self.seq_len

        for sample in data_iter:
            # Use the dataset-specific text processor
            sample_text = self._text_processor(sample)
            sample_tokens = self.tokenizer.encode(sample_text)
            self.buffer.extend(sample_tokens)

            while len(self.buffer) >= max_buffer_token_len:
                x = torch.LongTensor(self.buffer[:max_buffer_token_len])
                # update tokens to the remaining tokens
                self.buffer = self.buffer[max_buffer_token_len:]
                input = x[:-1]
                label = x[1:]
                yield {"input": input}, label

    def _bos_aligned_iter(
        self,
        data_iter: Iterable[dict[str, Any]],
    ):
        # repeat picking the longest sequences from the buffer,  that fit local buffer (max length seq_len + 1)
        # if there's no sequence that can fit, pick the shortest one and crop it to fit / pad bos token to fit
        # this strategy is used in nanochat

        output_buffer: list[int] = []
        pad_size = 0

        for sample in iter(data_iter):
            # fill the buffer
            while len(self.buffer) < self.buffer_size:
                sample_text = self._text_processor(sample)
                sample_tokens = self.tokenizer.encode(sample_text)
                self.buffer.append(sample_tokens)

            self.buffer.sort(key=len, reverse=True)

            i = 0
            while i < len(self.buffer):
                seq = self.buffer[i]
                if len(output_buffer) + len(seq) <= self.seq_len + 1:
                    output_buffer.extend(seq)
                    self.buffer.pop(i)
                else:
                    i += 1

            # if no sequence can fit, pick the shortest one and crop it to fit / pad bos token to fit
            if len(output_buffer) < self.seq_len + 1:
                if self.strategy == "bos_aligned_pad":
                    pad_size = self.seq_len + 1 - len(output_buffer)
                    output_buffer += [self.tokenizer.bos_token_id] * pad_size
                else:
                    seq = self.buffer[i]
                    output_buffer.extend(seq)
                    output_buffer = output_buffer[: self.seq_len + 1]
                    self.buffer.pop(i)

            output_buffer = output_buffer[: self.seq_len + 1]

            input = output_buffer[:-1]
            label = output_buffer[1:]
            if pad_size > 0:
                label = label[:-pad_size] + [IGNORE_INDEX] * pad_size

            yield {"input": input}, label

            output_buffer = []
            pad_size = 0

    def __iter__(self):
        while True:
            data_iter = iter(self.data)
            match self.strategy:
                case "default":
                    yield from self._default_iter(data_iter)
                case "bos_aligned_crop" | "bos_aligned_pad":
                    yield from self._bos_aligned_iter(data_iter)

            if not self.infinite:
                break
            else:
                # Reset offset for the next iteration
                logger.warning("Dataset is being re-looped")
                if hasattr(self.data, "set_epoch") and hasattr(self.data, "epoch"):
                    self.data.set_epoch(self.data.epoch + 1)
