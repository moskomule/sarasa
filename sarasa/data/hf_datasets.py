from collections.abc import Iterable
from functools import partial
from typing import Any, Callable, Literal

import torch
from datasets import IterableDataset as HFIterableDataset
from datasets import disable_progress_bars
from datasets.distributed import split_dataset_by_node
from loguru import logger
from torch.utils.data import IterableDataset

from sarasa.data.utils import prepare_varlen_metadata
from sarasa.utils import IGNORE_INDEX, rank, world_size


class HFTextDataset(IterableDataset):
    """
    A wrapper around HF datasets.

    Supports three strategies for creating input-label pairs:
    1. "streaming": Concatenate tokens from the dataset and yield input-label pairs of length seq_len + 1.
    2. "document_pack_crop": Create a buffer of tokens and repeatedly pick the longest sequences that fit within seq_len + 1, cropping if necessary.
    3. "document_pack_pad": Similar to "document_pack_crop", but if no sequence can fit, pad with bos_token_id instead of cropping. Useful for small datasets.
    """

    def __init__(
        self,
        dataset: HFIterableDataset,
        tokenizer: Callable[[str], list[int]],
        seq_len: int,
        use_varlen: bool,
        strategy: Literal["streaming", "document_pack_crop", "document_pack_pad"],
        infinite: bool = True,
        buffer_size: int = 1_000,
    ):
        if rank() != 0:
            disable_progress_bars()

        self.data = split_dataset_by_node(dataset, rank=rank(), world_size=world_size())
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.infinite = infinite
        self.strategy = strategy
        self.buffer_size = buffer_size
        self.post_process_fn = (
            partial(prepare_varlen_metadata, bos_token_id=tokenizer.bos_token_id) if use_varlen else lambda x: x
        )

    def _text_processor(
        self,
        sample: dict,
    ) -> str:
        # Default text processor: extract 'text' field
        return sample["text"]

    def _streaming_iter(
        self,
        data_iter: Iterable[dict[str, Any]],
    ):
        buffer: list[int] = []
        max_buffer_token_len = 1 + self.seq_len

        for sample in data_iter:
            # Use the dataset-specific text processor
            sample_text = self._text_processor(sample)
            sample_tokens = self.tokenizer.encode(sample_text)
            buffer.extend(sample_tokens)

            while len(buffer) >= max_buffer_token_len:
                x = torch.tensor(buffer[:max_buffer_token_len], dtype=torch.int32)
                # update tokens to the remaining tokens
                buffer = buffer[max_buffer_token_len:]
                input = x[:-1]
                label = x[1:]
                yield self.post_process_fn({"input": input}), label

    def _document_pack_iter(
        self,
        data_iter: Iterable[dict[str, Any]],
    ):
        # repeat picking the longest sequences from the buffer,  that fit local buffer (max length seq_len + 1)
        # if there's no sequence that can fit, pick the shortest one and crop it to fit / pad bos token to fit
        # this strategy is used in nanochat

        document_buffer: list[list[int]] = []
        output_buffer: list[int] = []
        pad_size = 0

        for sample in iter(data_iter):
            # fill the buffer
            while len(document_buffer) < self.buffer_size:
                sample_text = self._text_processor(sample)
                sample_tokens = self.tokenizer.encode(sample_text)
                document_buffer.append(sample_tokens)

            document_buffer.sort(key=len, reverse=True)

            i = 0
            while i < len(document_buffer):
                seq = document_buffer[i]
                if len(output_buffer) + len(seq) <= self.seq_len + 1:
                    output_buffer.extend(seq)
                    document_buffer.pop(i)
                else:
                    i += 1

            # if no sequence can fit, pick the shortest one and crop it to fit / pad bos token to fit
            if len(output_buffer) < self.seq_len + 1:
                if self.strategy == "bos_aligned_pad":
                    pad_size = self.seq_len + 1 - len(output_buffer)
                    output_buffer += [self.tokenizer.bos_token_id] * pad_size
                else:
                    seq = document_buffer[-1]
                    output_buffer.extend(seq)
                    output_buffer = output_buffer[: self.seq_len + 1]
                    document_buffer.pop(-1)

            output_buffer = torch.tensor(output_buffer[: self.seq_len + 1], dtype=torch.int32)

            input = output_buffer[:-1]
            label = output_buffer[1:]
            if pad_size > 0:
                label = label[:-pad_size] + [IGNORE_INDEX] * pad_size

            yield self.post_process_fn({"input": input}), label

            output_buffer = []
            pad_size = 0

    def __iter__(self):
        while True:
            data_iter = iter(self.data)
            match self.strategy:
                case "streaming":
                    yield from self._streaming_iter(data_iter)
                case "document_pack_crop" | "document_pack_pad":
                    yield from self._document_pack_iter(data_iter)
                case _:
                    raise ValueError(f"Invalid packing strategy: {self.strategy}")

            if not self.infinite:
                break
            else:
                # Reset offset for the next iteration
                logger.warning("Dataset is being re-looped")
                if hasattr(self.data, "set_epoch") and hasattr(self.data, "epoch"):
                    self.data.set_epoch(self.data.epoch + 1)
