from collections.abc import Iterable
from functools import partial
from typing import Any, Literal, cast

import torch
from datasets import Dataset as HFDataset
from datasets import IterableDataset as HFIterableDataset
from datasets import disable_progress_bars
from datasets.distributed import split_dataset_by_node
from loguru import logger
from torch.utils.data import IterableDataset

from sarasa.data.tokenizer import BaseTokenizerWrapper
from sarasa.data.utils import prepare_varlen_metadata
from sarasa.utils import IGNORE_INDEX, rank, world_size


class HFTextDataset(IterableDataset):
    """
    A wrapper around HF datasets.

    Supports three strategies for creating input-label pairs:
    1. "streaming": Concatenate tokens from the dataset and yield input-label pairs of length seq_len + 1.
    2. "streaming_pad": Similar to "streaming", but if the buffer has fewer than seq_len + 1 tokens, pad with bos_token_id until it does.
    3. "document_pack": Create a buffer of tokens and repeatedly pick the longest sequences that fit within seq_len + 1, cropping if necessary.
    4. "document_pack_pad": Similar to "document_pack", but if no sequence can fit, pad with bos_token_id instead of cropping.
    """

    def __init__(
        self,
        dataset: HFDataset | HFIterableDataset,
        tokenizer: BaseTokenizerWrapper,
        seq_len: int,
        use_varlen: bool,
        strategy: Literal["streaming", "streaming_pad", "document_pack", "document_pack_pad"],
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

        for sample in data_iter:
            sample_text = self._text_processor(sample)
            sample_tokens = self.tokenizer.encode(sample_text)
            buffer.extend(sample_tokens)

            while len(buffer) >= 1 + self.seq_len:
                output = torch.tensor(buffer[: 1 + self.seq_len], dtype=torch.int64)
                buffer = buffer[1 + self.seq_len :]
                input = output[:-1]
                label = output[1:]
                yield self.post_process_fn({"input": input}), label

    def _streaming_pad_iter(
        self,
        data_iter: Iterable[dict[str, Any]],
    ):
        for sample in data_iter:
            sample_text = self._text_processor(sample)
            sample_tokens = self.tokenizer.encode(sample_text)

            pad_size = 0
            if len(sample_tokens) > self.seq_len + 1:
                sample_tokens = sample_tokens[: self.seq_len + 1]
            else:
                pad_size = self.seq_len + 1 - len(sample_tokens)
                sample_tokens += [self.tokenizer.bos_token_id] * pad_size

            input = sample_tokens[:-1]
            label = sample_tokens[1 : self.seq_len + 1 - pad_size] + [IGNORE_INDEX] * pad_size
            input = torch.tensor(input, dtype=torch.int64)
            label = torch.tensor(label, dtype=torch.int64)
            yield self.post_process_fn({"input": input}), label

    def _document_pack_iter(
        self,
        data_iter: Iterable[dict[str, Any]],
    ):
        # repeat picking the longest sequences from the buffer,  that fit local buffer (max length seq_len + 1)
        # if there's no sequence that can fit, pick the shortest one and crop it to fit / pad bos token to fit
        # this strategy is used in nanochat

        document_buffer: list[list[int]] = []
        output: list[int] = []
        pad_size = 0
        data_iter = iter(data_iter)

        while True:
            # fill the buffer
            while (sample := next(data_iter, None)) is not None and (len(document_buffer) < self.buffer_size):
                sample_text = self._text_processor(sample)
                sample_tokens = self.tokenizer.encode(sample_text)
                document_buffer.append(sample_tokens)

            if sample is None and len(document_buffer) == 0:
                break

            document_buffer.sort(key=len, reverse=True)

            i = 0
            while i < len(document_buffer):
                seq = document_buffer[i]
                if len(output) + len(seq) <= self.seq_len + 1:
                    output.extend(seq)
                    document_buffer.pop(i)
                else:
                    i += 1

            # if no sequence can fit, pick the shortest one and crop it to fit / pad bos token to fit
            if len(output) < self.seq_len + 1:
                if self.strategy == "document_pack":
                    seq = document_buffer[-1]
                    output.extend(seq)
                    output = output[: self.seq_len + 1]
                    document_buffer.pop(-1)
                else:
                    pad_size = self.seq_len + 1 - len(output)
                    output += [self.tokenizer.bos_token_id] * pad_size

            output = output[: self.seq_len + 1]
            input = output[:-1]
            label = output[1 : self.seq_len + 1 - pad_size] + [IGNORE_INDEX] * pad_size
            input = torch.tensor(input, dtype=torch.int64)
            label = torch.tensor(label, dtype=torch.int64)
            yield self.post_process_fn({"input": input}), label

            output = []
            pad_size = 0

    def __iter__(self):
        while True:
            data_iter = cast(Iterable, iter(self.data))
            match self.strategy:
                case "streaming":
                    yield from self._streaming_iter(data_iter)
                case "streaming_pad":
                    yield from self._streaming_pad_iter(data_iter)
                case "document_pack" | "document_pack_pad":
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
