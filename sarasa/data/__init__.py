from __future__ import annotations

import dataclasses
import enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from datasets import IterableDataset as HFIterableDataset
from datasets import load_dataset
from torch.utils.data import DataLoader

from sarasa.data.hf_datasets import HFTextDataset
from sarasa.data.tokenizer import HFTokenizerWrapper

if TYPE_CHECKING:
    from sarasa.config import Evaluate as EvaluateConfig


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

        train_ds, val_ds = ds, None
        if val_size > 0:
            val_ds = ds.take(val_size)
            train_ds = ds.skip(val_size)
        return train_ds, val_ds


@dataclasses.dataclass
class DataConfig:
    dataset: Datasets = Datasets.fineweb_edu_100b
    """Dataset to use for training. Can be a predefined dataset or a custom dataset path."""

    tokenizer_path: Path | str = Path("./tokenizer")
    """Path to `tokenizer.json` and `tokenizer_config.json` files."""

    seq_len: int = 2048

    num_workers: int = 4

    pin_memory: bool = True

    cache_dir: str | None = None
    """Path to cache directory for datasets. If None, default cache directory is used."""

    def create(
        self,
        batch_size: int,
        val_cfg: EvaluateConfig,
        use_varlen: bool,
    ) -> dict[str, Any]:
        # return {"tokenizer": tokenizer, "train_loader": train_loader, "val_loader": val_loader | None}
        tokenizer = HFTokenizerWrapper(Path(self.tokenizer_path))
        train_ds, val_ds = self.dataset.load(cache_dir=self.cache_dir, val_size=val_cfg.val_size)
        data_loader = DataLoader(
            HFTextDataset(train_ds, tokenizer, self.seq_len, use_varlen=use_varlen),
            batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        val_loader = None
        if val_cfg.val_size > 0:
            val_loader = DataLoader(
                HFTextDataset(val_ds, tokenizer, self.seq_len, infinite=False, use_varlen=use_varlen),
                batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )

        return {
            "tokenizer": tokenizer,
            "train_loader": data_loader,
            "val_loader": val_loader,
        }
