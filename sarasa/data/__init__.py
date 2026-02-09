from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, Any

from torch.utils.data import DataLoader

from sarasa.data.hf_datasets import Datasets, HFTextDataset
from sarasa.data.tokenizer import HFTokenizerWrapper

if TYPE_CHECKING:
    from sarasa.config import Evaluate as EvaluateConfig


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

    def create(
        self,
        batch_size: int,
        val_cfg: EvaluateConfig,
    ) -> dict[str, Any]:
        # return {"tokenizer": tokenizer, "train_loader": train_loader, "val_loader": val_loader | None}
        tokenizer = HFTokenizerWrapper(Path(self.tokenizer_path))
        train_ds, val_ds = self.dataset.load(cache_dir=self.cache_dir, val_size=val_cfg.val_size)
        data_loader = DataLoader(
            HFTextDataset(train_ds, tokenizer, self.seq_len),
            batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        val_loader = None
        if val_cfg.val_size > 0:
            val_loader = DataLoader(
                HFTextDataset(val_ds, tokenizer, self.seq_len, infinite=False),
                batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )

        return {
            "tokenizer": tokenizer,
            "train_loader": data_loader,
            "val_loader": val_loader,
        }
