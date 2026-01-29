import dataclasses
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from sarasa.data.hf_datasets import Datasets, HFTextDataset
from sarasa.data.tokenizer import HFTokenizerWrapper


@dataclasses.dataclass
class DataConfig:
    dataset: Datasets = Datasets.fineweb_edu_100b
    """Dataset to use for training. Can be a predefined dataset or a custom dataset path."""

    tokenizer_path: Path | str = Path("./tokenizer")
    """Path to `tokenizer.json` and `tokenizer_config.json` files."""

    seq_len: int = 2048

    num_workers: int = 4
    pin_memory: bool = True

    def create(
        self,
        batch_size: int,
    ) -> dict[str, Any]:
        # return {"tokenizer": tokenizer, "train_loader": train_loader, "val_loader": val_loader | None}
        tokenizer = HFTokenizerWrapper(Path(self.tokenizer_path))
        ds = HFTextDataset(self.dataset, "train", tokenizer, self.seq_len)
        data_loader = DataLoader(ds, batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory)

        return {
            "tokenizer": tokenizer,
            "train_loader": data_loader,
        }
