from sarasa import Config, Trainer
from sarasa.optimizers import AdamW, Muon
from sarasa.utils import setup_logger

if __name__ == "__main__":
    # optim:adam-w, optim:muon, ... to select optimizer
    config = Config.from_cli(optim_type=AdamW | Muon)

    setup_logger(config)
    trainer = Trainer(config)
    trainer.train()
