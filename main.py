from sarasa import Config, Trainer
from sarasa.utils import setup_logger

if __name__ == "__main__":
    config = Config.from_cli()

    setup_logger(config)
    trainer = Trainer(config)
    trainer.train()
