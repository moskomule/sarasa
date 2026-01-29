from sarasa import Config, Trainer
from sarasa.utils import setup_logger

if __name__ == "__main__":
    # config = Config.from_cli()
    from configs.example import config  # load a predefined config

    config: Config

    setup_logger(config)
    trainer = Trainer(config)
    trainer.train()
