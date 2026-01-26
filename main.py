import os
import sys

from loguru import logger

from sarasa import Config, Trainer

if __name__ == "__main__":
    config = Config.from_cli()
    logger.remove()
    rank = int(os.environ.get("RANK", 0))

    if config.debug:
        logger_format = f"<blue>RANK={rank}</blue> | " + (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
        logger.add(
            sys.stderr,
            format=logger_format,
            backtrace=True,
            diagnose=True,
            level="DEBUG",
        )
    else:
        # log to stderr only for rank 0
        logger.add(sys.stderr, backtrace=True, diagnose=True, level="INFO" if rank == 0 else "WARNING")
    trainer = Trainer(config)
    trainer.train()
