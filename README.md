# sarasa

A minimum LLM training framework built on pure PyTorch with simplicity and extensibility.

## Installation

```bash
uv sync
```

or

```bash
uv add git+https://github.com/moskomule/sarasa.git
```

## Usage

```bash
uv run torchrun --nproc_per_node="gpu" train.py\
[--config-file /path/to/config.yaml]\ # config file is optional
[--seed 42] [--train.steps 1000] # override config values from command line
```

### Extending with Custom Components

Extending Sarasa is as simple as defining your own configuration dataclasses with `create` methods for custom models, optimizers, data loaders, etc. 
Here's an example of using a custom optimizer:

```python
from sarasa import Trainer, Config

class CustomOptimizer(torch.optim.Optimizer):
    ...

class CustomOptimConfig:
    lr: float = ...

    def create(self,
               model: torch.nn.Module
    ) -> torch.optim.Optimizer:
        return CustomOptimizer(model.parameters(), lr=self.lr, ...)


if __name__ == "__main__":
    config = Config.from_cli(optim=CustomOptimConfig)
    trainer = Trainer(config)
    trainer.train()
```

## Acknowledgements

This project is heavily inspired by and borrows code from `torchtitan`.
