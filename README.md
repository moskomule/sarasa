# sarasa

A minimum LLM training framework built on pure PyTorch with simplicity and extensibility.

## Installation

```bash
uv sync [--extra cpu|cu128|cu130]
```

or

```bash
uv add git+https://github.com/moskomule/sarasa.git
```

## Usage

```bash
uv run torchrun --nproc_per_node="gpu" main.py\
[--config-file /path/to/config.py]\ # config file is optional
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

### Config File Example

It's very simple. IDE autocompletion will help you.

```python
from sarasa.config import Config, Data, LRScheduler, Model, Train

# only one Config instance should be defined in each config file
config = Config(
    model=Model(num_layers=12),
    train=Train(
        local_batch_size=16,
        global_batch_size=256,
        dtype="bfloat16",
    ),
    data=Data(tokenizer_path="./tokenizer"),
    seed=12,
)
```

## Acknowledgements

This project is heavily inspired by and borrows code from `torchtitan`.
