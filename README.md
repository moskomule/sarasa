# sarasa

A minimum LLM training framework built on pure PyTorch with simplicity and extensibility.

## Installation

```bash
uv sync [--extra cpu|cu128|cu130] [--extra flash_attn]
```

or

```bash
uv add sarasa[cpu|cu128|cu130]
```

## Features

- Pure PyTorch implementation
- Flexible configuration system with command-line overrides
- Support from a single GPU to multiple GPUs (simple DDP and FSDP for now)
- Selective activation checkpointing (SAC) for memory efficiency
- Async distributed checkpoint saving

- [ ] Checkpoint loading

## Usage

It's (almost) ready to use.
First, set up tokenizer, e.g.,

```bash
mkdir tokenizer
cd tokenizer
uvx hf download --local-dir . --include "tokenizer*" "meta-llama/Llama-3.1-8B"
```

Then, the following command starts training of a GPT model on FineWeb-edu with a single or multiple GPUs.

```bash
uv run torchrun --nproc_per_node="gpu" main.py \
--config-file configs/example.py \
[--train.local-batch-size 8 ...] # override config options as needed
```

### Extending with Custom Components

Extending Sarasa is as simple as defining your own configuration dataclasses with `create` methods for custom models, optimizers, data loaders, etc. 
Here's an example of using a custom optimizer:

```python
from sarasa import Trainer, Config

class CustomOptimizer(torch.optim.Optimizer):
    ...

class CustomOptim:
    lr: float = ...

    def create(self,
               model: torch.nn.Module
    ) -> torch.optim.Optimizer:
        return CustomOptimizer(model.parameters(), lr=self.lr, ...)

class CustomOptim2:
    lr: float = ...

    def create(self,
               model: torch.nn.Module
    ) -> torch.optim.Optimizer:
        return CustomOptimizer(model.parameters(), lr=self.lr, ...)


if __name__ == "__main__":
    config = Config.from_cli(optim_type=CustomOptim | CustomOptim2)
    trainer = Trainer(config)
    trainer.train()
```

From the command line, you can specify which custom optimizer to use:

```bash
python script.py optim:custom_optim --optim.lr 0.001 ...
```

### Config File Example

It's very simple. IDE autocompletion will help you.

```python
from sarasa.config import Config, Data, LRScheduler, Model, Train, LRScheduler
from custom_optim import CustomOptim

# only one Config instance should be defined in each config file
config = Config.create(
    model=Model(num_layers=12),
    train=Train(
        local_batch_size=16,
        global_batch_size=256,
        dtype="bfloat16",
    ),
    optim=CustomOptim(lr=0.001),
    lr_scheduler=LRScheduler(
        decay_type="linear",
        warmup_steps=1000,
        total_steps=100000,
    ),
    data=Data(tokenizer_path="./tokenizer"),
    seed=12,
)
```

## Acknowledgements

This project is heavily inspired by and borrows code from `torchtitan`.
