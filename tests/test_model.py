import typing

import pytest
import torch

from sarasa.models import ModelConfig


@pytest.mark.parametrize("name", typing.get_type_hints(ModelConfig)["name"].__args__)
@torch.no_grad()
def test_model_shape(name):
    config = ModelConfig(name=name, num_layers=4, head_dim=64, vocab_size=32, seq_len=16)
    with torch.device("meta"):
        model = config.create()
        input = torch.randint(0, config.vocab_size, (1, 16))
        output = model(input)
    assert output.shape == (1, 16, config.vocab_size)
