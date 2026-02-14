import typing

import pytest
import torch

from sarasa.models import ModelConfig


@pytest.mark.parametrize("name", typing.get_type_hints(ModelConfig)["name"].__args__)
@pytest.mark.parametrize("attn_type", ["spda", "varlen"])
@torch.no_grad()
def test_model_shape(name, attn_type):
    config = ModelConfig(
        name=name,
        num_layers=4,
        head_dim=64,
        vocab_size=32,
        seq_len=16,
        attn_type=attn_type,
        bos_token_id=1,
    )
    with torch.device("meta"):
        model = config.create()
        input = torch.tensor([0, 1, 2, 3, 5, 8, 13, 21, 34], dtype=torch.long)[None, :]
        output = model(input)
    assert output.shape == (1, 16, config.vocab_size)
