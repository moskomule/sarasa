import typing

import pytest
import torch

from sarasa.models import ModelConfig


@pytest.mark.parametrize("name", typing.get_type_hints(ModelConfig)["name"].__args__)
@torch.no_grad()
def test_model_shape_sdpa(name):
    config = ModelConfig(
        name=name,
        num_layers=4,
        head_dim=64,
        vocab_size=32,
        seq_len=16,
        attn_type="sdpa",
    )
    with torch.device("meta"):
        model = config.create()
        input = torch.randint(0, config.vocab_size, (1, config.seq_len), dtype=torch.long)
        output = model(input)
    assert output.shape == (1, config.seq_len, config.vocab_size)


@pytest.mark.parametrize("name", typing.get_type_hints(ModelConfig)["name"].__args__)
@torch.no_grad()
def test_model_shape_varlen(name):
    from sarasa.data.utils import prepare_varlen_metadata

    config = ModelConfig(
        name=name,
        num_layers=4,
        head_dim=64,
        vocab_size=32,
        seq_len=16,
        attn_type="varlen",
    )
    input = torch.tensor([0, 1, 1, 1, 0, 1, 1], dtype=torch.long)
    input_dict = {"input": input}
    input_dict = prepare_varlen_metadata(input_dict, bos_token_id=0)
    input_dict["input"] = input_dict["input"].unsqueeze(0)  # collate
    with torch.device("meta"):
        model = config.create()
        output = model(**input_dict)
    assert output.shape == (1, input.size(0), config.vocab_size)
