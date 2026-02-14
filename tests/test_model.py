import typing

import pytest
import torch

from sarasa.models import ModelConfig
from sarasa.models.utils import create_varlen_metadata_prehook


@pytest.mark.parametrize("name", typing.get_type_hints(ModelConfig)["name"].__args__)
@pytest.mark.parametrize("attn_type", ["sdpa", "varlen"])
@torch.no_grad()
def test_model_shape_sdpa(name, attn_type):
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
        input = torch.randint(0, config.vocab_size, (1, config.seq_len), dtype=torch.long)
        output = model(input)
    assert output.shape == (1, config.seq_len, config.vocab_size)


def test_create_varlen_metadata_prehook():
    prehook = create_varlen_metadata_prehook(bos_token_id=0)

    class DummyModule(torch.nn.Module):
        def forward(self, x, metadata=None):
            return x, metadata

    module = DummyModule()
    module.register_forward_pre_hook(prehook, with_kwargs=True)
    input = torch.tensor([[0, 1, 1, 1, 0, 1, 1]], dtype=torch.long)
    output, metadata = module(input)
    assert metadata.cu_seq_q.tolist() == [0, 4, 7]
    assert metadata.cu_seq_k.tolist() == [0, 4, 7]
    assert metadata.max_q == 4
    assert metadata.max_k == 4
