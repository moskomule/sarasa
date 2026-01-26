import torch

from sarasa.models import ModelConfig


def test_nanochat_gpt():
    config = ModelConfig(name="nanochat_gpt", num_layers=4, head_dim=64, vocab_size=32, seq_len=16)
    with torch.device("meta"):
        model = config.create()
        input = torch.randint(0, config.vocab_size, (1, 16))
        output = model(input)
    assert output.shape == (1, 16, config.vocab_size)
