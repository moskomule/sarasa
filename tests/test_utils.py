import torch

from sarasa.utils import set_dtype


def test_set_dtype() -> None:
    torch.set_default_dtype(torch.float64)
    a = torch.randn(1)
    assert a.dtype == torch.float64

    with set_dtype(torch.float32):
        a = torch.randn(1)
        assert a.dtype == torch.float32

    a = torch.randn(1)
    assert a.dtype == torch.float64
