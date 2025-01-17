from src.model import get_model
import torch
import pytest

@pytest.mark.parametrize("batch_size", [32, 64])
def test_model(batch_size: int) -> None:
    model = get_model('resnet50', 10)
    x = torch.randn(batch_size, 3, 7, 7)
    y = model(x)
    assert y.shape == (batch_size, 10)
