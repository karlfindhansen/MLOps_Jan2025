from src.model import get_model
import torch
import pytest

@pytest.mark.parametrize("batch_size", [32, 64])
def test_model(batch_size: int) -> None:
    model = get_model('resnet50', 10)
    x = torch.randn(batch_size, 3, 7, 7)
    y = model(x)
    assert y.shape == (batch_size, 10)

# Validate that the model can handle different number of classes correctly
@pytest.mark.parametrize("num_classes", [1, 5, 100])
def test_num_classes(num_classes: int) -> None:
    model = get_model('resnet50', num_classes)
    x = torch.randn(8, 3, 224, 224)
    y = model(x)
    assert y.shape == (8, num_classes)  # Output shape should match num_classes

# Verify that the model's parameters are initialized correctly, with no NaN values
def test_parameter_initialization() -> None:
    model = get_model('resnet50', 10)
    for param in model.parameters():
        assert not torch.isnan(param).any(), "Model contains NaN values in parameters"