from src.model import get_model
import torch
import pytest

@pytest.mark.parametrize("batch_size", [4, 8])
@pytest.mark.parametrize("backbone", ["resnet18", "resnet50"])
def test_custom_classifier_forward(batch_size: int, backbone: str) -> None:
    num_classes = 10
    image_size = (3, 224, 224)

    model = CustomClassifier(backbone=backbone, num_classes=num_classes, pretrained=False)

    dummy_input = torch.randn(batch_size, *image_size)

    output = model(dummy_input)

    assert output.shape == (batch_size, num_classes), f"Output shape mismatch for backbone {backbone}."

@pytest.mark.parametrize("optimizer", ["adam", "sgd"])
def test_configure_optimizers(optimizer: str) -> None:
    backbone = "resnet18"
    num_classes = 10
    learning_rate = 1e-3

    model = CustomClassifier(backbone=backbone, num_classes=num_classes, learning_rate=learning_rate, optimizer=optimizer)

    optimizers, schedulers = model.configure_optimizers()

    if optimizer == "adam":
        assert isinstance(optimizers[0], torch.optim.Adam), "Optimizer should be Adam."
    elif optimizer == "sgd":
        assert isinstance(optimizers[0], torch.optim.SGD), "Optimizer should be SGD."

    assert isinstance(schedulers[0], torch.optim.lr_scheduler.StepLR), "Scheduler should be StepLR."

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