import pytest
from pytorch_lightning import Trainer
import torch
from src.model import CustomClassifier
from torch.utils.data import DataLoader, TensorDataset
import torchmetrics


@pytest.mark.parametrize("batch_size", [4, 8])
@pytest.mark.parametrize("backbone", ["resnet18", "resnet50"])
def test_custom_classifier_forward(batch_size: int, backbone: str) -> None:
    num_classes = 10
    image_size = (3, 224, 224)

    model = CustomClassifier(backbone=backbone, num_classes=num_classes, pretrained=False)

    dummy_input = torch.randn(batch_size, *image_size)

    output = model(dummy_input)

    assert output.shape == (batch_size, num_classes), f"Output shape mismatch for backbone {backbone}."


@pytest.mark.parametrize("batch_size", [4, 8])
def test_custom_classifier_training_step(batch_size: int) -> None:
    model = CustomClassifier(
        backbone="resnet50", num_classes=10, pretrained=False
    )

    dummy_inputs = torch.randn(batch_size, 3, 224, 224)
    dummy_targets = torch.randint(0, 10, (batch_size,))

    train_dataset = TensorDataset(dummy_inputs, dummy_targets)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=3, persistent_workers=True)

    val_dataset = TensorDataset(dummy_inputs, dummy_targets)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=3, persistent_workers=True)

    trainer = Trainer(fast_dev_run=True)

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    assert model.trainer is not None, "Trainer was not properly attached to the model."

    accuracy_metric = torchmetrics.classification.MulticlassAccuracy(num_classes=10)
    precision_metric = torchmetrics.classification.MulticlassPrecision(num_classes=10)
    recall_metric = torchmetrics.classification.MulticlassRecall(num_classes=10)

    model_outputs = model(dummy_inputs)

    accuracy_metric.update(model_outputs, dummy_targets)
    precision_metric.update(model_outputs, dummy_targets)
    recall_metric.update(model_outputs, dummy_targets)

    accuracy_result = accuracy_metric.compute()
    precision_result = precision_metric.compute()
    recall_result = recall_metric.compute()

    assert accuracy_result is not None, "Accuracy metric was not computed."
    assert precision_result is not None, "Precision metric was not computed."
    assert recall_result is not None, "Recall metric was not computed."

    outputs = model(dummy_inputs)
    assert outputs.shape == (batch_size, 10), (
        f"Model output shape mismatch: expected ({batch_size}, 10), "
        f"but got {outputs.shape}"
    )

@pytest.mark.parametrize("optimizer", ["adam", "sgd"])
def test_configure_optimizers(optimizer: str) -> None:
    backbone = "resnet50"
    num_classes = 10
    learning_rate = 1e-3

    model = CustomClassifier(backbone=backbone, num_classes=num_classes, learning_rate=learning_rate, optimizer=optimizer)

    optimizers, schedulers = model.configure_optimizers()

    if optimizer == "adam":
        assert isinstance(optimizers[0], torch.optim.Adam), "Optimizer should be Adam."
    elif optimizer == "sgd":
        assert isinstance(optimizers[0], torch.optim.SGD), "Optimizer should be SGD."

    assert isinstance(schedulers[0], torch.optim.lr_scheduler.StepLR), "Scheduler should be StepLR."