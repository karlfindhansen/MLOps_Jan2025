'''
from timm import create_model
from torch import nn

def get_model(model_name, num_classes, pretrained=False):
    model = create_model(model_name, pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model
'''

from pytorch_lightning import LightningModule
from timm import create_model
import torch
from torch import nn
import torchmetrics

class CustomClassifier(LightningModule):
    def __init__(
        self,
        backbone: str,
        num_classes: int,
        learning_rate: float = 1e-3,
        optimizer: str = "adam",
        pretrained: bool = False,
    ):
        """
        A custom LightningModule classifier.

        Args:
            backbone (str): Name of the model backbone (e.g., 'resnet50').
            num_classes (int): Number of output classes.
            learning_rate (float): Learning rate for the optimizer.
            optimizer (str): Optimizer type ('adam' or 'sgd').
            pretrained (bool): Whether to use a pretrained backbone.
        """
        super().__init__()
        self.save_hyperparameters()

        # Backbone
        self.backbone = create_model(backbone, pretrained=pretrained, num_classes=num_classes)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Optimizer setup
        self.learning_rate = learning_rate
        self.optimizer_class = torch.optim.Adam if optimizer.lower() == "adam" else torch.optim.SGD

        # Metrics
        metrics = torchmetrics.MetricCollection(
            {
                "accuracy": torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes),
                "precision": torchmetrics.classification.MulticlassPrecision(num_classes=num_classes),
                "recall": torchmetrics.classification.MulticlassRecall(num_classes=num_classes),
            }
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def forward(self, x):
        """Forward pass."""
        return self.backbone(x)

    def _shared_step(self, batch):
        """Shared logic for training, validation, and testing steps."""
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds

    def training_step(self, batch, batch_idx):
        """Training step."""
        loss, preds = self._shared_step(batch)
        self.train_metrics.update(preds, batch[1])
        self.log("train_loss", loss, prog_bar=True)
        self.log_dict(self.train_metrics, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        loss, preds = self._shared_step(batch)
        self.val_metrics.update(preds, batch[1])
        self.log("val_loss", loss, prog_bar=True)

    def on_validation_epoch_end(self):
        """Log validation metrics at the end of each epoch."""
        metrics = self.val_metrics.compute()
        self.log_dict(metrics, prog_bar=True)
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        """Test step."""
        loss, preds = self._shared_step(batch)
        self.test_metrics.update(preds, batch[1])
        self.log("test_loss", loss)

    def on_test_epoch_end(self):
        """Log test metrics at the end of each epoch."""
        metrics = self.test_metrics.compute()
        self.log_dict(metrics)
        self.test_metrics.reset()

    def configure_optimizers(self):
        """Set up optimizer and learning rate scheduler."""
        optimizer = self.optimizer_class(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        return [optimizer], [scheduler]
