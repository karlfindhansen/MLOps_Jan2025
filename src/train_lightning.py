import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch import nn
import torch

from datasets import CustomDataset
from model import get_model
from config import NUM_CLASSES


class LightningClassifier(pl.LightningModule):
    def __init__(self, model_name, num_classes, learning_rate=1e-3, weight_decay=1e-4):
        super(LightningClassifier, self).__init__()
        self.model = get_model(model_name, num_classes, pretrained=False)
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        images, labels = batch
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        self.log("val_loss", loss)

    def test_step(self, batch):
        images, labels = batch
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return optimizer


def main():
    image_path = "../data"
    model_name = "resnet50"
    num_classes = NUM_CLASSES

    # Dataset and Dataloader setup
    dataset = CustomDataset(image_path, model_name, num_classes)

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Model and Trainer
    model = LightningClassifier(model_name=model_name, num_classes=num_classes)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="../checkpoints",
        filename="model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=3, mode="min")

    trainer = pl.Trainer(
        max_epochs=10,
        callbacks=[checkpoint_callback, early_stopping_callback],
        log_every_n_steps=1,
    )

    # Training
    trainer.fit(model, train_dataloader, val_dataloader)

    # Testing
    trainer.test(model, test_dataloader)

    # Save model
    trainer.save_checkpoint("../model.pth")


if __name__ == "__main__":
    main()
