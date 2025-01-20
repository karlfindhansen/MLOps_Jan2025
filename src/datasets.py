import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from timm import create_model
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningDataModule
from torchvision import transforms
from dataclasses import dataclass
from hydra_loggers import HydraRichLogger, show_image_and_target

logger = HydraRichLogger(level=os.getenv("LOG_LEVEL", "INFO"))

class CustomDataset(Dataset):
    def __init__(self, image_dir, model_name, num_classes):
        self.image_paths = []
        self.labels = []
        self.folders = os.listdir(image_dir)[:num_classes]

        for idx, folder in enumerate(self.folders):
            folder_path = os.path.join(image_dir, folder)
            images = os.listdir(folder_path)
            for image in images:
                self.image_paths.append(os.path.join(folder_path, image))
                self.labels.append(idx)

        self.model_name = model_name

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert("RGB")
        model_config = create_model(self.model_name, pretrained=False).default_cfg
        image = image.resize(model_config["input_size"][1:], Image.BILINEAR)
        image = np.array(image) / 255.0
        mean = np.array(model_config["mean"]).reshape(1, 1, 3)
        std = np.array(model_config["std"]).reshape(1, 1, 3)
        image = (image - mean) / std
        image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)
        return image, label

@dataclass
class CustomDataModule(LightningDataModule):
    """Data module for the CustomDataset."""
    image_dir: str
    model_name: str
    num_classes: int
    val_split: float = 0.15
    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = True

    def __post_init__(self):
        """Initialize the data module."""
        super().__init__()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str) -> None:
        """Set up datasets for different stages."""
        if stage in ("fit", "validate"):
            full_dataset = CustomDataset(
                image_dir=self.image_dir,
                model_name=self.model_name,
                num_classes=self.num_classes,
            )
            n_total = len(full_dataset)
            n_val = int(n_total * self.val_split)
            n_train = n_total - n_val
            self.train_dataset, self.val_dataset = random_split(
                full_dataset, [n_train, n_val]
            )

        if stage == "test":
            self.test_dataset = CustomDataset(
                image_dir=self.image_dir,
                model_name=self.model_name,
                num_classes=self.num_classes,
            )

    def train_dataloader(self) -> DataLoader:
        """Return train dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
