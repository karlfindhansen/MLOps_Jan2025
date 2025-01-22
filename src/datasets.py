import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pytorch_lightning import LightningDataModule
from timm import create_model
from dataclasses import dataclass
from fastai.vision.all import *


class CUB200_201(Dataset):
    """Custom Dataset class for CUB-200-2011 dataset."""

    def __init__(
        self,
        image_dir: Path | str,
        model_name: str,
        num_classes: int,
        transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        """Initialize dataset."""
        self.image_dir = Path('../data')
        self.model_name = model_name
        self.num_classes = num_classes
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        if download:
            self._download()

        # Prepare image paths and labels
        self.image_paths = []
        self.labels = []
        self.folders = [folder for folder in os.listdir('../data') if os.path.isdir(os.path.join('../data', folder))][:num_classes]

        for idx, folder in enumerate(self.folders):
            folder_path = os.path.join('../data', folder)
            images = os.listdir(folder_path)
            for image in images:
                image_path = os.path.join(folder_path, image)
                if os.path.isfile(image_path):  # Check if it's a file (not a sub-folder)
                    self.image_paths.append(image_path)
                    self.labels.append(idx)

        # Load model configuration
        model_config = create_model(self.model_name, pretrained=False).default_cfg
        self.input_size = model_config["input_size"][1:]
        self.mean = model_config["mean"]
        self.std = model_config["std"]

    def __len__(self) -> int:
        """Length of dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        """Get item from dataset."""
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, label

    def _download(self):
        """Download and prepare the CUB-200-2011 dataset."""
        data_path = '../data'
        if os.path.exists(data_path):
            shutil.rmtree(data_path)

        # Download and extract dataset
        default_path = untar_data(URLs.CUB_200_2011)
        shutil.move(default_path, data_path)

        # Define the DataBlock
        cub_data = DataBlock(
            blocks=(ImageBlock, CategoryBlock),
            get_items=get_image_files,
            splitter=RandomSplitter(valid_pct=0.2, seed=42),
            get_y=parent_label,
            item_tfms=Resize(460),
            batch_tfms=aug_transforms(size=224, min_scale=0.75)
        )

        dls = cub_data.dataloaders(data_path, bs=64)

        torch.save(dls.train, os.path.join(data_path, 'train_dataloader.pth'))
        torch.save(dls.valid, os.path.join(data_path, 'valid_dataloader.pth'))


@dataclass
class CUB200201DataModule(LightningDataModule):
    """Data module for the Custom CUB200_201 Dataset."""

    image_dir: str
    model_name: str
    num_classes: int
    batch_size: int = 64
    download: bool = False

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets."""
        self.dataset = CUB200_201('../data', self.model_name, self.num_classes, download=self.download)

    def train_dataloader(self) -> DataLoader:
        """Return train dataloader."""
        if not hasattr(self, 'dataset'):
            self.setup()
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        if not hasattr(self, 'dataset'):
            self.setup()
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)


def preprocess_data(image_dir: str, num_classes: int, batch_size: int = 64, download: bool = False) -> None:
    """Preprocess and prepare data for training."""
    # Setup data module
    data_module = CUB200201DataModule(
        image_dir='../data',
        model_name="resnet50",
        num_classes=num_classes,
        batch_size=batch_size,
        download=download
    )

    # Call setup method to initialize the dataset before using dataloaders
    data_module.setup()

    torch.save(data_module.train_dataloader(), os.path.join('../data', 'train_dataloader.pth'))
    torch.save(data_module.val_dataloader(), os.path.join('../data', 'valid_dataloader.pth'))


def main(image_dir: str = '../data', num_classes: int = 200, batch_size: int = 64, download: bool = False) -> None:
    """Main function to preprocess and save dataset."""
    preprocess_data(image_dir='../data', num_classes=num_classes, batch_size=batch_size, download=download)


if __name__ == '__main__':
    main(download=True)
