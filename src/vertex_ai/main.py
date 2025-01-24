import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch import nn
import torch

from dataset import CustomDataset
from model import CustomClassifier
from train import train_model
from utils import plot_train_val_losses, save_model
from config import NUM_CLASSES
from omegaconf import DictConfig, OmegaConf
import hydra
from google.cloud import storage

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Initialize GCS client
client = storage.Client()

@hydra.main(version_base=None, config_path="../config", config_name="train")
def main(cfg: DictConfig) -> None:
    # GCS bucket and dataset path
    bucket_name = "bird-classification-data"  # Replace with your GCS bucket name
    dataset_prefix = "files/md5"  # Replace with the path to your dataset in GCS

    # Use the GCS bucket directly for training
    backbone = "resnet50"
    num_classes = NUM_CLASSES

    # Make sure the config is printed and can be inspected
    print(cfg)
    print(OmegaConf.to_yaml(cfg))

    # Initialize dataset
    dataset = CustomDataset(
        bucket_name=bucket_name,
        prefix=dataset_prefix,
        model_name=backbone,
        num_classes=num_classes
    )

    # Split dataset into train, validation, and test sets
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    # Call train_model function
    train_model(cfg, train_dataloader, val_dataloader, test_dataloader)

if __name__ == "__main__":
    main()
