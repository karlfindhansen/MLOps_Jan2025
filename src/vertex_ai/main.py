import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch import nn
import torch

from dataset import CustomDataset  # Updated to work with GCS
from model import CustomClassifier
from train import train_model
from utils import plot_train_val_losses, compute_smoothgrad, save_model
from config import NUM_CLASSES
from omegaconf import DictConfig, OmegaConf
import hydra
from google.cloud import storage

# Initialize GCS client
client = storage.Client()

@hydra.main(version_base=None, config_path="../config", config_name="default_config")
def main(cfg: DictConfig) -> None:
    # GCS bucket and dataset path
    bucket_name = "bird-classification-data"  # Replace with your GCS bucket name
    dataset_prefix = "files/md5"  # Replace with the path to your dataset in GCS

    # Use the GCS bucket directly for training
    backbone = "resnet50"
    num_classes = NUM_CLASSES

    print(cfg)
    print(OmegaConf.to_yaml(cfg))

    hparams = cfg.experiment.hyperparameters

    torch.manual_seed(hparams["seed"])
    batch_size = hparams["batch_size"]
    l_r = hparams["lr"]
    n_epochs = hparams["n_epochs"]
    weight_decay = hparams["weight_decay"]

    print(batch_size, l_r, n_epochs, weight_decay)

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

    print(f"Train size: {train_size}, Val size: {val_size}, Test size: {test_size}")

    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = CustomClassifier(
        backbone=backbone,
        num_classes=num_classes,
        learning_rate=l_r
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=l_r, weight_decay=weight_decay)


    from hydra.core.global_hydra import GlobalHydra

    if GlobalHydra().is_initialized():
        GlobalHydra().clear()

    # Train the model
    train_losses, val_losses = train_model()

    # Plot losses and save the model
    plot_train_val_losses(train_losses, val_losses)
    save_model(model, "../model.pth")

if __name__ == "__main__":
    main()