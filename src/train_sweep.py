import wandb
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch import nn
import torch
from datasets import CustomDataset
from model import get_model
from train import train
from test import test
from utils import plot_train_val_losses, save_model
from config import NUM_CLASSES


def train_with_wandb(config=None):
    with wandb.init(config=config):
        config = wandb.config

        # Load dataset
        image_path = "../data"
        model_name = "resnet50"
        num_classes = NUM_CLASSES
        dataset = CustomDataset(image_path, model_name, num_classes)

        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )

        train_dataloader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=config.batch_size, shuffle=False
        )

        # Initialize model, criterion, and optimizer
        model = get_model(model_name, num_classes, pretrained=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )

        # Train and validate the model
        train_losses, val_losses = train(
            model,
            train_dataloader,
            val_dataloader,
            criterion,
            optimizer,
            device,
            num_epochs=config.epochs,
        )

        # Log metrics to wandb
        for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
            wandb.log(
                {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss}
            )

        # Test the model
        test_accuracy = test(model, test_dataloader, device)
        wandb.log({"test_accuracy": test_accuracy})

        # Save the model with hyperparameters in the filename
        model_filename = f"../model_lr{config.lr}_bs{config.batch_size}_wd{config.weight_decay}_ep{config.epochs}.pth"
        save_model(model, model_filename)
        # Log the model file name to wandb
        wandb.log({"model_filename": model_filename})


if __name__ == "__main__":
    sweep_config = {
        "method": "bayes",
        "metric": {"name": "val_loss", "goal": "minimize"},
        "parameters": {
            "lr": {"distribution": "log_uniform", "min": 1e-5, "max": 1e-1},
            "batch_size": {"values": [16, 32, 64, 128]},
            "epochs": {"values": [5, 10, 20]},
            "weight_decay": {"distribution": "uniform", "min": 0.00001, "max": 0.001},
        },
    }

    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project="hyperparameter-sweep")

    # Run sweep
    wandb.agent(sweep_id, train_with_wandb)
