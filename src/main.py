from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch import nn
import torch

from datasets import CustomDataset
from model import get_model
from train import train
from test import test
from utils import plot_train_val_losses, compute_smoothgrad, save_model
from config import NUM_CLASSES
from omegaconf import DictConfig, OmegaConf
import hydra
import os




@hydra.main( version_base=None, config_path="../src/config", config_name="default_config")
def main(cfg : DictConfig) -> None:
    image_path = '../data'
    model_name = "resnet50"
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

    
    dataset = CustomDataset(image_path, model_name, num_classes)
    
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = get_model(model_name, num_classes, pretrained=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=l_r, weight_decay=weight_decay)
    
    train_losses, val_losses = train(model, train_dataloader, val_dataloader, criterion, optimizer, device, num_epochs=n_epochs)
    test(model, test_dataloader, device)

    plot_train_val_losses(train_losses, val_losses)
    save_model(model, "../model.pth")

if __name__ == "__main__":
    main()
