import os
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

def main():
    image_path = '../data'
    model_name = "resnet50"
    num_classes = NUM_CLASSES
    
    dataset = CustomDataset(image_path, model_name, num_classes)
    
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    model = get_model(model_name, num_classes, pretrained=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    train_losses, val_losses = train(model, train_dataloader, val_dataloader, criterion, optimizer, device, num_epochs=10)
    test(model, test_dataloader, device)

    plot_train_val_losses(train_losses, val_losses)
    save_model(model, "../model.pth")

if __name__ == "__main__":
    main()
