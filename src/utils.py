import matplotlib.pyplot as plt
import torch
import tqdm

def plot_train_val_losses(train_losses, val_losses):
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('train_val_losses.png')


def compute_smoothgrad(images, labels, model, criterion, n_samples=10, noise_level=0.1):
    device = images.device
    smoothgrad = torch.zeros_like(images)

    for _ in tqdm(range(n_samples)):
        noisy_images = images + noise_level * torch.randn_like(images).to(device)  # Add Gaussian noise
        noisy_images.requires_grad = True

        output = model(noisy_images)
        loss = criterion(output, labels)
        loss.backward()

        smoothgrad += noisy_images.grad.abs()  # Accumulate the absolute gradients

    smoothgrad /= n_samples  # Average the gradients
    smoothgrad = smoothgrad.max(dim=1)[0]  # Take the maximum across color channels
    return smoothgrad

def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)
