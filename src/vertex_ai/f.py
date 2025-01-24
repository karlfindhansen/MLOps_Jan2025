from tqdm import tqdm
import torch
from torch.profiler import record_function


def frain_model(
    model,
    train_dataloader,
    val_dataloader,
    criterion,
    optimizer,
    device,
    num_epochs=5,
    profiler=None,
):
    """
    Train and validate the model.

    Args:
        model: PyTorch model to train.
        train_dataloader: DataLoader for training data.
        val_dataloader: DataLoader for validation data.
        criterion: Loss function.
        optimizer: Optimizer for updating model weights.
        device: Device to run the training on (e.g., 'cuda' or 'cpu').
        num_epochs: Number of training epochs.
        profiler: Optional PyTorch profiler for performance analysis.

    Returns:
        train_losses: List of training losses per epoch.
        val_losses: List of validation losses per epoch.
    """
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        model.train()
        running_loss = 0.0

        # Training loop
        for images, labels in tqdm(train_dataloader, desc="Training Batches"):
            images, labels = images.to(device), labels.to(device)

            # Optionally wrap the forward pass in a profiler's record_function
            with record_function("model_forward") if profiler else torch.no_grad():
                outputs = model(images)

            with record_function("loss_computation") if profiler else torch.no_grad():
                loss = criterion(outputs, labels)

            with record_function("optimizer_step") if profiler else torch.no_grad():
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

        epoch_train_loss = running_loss / len(train_dataloader)
        train_losses.append(epoch_train_loss)
        print(f"Training Loss: {epoch_train_loss:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(val_dataloader, desc="Validation Batches"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / len(val_dataloader)
        val_losses.append(epoch_val_loss)
        accuracy = 100 * correct / total
        print(f"Validation Loss: {epoch_val_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return train_losses, val_losses
