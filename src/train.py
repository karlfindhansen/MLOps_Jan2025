from tqdm import tqdm
import torch
import wandb

def train(model, train_dataloader, val_dataloader, criterion, optimizer, device, num_epochs=5):
    train_losses = []
    val_losses = []

    # Initialize wandb
    wandb.init(
        project="mlops-jan-2025",
        config={"model": model, "criterion": criterion, "epochs": num_epochs},  # Other thing to add: leraning rate, optimizer, model name
    )

    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_dataloader, desc="Training Batches"):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            # Log training loss to wandb
            wandb.log({"train_loss": loss.item()})
            # Add a plot of histogram of the gradients
            grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None], 0)
            wandb.log({"gradients": wandb.Histogram(grads)})

        train_losses.append(running_loss / len(train_dataloader))
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {running_loss / len(train_dataloader):.4f}")
        
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
        val_losses.append(val_loss / len(val_dataloader))
        
        # Calculate validation accuracy
        val_accuracy = 100 * correct / total
        val_accuracy_rounded = round(val_accuracy, 2)        


        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss / len(val_dataloader):.4f}, Accuracy: {val_accuracy_rounded:.2f}%")

        # Log validation loss and accuracy to wandb
        wandb.log({
            "val_loss": val_loss,
            "val_accuracy": val_accuracy_rounded,
            "epoch": epoch + 1
        })
    return train_losses, val_losses
