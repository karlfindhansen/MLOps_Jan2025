from tqdm import tqdm
import torch

def train(model, train_dataloader, val_dataloader, criterion, optimizer, device, num_epochs=5):
    train_losses = []
    val_losses = []
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
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss / len(val_dataloader):.4f}, Accuracy: {100 * correct / total:.2f}%")
    return train_losses, val_losses
