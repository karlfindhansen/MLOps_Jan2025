import torch
import torchvision.models as models

# Load the pretrained ResNet-50 model
resnet50 = models.resnet50(pretrained=True)

# Save the model to a file
torch.save(resnet50.state_dict(), "resnet50.pth")
