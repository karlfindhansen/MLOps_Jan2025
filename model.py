from timm import create_model
from torch import nn

def get_model(model_name, num_classes, pretrained=False):
    model = create_model(model_name, pretrained=pretrained)
    
    if hasattr(model, "fc"):  # For ResNet-like models
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model
