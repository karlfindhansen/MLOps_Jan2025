import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from timm import create_model

class CustomDataset(Dataset):
    def __init__(self, image_dir, model_name, num_classes):
        self.image_paths = []
        self.labels = []
        self.folders = os.listdir(image_dir)[:num_classes] 
        
        for idx, folder in enumerate(self.folders):
            folder_path = os.path.join(image_dir, folder)
            images = os.listdir(folder_path)
            for image in images:
                self.image_paths.append(os.path.join(folder_path, image))
                self.labels.append(idx)
        
        self.model_name = model_name

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(image_path).convert("RGB")
        model_config = create_model(self.model_name, pretrained=True).default_cfg
        image = image.resize(model_config['input_size'][1:], Image.BILINEAR)
        image = np.array(image) / 255.0  # Scale to [0, 1]
        mean = np.array(model_config['mean']).reshape(1, 1, 3)
        std = np.array(model_config['std']).reshape(1, 1, 3)
        image = (image - mean) / std
        image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)
        return image, label
