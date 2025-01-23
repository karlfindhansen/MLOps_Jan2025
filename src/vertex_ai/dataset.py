import io
from collections.abc import Callable
from typing import Optional

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from google.cloud import storage


class CustomDataset(Dataset):
    def __init__(
        self,
        bucket_name: str,
        prefix: str,
        model_name: str,
        num_classes: int,
        transform: Optional[Callable] = None,
    ) -> None:
        """Initialize dataset."""
        self.bucket_name = bucket_name
        self.prefix = prefix.rstrip("/")  # Remove trailing slash if present
        self.model_name = model_name
        self.num_classes = num_classes
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        # Initialize GCS client
        self.client = storage.Client()
        self.bucket = self.client.bucket(self.bucket_name)

        # Prepare image paths and labels
        self.image_paths = []
        self.labels = []

        # Fetch all blobs in the bucket under the prefix
        all_blobs = self._list_blobs()

        # Debug: Print the number of blobs found
        print(f"Found {len(all_blobs)} blobs in the bucket.")

        # Group blobs by folder (assumes each folder represents a class)
        folder_to_idx = {}
        for blob in all_blobs:
            # Extract folder name from blob path
            path_parts = blob.name[len(self.prefix) + 1:].split("/")  # Remove prefix and split
            if len(path_parts) < 2:  # Skip files not organized in a folder structure
                continue
            folder_name = path_parts[0]

            # Assign folder to a class index if not already assigned
            if folder_name not in folder_to_idx and len(folder_to_idx) < self.num_classes:
                folder_to_idx[folder_name] = len(folder_to_idx)

            # Add image to dataset if folder is assigned to a class
            if folder_name in folder_to_idx:
                self.image_paths.append(blob.name)
                self.labels.append(folder_to_idx[folder_name])

        # Debug: Print the folders and their assigned indices
        print(f"Class mapping: {folder_to_idx}")
        print(f"Loaded {len(self.image_paths)} images and {len(self.labels)} labels.")

    def __len__(self) -> int:
        """Length of dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        """Get item from dataset."""
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Download image bytes from GCS
        blob = self.bucket.blob(image_path)
        image_bytes = blob.download_as_bytes()

        try:
            # Attempt to open the image
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except:
            raise RuntimeError(f"File {image_path} is not a valid image.")

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, label

    def _list_blobs(self):
        """List all blobs in the bucket under the prefix."""
        blobs = self.bucket.list_blobs(prefix=self.prefix)
        return [blob for blob in blobs]
