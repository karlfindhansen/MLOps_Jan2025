import os
import pytest
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from src.datasets import CUB200_201 

N_IMAGES = 6

def test_cub_200_2011_initialization():
    dataset = CUB200_201(
        image_dir='./data/CUB_200_2011/images',
        model_name='resnet50',
        num_classes=200,
        download=True
    )
    assert isinstance(dataset, Dataset), "Dataset is not an instance of torch.utils.data.Dataset."
    assert len(dataset) > 0, "Dataset is empty."
    assert len(dataset) == N_IMAGES, f"Dataset size is {len(dataset)}, expected {N_IMAGES}."
    assert hasattr(dataset, 'image_paths'), "Dataset is missing the 'image_paths' attribute."
    assert isinstance(dataset.image_paths, list), "'image_paths' should be a list."

def test_cub_200_2011_file_validation():
    dataset = CUB200_201(
        image_dir='./data/CUB_200_2011/images',
        model_name='resnet50',
        num_classes=200,
        download=True
    )
    dataset.image_paths = [path for path in dataset.image_paths if not path.endswith('README')]
    for path in dataset.image_paths:
        assert os.path.exists(path), f"File does not exist: {path}"
        assert path.lower().endswith(('.jpg', '.jpeg', '.png', '.txt', 'README')), f"Invalid file type: {path}"
