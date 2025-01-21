import torch
import os
from src.datasets import CUB200201DataModule, CUB200_201
from torch.utils.data import Dataset

#def test_data():
#    data_path = 'data'
#    assert os.path.exists(data_path), f"Data path {data_path} does not exist"
#    assert len(os.listdir(data_path)) > 0, f"Data path {data_path} has no subfolders"
#    for folder in os.listdir(data_path):
#        assert len(os.listdir(os.path.join(data_path, folder))) > 0, f"Folder {folder} is empty"


# import os
# import subprocess

# def test_data():
#     # Path to the local data directory
#     data_path = 'data'

#     # first ensure data is pulled from the DVC remote (Google Cloud Storage)
#     try:
#         subprocess.run(['dvc', 'pull'], check=True)
#     except subprocess.CalledProcessError:
#         raise AssertionError("Failed to pull data using DVC. Ensure your DVC remote is correctly configured.")

#     # verify the local data path exists
#     assert os.path.exists(data_path), f"Data path {data_path} does not exist locally after pulling"

#     # check if the directory contains subfolders or files
#     assert len(os.listdir(data_path)) > 0, f"Data path {data_path} is empty after pulling"

#     # verify each subfolder is non-empty
#     for folder in os.listdir(data_path):
#         folder_path = os.path.join(data_path, folder)
#         assert os.path.isdir(folder_path), f"{folder_path} is not a directory"
#         assert len(os.listdir(folder_path)) > 0, f"Folder {folder} in {data_path} is empty"

# run the datamodule

def test_cub_200_2011_load():
    """Test the Caltech256 class."""
    dataset = CUB200_201(image_dir='../data/CUB_200_2011', model_name='resnet50', num_classes=200, download=True)
    assert isinstance(dataset, Dataset)
   #assert len(dataset) == N_IMAGES
