import torch
import os

def test_data():
    data_path = 'data'
    assert os.path.exists(data_path), f"Data path {data_path} does not exist"
    assert len(os.listdir(data_path)) > 0, f"Data path {data_path} has no subfolders"
    for folder in os.listdir(data_path):
        assert len(os.listdir(os.path.join(data_path, folder))) > 0, f"Folder {folder} is empty"
