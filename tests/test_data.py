import torch
import os

data_path = r'data'

def test_data():
    assert os.path.exists(data_path), f"Data path {data_path} does not exist"
    assert len(os.listdir(data_path)) == 10, f"Data path {data_path} does not contain 10 folders"
    for folder in os.listdir(data_path):
        assert len(os.listdir(os.path.join(data_path, folder))) > 0, f"Folder {folder} is empty"


