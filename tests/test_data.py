import torch
import os
import hashlib

def test_data():
    data_path = 'data'
    assert os.path.exists(data_path), f"Data path {data_path} does not exist"
    assert len(os.listdir(data_path)) == 4, f"Data path {data_path} does not contain 2 folders"
    for folder in os.listdir(data_path):
        assert len(os.listdir(os.path.join(data_path, folder))) > 0, f"Folder {folder} is empty"

# tets if all classes (data folders) have data, flag if tehre is extreme class imbalance
def test_class_distribution():
    data_path = 'data'
    class_counts = {folder: len(os.listdir(os.path.join(data_path, folder))) for folder in os.listdir(data_path)}
    for cls, count in class_counts.items():
        assert count > 0, f"Class {cls} has no data"
    print(f"Class distribution: {class_counts}")

# test is tehre are any duplicate files in the data
def test_no_duplicate_files():
    data_path = 'data'
    file_hashes = set()
    for folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder)
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            assert file_hash not in file_hashes, f"Duplicate file detected: {file_path}"
            file_hashes.add(file_hash)
