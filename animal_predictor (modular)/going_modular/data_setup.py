import os
import requests
import zipfile
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(train_dir : str, test_dir : str, train_data_transform : transforms.Compose, test_data_transform : transforms.Compose, batch_size : int, num_workers : int):
    train_data = datasets.ImageFolder(train_dir, transform = train_data_transform)
    test_data = datasets.ImageFolder(test_dir, transform = test_data_transform)
    class_names = train_data.classes
    train_dataloader = DataLoader(train_data, batch_size = batch_size, num_workers = NUM_WORKERS, shuffle = True, pin_memory = True)
    test_dataloader = DataLoader(test_data, batch_size = batch_size, num_workers = NUM_WORKERS, shuffle = True, pin_memory = True)
    return train_dataloader, test_dataloader, class_names

