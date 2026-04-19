#import libraries
import torch
from torch import nn

import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def create_dataloader(batch_size):
    """Creates a dataloader for the naira note classifier"""
    train_path = "data/train/"
    test_path = "data/test/"
    val_path = "data/valid/"

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    train_dataset = datasets.ImageFolder(train_path, train_transform)
    test_dataset = datasets.ImageFolder(test_path, test_transform)
    val_dataset = datasets.ImageFolder(val_path, test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    class_names = train_dataset.classes
    print(f"Classes: {class_names}, No of train samples: {len(train_dataset)}, No of test samples: {len(test_dataset)}, No of val samples: {len(val_dataset)}")
    return train_loader, test_loader, val_loader, class_names


