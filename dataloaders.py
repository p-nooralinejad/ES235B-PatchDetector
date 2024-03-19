import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, Caltech101
from torch.utils.data import DataLoader

def get_cifar10_dataloader(batch_size=64, dataset_path='./data'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = CIFAR10(root=dataset_path, train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root=dataset_path, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def get_caltech101_dataloader(batch_size=64, dataset_path='./data'):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = Caltech101(root=dataset_path, download=True, transform=transform, target_type='category')
    test_dataset = Caltech101(root=dataset_path, download=True, transform=transform , target_type='category')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
