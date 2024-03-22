import argparse
import os
import shutil
import random
import torchvision.models as tmodels
import torch
import torch.nn as nn
from models import ResNet18, ResNet20
from dataloaders import get_cifar10_dataloader, get_caltech101_dataloader
from train_test import train, test

def attack(img, patch_size=6):
    x = random.randint(10, img.shape[2] - patch_size - 10)
    y = random.randint(10, img.shape[3] - patch_size - 10)
    
    # Create a random patch
    patch = torch.randn((img.shape[0], 3, patch_size, patch_size), device=img.device)
    
    # Apply the patch to the input tensor
    img[:, :, x:x+patch_size, y:y+patch_size] = patch
    return img

def test(model, test_loader):
    total = 0 
    correct = 0
    for idx, data in enumerate(test_loader):
        inputs, _ = data
        coin = random.randint(0,1)

        if coin:
            inputs = attack(inputs)
            labels = torch.ones(inputs.shape[0], dtype=torch.long)
        
        else:
            labels = torch.zeros(inputs.shape[0],  dtype=torch.long)
        
        out = model(inputs)
        _, predicted = out.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return (correct / total) * 100

def main():
    torch.manual_seed(0)
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10/Caltech-101 Training')
    parser.add_argument('--model', default='resnet20', choices=['resnet18', 'resnet20'], help='Model architecture')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'caltech101'], help='Dataset')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--checkpoint_path', type=str, default='saved_models', help='Path to save checkpoints')
    parser.add_argument('--dataset_path', type=str, default='./data', help='Path to dataset')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == 'cifar10':
        train_loader, test_loader = get_cifar10_dataloader(batch_size=args.batch_size, dataset_path=args.dataset_path)
        num_classes = 10
    else:
        pass
       

    model = tmodels.resnet18(pretrained=True)

    num_ftrs = model.fc.in_features
    for p in model.parameters():
        p.requires_grad = False
    
    model.fc = nn.Linear(num_ftrs, 2)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=args.lr)

    best_accuracy = 0.0
    best_model_state = {}

    for epoch in range(args.epochs):


        for idx, data in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, _ = data
            coin = random.randint(0,1)

            if coin:
                inputs = attack(inputs)
                labels = torch.ones(inputs.shape[0], dtype=torch.long)
            
            else:
                labels = torch.zeros(inputs.shape[0],  dtype=torch.long)
            
            out = model(inputs)
            loss = criterion(out, labels)

            if idx % 50 == 0:
                print(epoch, idx, loss.item())
            loss.backward()
            optimizer.step()
        
        acc = test(model, test_loader)
        print(acc)
        if acc > best_accuracy:
            best_model_state = model.state_dict()

    torch.save(best_model_state, 'best_detector.pt')

if __name__ == '__main__':
    main()
