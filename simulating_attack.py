import argparse
import os
import random
import shutil
import torch
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

def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10/Caltech-101 Training')
    parser.add_argument('--model', default='resnet20', choices=['resnet18', 'resnet20'], help='Model architecture')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'caltech101'], help='Dataset')
    parser.add_argument('--checkpoint_path', type=str, default='saved_models', help='Path to save checkpoints')
    parser.add_argument('--dataset_path', type=str, default='./data', help='Path to dataset')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == 'cifar10':
        train_loader, test_loader = get_cifar10_dataloader(batch_size=100, dataset_path=args.dataset_path)
        num_classes = 10
    else:
        pass
        # train_loader, test_loader = get_caltech101_dataloader(batch_size=args.batch_size, dataset_path=args.dataset_path)
        # num_classes = 101  # Change this according to the number of classes in Caltech-101

    if args.model == 'resnet18':
        model = ResNet18(num_classes=num_classes).to(device)
    else:
        model = ResNet20(num_classes=num_classes).to(device)

    dictionary = torch.load(args.checkpoint_path)['state_dict']
    if 'linear.weight' in dictionary.keys():
        dictionary['fc.weight'] = dictionary['linear.weight']
        dictionary.pop('linear.weight')
    if 'linear.bias' in dictionary.keys():
        dictionary['fc.bias'] = dictionary['linear.bias']
        dictionary.pop('linear.bias')
    
    model.load_state_dict(dictionary)


    total =0
    correct = 0
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            inputs, label = data
            patch_data = attack(inputs)
            outputs = model(patch_data)
            _, predicted = outputs.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
        
        print("accuracy:", correct / total * 100)

if __name__ == '__main__':
    main()
