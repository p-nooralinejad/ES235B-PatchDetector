import argparse
import os
import shutil
import random
import torch
import torch.nn as nn
from models import ResNet18, ResNet20
from dataloaders import get_cifar10_dataloader, get_caltech101_dataloader
from train_test import train, test

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    directory = "saved_models"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, filename)
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(directory, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)

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

    num_ftrs = model.fc.in_features
    for p in model.parameters():
        p.requires_grad = False
    
    model.fc = nn.Linear(num_ftrs, 2)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=args.lr)

    best_accuracy = 0.0
    best_model_state = {}

    for epoch in range(args.epochs):
        # train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
        # test_loss, test_accuracy = test(model, test_loader, criterion, device)

        # print(f'Epoch [{epoch + 1}/{args.epochs}], '
        #       f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, '
        #       f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

        # if test_accuracy > best_accuracy:
        #     best_accuracy = test_accuracy
        #     save_checkpoint({
        #         'epoch': epoch + 1,
        #         'state_dict': model.state_dict(),
        #         'best_accuracy': best_accuracy,
        #         'optimizer': optimizer.state_dict(),
        #     }, is_best=True, filename=f'model_best_{args.model}.pth.tar')

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
            best_model_state = model.state_dict

    torch.save(best_model_state, 'best_detector.pt')

if __name__ == '__main__':
    main()
