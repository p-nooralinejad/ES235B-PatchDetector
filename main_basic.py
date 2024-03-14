import argparse
import os
import shutil
import torch
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

def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10/Caltech-101 Training')
    parser.add_argument('--model', default='resnet18', choices=['resnet18', 'resnet20'], help='Model architecture')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'caltech101'], help='Dataset')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
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
        train_loader, test_loader = get_caltech101_dataloader(batch_size=args.batch_size, dataset_path=args.dataset_path)
        num_classes = 101  # Change this according to the number of classes in Caltech-101

    if args.model == 'resnet18':
        model = ResNet18(num_classes=num_classes).to(device)
    else:
        model = ResNet20(num_classes=num_classes).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_accuracy = 0.0
    for epoch in range(args.epochs):
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy = test(model, test_loader, criterion, device)

        print(f'Epoch [{epoch + 1}/{args.epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, '
              f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_accuracy': best_accuracy,
                'optimizer': optimizer.state_dict(),
            }, is_best=True, filename=f'model_best_{args.model}.pth.tar')

if __name__ == '__main__':
    main()
