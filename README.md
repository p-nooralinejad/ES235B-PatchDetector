# PyTorch ResNet Models for CIFAR-10 

This repository contains PyTorch implementations of ResNet-18 and ResNet-20 architectures for image classification on CIFAR-10 dataset. It also provides training and testing scripts along with a convenient interface to select models, datasets, and other hyperparameters.

## Installation
Install the required dependencies:

```pip install -r requirements.txt```


## Usage

### Basic Training of models

To train a model, use the `main_basic.py` script. Below are the available arguments:

- `--model`: Model architecture (`resnet18` or `resnet20`)
- `--dataset`: Dataset (`cifar10` or `caltech101`)
- `--batch_size`: Batch size
- `--epochs`: Number of epochs
- `--lr`: Learning rate
- `--checkpoint_path`: Path to save checkpoints
- `--dataset_path`: Path to dataset

Example usage:

```python main_basic.py --model resnet18 --dataset cifar10 --batch_size 64 --epochs 10 --lr 0.001 --checkpoint_path saved_models```

### Attack Simulation

To simulate an attack on a pretrained model, use the `simulating_attack.py` script. Below are the available arguments:

- `--model`: Model architecture (`resnet18` or `resnet20`)
- `--dataset`: Dataset (`cifar10` or `caltech101`)
- `--checkpoint_path`: Path to save checkpoints
- `--dataset_path`: Path to dataset

Example usage:

```python simulating_attack.py --model resnet18 --dataset cifar10  --checkpoint_path saved_models```

### Training a Detector

To train a simple detector for a pretrained model, use the `main_new_head.py` script. Below are the available arguments:

- `--model`: Model architecture (`resnet18` or `resnet20`)
- `--dataset`: Dataset (`cifar10` or `caltech101`)
- `--batch_size`: Batch size
- `--epochs`: Number of epochs
- `--lr`: Learning rate
- `--checkpoint_path`: Path to save checkpoints
- `--dataset_path`: Path to dataset

Example usage:

```python main_new_head.py --model resnet18 --dataset cifar10 --batch_size 64 --epochs 10 --lr 0.001 --checkpoint_path saved_models```

## Requirements

- Python 3.x
- PyTorch
- Torchvision
- tqdm

