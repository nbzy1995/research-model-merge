import os
import torch
import json
import glob
import collections
import random

import numpy as np

from tqdm import tqdm

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Sampler


class SubsetSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)

class ImageFolderWithPaths(datasets.ImageFolder):
    def __init__(self, path, transform):
        super().__init__(path, transform)

    def __getitem__(self, index):
        image, label = super(ImageFolderWithPaths, self).__getitem__(index)
        return {
            'images': image,
            'labels': label,
            'image_paths': self.samples[index][0]
        }


# CIFAR-10 Transform Factories
def get_cifar10_train_transform():
    """
    CIFAR-10 training transform with Git Re-Basin augmentation.
    - Random resized crop (0.8x-1.2x)
    - Random horizontal flips
    - Random rotations (±30°)
    """
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    return transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


def get_cifar10_eval_transform():
    """
    CIFAR-10 evaluation transform (no augmentation).
    """
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


import torch
import time
import numpy as np
import matplotlib.pyplot as plt


def eval_model_on_dataset(model, dataloader):
    """
    Evaluate model accuracy on the given dataloader.

    Args:
        model: PyTorch model
        dataloader: DataLoader for the dataset

    Returns:
        float: Accuracy (0-1 range)
    """
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        correct, n = 0, 0
        end = time.time()

        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            data_time = time.time() - end

            logits = model(inputs)
            pred = logits.argmax(dim=1, keepdim=True).to(device)
            correct += pred.eq(labels.view_as(pred)).sum().item()
            n += labels.size(0)

            batch_time = time.time() - end
            end = time.time()

            if i % 20 == 0:
                percent_complete = 100.0 * i / len(dataloader)
                print(
                    f"[{percent_complete:.0f}% {i}/{len(dataloader)}]\t"
                    f"Acc: {100 * (correct/n):.2f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}"
                )

        accuracy = correct / n
        return accuracy


def show_split_samples(dataset, split_name, batch_index=0):
    """
    Show sample images with labels from a dataset split.

    Args:
        dataset: CIFAR10 dataset object with train_loader, val_loader, test_loader
        split_name: 'train', 'validation', or 'test'
        batch_index: Which batch to visualize
    """
    # Get the appropriate loader
    if split_name == "train":
        loader = dataset.train_loader
    elif split_name == "validation":
        loader = dataset.val_loader
    elif split_name == "test":
        loader = dataset.test_loader
    else:
        raise ValueError(f"Unknown split: {split_name}. Use 'train', 'validation', or 'test'")

    # Get one batch
    from itertools import islice
    batch = next(islice(loader, batch_index, None))
    images, labels = batch

    print(f"Fetching {len(labels)} samples from {split_name} split")

    # Create subplot (2x4 grid for 8 images)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    # CIFAR-10 normalization constants
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)

    for i in range(min(len(images), 8)):
        # Denormalize image
        img = images[i] * std + mean
        img = torch.clamp(img, 0, 1)
        img = img.permute(1, 2, 0)

        axes[i].imshow(img)
        axes[i].axis('off')

        # Get label info
        label_idx = int(labels[i])
        class_name = dataset.classnames[label_idx]

        axes[i].set_title(f"Class {label_idx}: {class_name}", fontsize=12)

    plt.tight_layout()
    plt.suptitle(f'{split_name.upper()} Split - Sample Images', fontsize=14, y=1.02)
    plt.show()
    print()


def show_split_samples_with_predictions(model, dataset, split_name, batch_index=0):
    """
    Show sample images with true labels and model predictions.

    Args:
        model: PyTorch model for prediction
        dataset: CIFAR10 dataset object
        split_name: 'train', 'validation', or 'test'
        batch_index: Which batch to visualize
    """
    # Get the appropriate loader
    if split_name == "train":
        loader = dataset.train_loader
    elif split_name == "validation":
        loader = dataset.val_loader
    elif split_name == "test":
        loader = dataset.test_loader
    else:
        raise ValueError(f"Unknown split: {split_name}. Use 'train', 'validation', or 'test'")

    # Get one batch
    from itertools import islice
    batch = next(islice(loader, batch_index, None))
    images, labels = batch

    print(f"Fetching {len(labels)} samples from {split_name} split")

    # Get predictions
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        logits = model(images.to(device))
    pred = logits.argmax(dim=1).cpu()

    print(f"Predictions: {pred.numpy()}")

    # Create subplot (2x4 grid for 8 images)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    # CIFAR-10 normalization constants
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)

    for i in range(min(len(images), 8)):
        # Denormalize image
        img = images[i] * std + mean
        img = torch.clamp(img, 0, 1)
        img = img.permute(1, 2, 0)

        axes[i].imshow(img)
        axes[i].axis('off')

        # Get label info
        pred_idx = int(pred[i])
        true_idx = int(labels[i])

        pred_class = dataset.classnames[pred_idx]
        true_class = dataset.classnames[true_idx]
        is_correct = (pred_idx == true_idx)

        # Green if correct, red if wrong
        color = 'green' if is_correct else 'red'
        axes[i].set_title(
            f"True: {true_class}\nPred: {pred_class}",
            color=color,
            fontsize=10
        )

    plt.tight_layout()
    plt.suptitle(f'{split_name.upper()} Split - Predictions', fontsize=14, y=1.02)
    plt.show()
    print()
