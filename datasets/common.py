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
