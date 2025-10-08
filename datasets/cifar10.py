import os
import torch
import numpy as np
import torchvision.datasets as datasets
from torch.utils.data import SubsetRandomSampler
from .common import SubsetSampler, get_cifar10_train_transform, get_cifar10_eval_transform


def load_persistent_indices(data_location):
    """
    Load persistent train/val split indices.

    ⚠️ These indices should be generated ONCE using generate_train_val_indices.py
    and then used consistently across all experiments.
    """
    idx_file = os.path.join(data_location, 'cifar10_train_val_indices.npy')

    if not os.path.exists(idx_file):
        raise FileNotFoundError(
            f"Persistent indices file not found: {idx_file}\n"
            f"Run generate_train_val_indices.py ONCE to create the indices."
        )

    val_indices = np.load(idx_file).astype(bool)
    return val_indices


class CIFAR10:
    """
    CIFAR-10 dataset class for training, validation, and test splits.

    Split strategy: 98/2 train/val from original training set (49k/1k)
    Test set: Official CIFAR-10 test set (10k)

    Uses persistent indices for consistent splits across experiments.
    """
    def __init__(self,
                 data_location,
                 train_preprocess=None,
                 eval_preprocess=None,
                 batch_size=100,
                 num_workers=2,
                 distributed=False):
        self.data_location = data_location
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed = distributed

        # Use default transforms if not provided
        self.train_preprocess = train_preprocess if train_preprocess else get_cifar10_train_transform()
        self.eval_preprocess = eval_preprocess if eval_preprocess else get_cifar10_eval_transform()

        # CIFAR-10 class names
        self.classnames = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]

        self.populate_train()
        self.populate_val()
        self.populate_test()

    def populate_train(self):
        """
        This is 98% subset of the original train set, we use as train split.
        Uses persistent indices for consistent train/val split.
        """
        # Load original train dataset with training transforms
        self.train_dataset = datasets.CIFAR10(
            root=self.data_location,
            train=True,
            download=True,
            transform=self.train_preprocess
        )

        # Get training split indices (where val_indices is False)
        val_indices = load_persistent_indices(self.data_location)
        train_indices = np.where(~val_indices)[0]
        self.train_sampler = SubsetRandomSampler(train_indices)

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            sampler=self.train_sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def populate_val(self):
        """
        This is 2% subset of the original train set, we use as val split.
        Uses persistent indices for consistent train/val split.
        """
        # Load original train dataset with evaluation transforms
        self.val_dataset = datasets.CIFAR10(
            root=self.data_location,
            train=True,
            download=True,
            transform=self.eval_preprocess
        )

        # Get validation split indices (where val_indices is True)
        val_indices = load_persistent_indices(self.data_location)
        val_indices = np.where(val_indices)[0]
        self.val_sampler = SubsetSampler(val_indices)  # sequential order

        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            sampler=self.val_sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def populate_test(self):
        """
        The official CIFAR-10 test set.
        """
        self.test_dataset = datasets.CIFAR10(
            root=self.data_location,
            train=False,
            download=True,
            transform=self.eval_preprocess
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False
        )

    def name(self):
        return 'cifar10'
