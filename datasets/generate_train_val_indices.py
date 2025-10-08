#!/usr/bin/env python3
"""
Generate persistent train/val split indices for datasets.

⚠️ WARNING: This script should be run ONLY ONCE to generate the indices.
The generated indices will be used throughout the entire project to ensure
consistent train/val splits across all experiments.

DO NOT run this script multiple times as it will change the train/val split!
"""

import os
import numpy as np
import torchvision.datasets as datasets
from torchvision import transforms
from common import ImageFolderWithPaths


def generate_persistent_indices(data_location, val_ratio=0.1, random_seed=42):
    """
    Generate persistent train/val split indices with stratified sampling.
    
    ⚠️ WARNING: This creates a permanent train/val split. Only run once!
    """
    print("⚠️  WARNING: Generating PERSISTENT train/val split indices!")
    print("   These indices will be used for ALL future experiments.")
    print("   DO NOT run this script multiple times!\n")
    
    np.random.seed(random_seed)
    
    # Load dataset with simple transform
    transform = transforms.Compose([transforms.Resize(64), transforms.ToTensor()])
    train_dir = os.path.join(data_location, 'tiny-imagenet-200', 'train')
    dataset = ImageFolderWithPaths(train_dir, transform=transform)
    
    print(f"Total samples: {len(dataset.samples)}")
    print(f"Number of classes: {len(dataset.classes)}")
    
    # Group samples by class
    class_to_samples = {}
    for idx, (path, class_idx) in enumerate(dataset.samples):
        if class_idx not in class_to_samples:
            class_to_samples[class_idx] = []
        class_to_samples[class_idx].append(idx)
    
    # Verify stratification
    samples_per_class = len(class_to_samples[0])
    print(f"Samples per class: {samples_per_class}")
    
    # Create validation indices array
    val_indices = np.zeros(len(dataset.samples), dtype=bool)
    val_samples_per_class = int(samples_per_class * val_ratio)
    
    print(f"Validation samples per class: {val_samples_per_class}")
    print(f"Training samples per class: {samples_per_class - val_samples_per_class}")
    
    # For each class, randomly select validation samples
    for class_idx, sample_indices in class_to_samples.items():
        class_val_indices = np.random.choice(
            sample_indices, 
            size=val_samples_per_class, 
            replace=False
        )
        val_indices[class_val_indices] = True
    
    return val_indices


def save_indices(data_location, val_indices):
    """Save indices with metadata"""
    idx_file = os.path.join(data_location, 'tiny_imagenet_train_val_indices.npy')
    
    print(f"\n💾 Saving indices to: {idx_file}")
    np.save(idx_file, val_indices.astype(np.uint8))
    
    # Verification
    loaded = np.load(idx_file).astype(bool)
    print(f"✅ Validation samples: {np.sum(loaded)}")
    print(f"✅ Training samples: {np.sum(~loaded)}")
    
    print(f"\n🔒 INDICES SAVED! Use these for ALL future experiments.")
    print(f"   File: {idx_file}")


def generate_cifar10_indices(data_location, val_ratio=0.02, random_seed=42):
    """
    Generate persistent train/val split indices for CIFAR-10.

    Split: 98/2 (49k train / 1k val)

    ⚠️ WARNING: This creates a permanent train/val split. Only run once!
    """
    print("⚠️  WARNING: Generating PERSISTENT CIFAR-10 train/val split indices!")
    print("   These indices will be used for ALL future experiments.")
    print("   DO NOT run this script multiple times!\n")

    np.random.seed(random_seed)

    # Load CIFAR-10 training set
    transform = transforms.ToTensor()
    train_dataset = datasets.CIFAR10(
        root=data_location,
        train=True,
        download=True,
        transform=transform
    )

    total_samples = len(train_dataset)
    print(f"Total samples: {total_samples}")
    print(f"Number of classes: 10")

    # Group samples by class
    class_to_samples = {}
    for idx in range(total_samples):
        _, label = train_dataset[idx]
        if label not in class_to_samples:
            class_to_samples[label] = []
        class_to_samples[label].append(idx)

    # Verify stratification
    samples_per_class = len(class_to_samples[0])
    print(f"Samples per class: {samples_per_class}")

    # Create validation indices array
    val_indices = np.zeros(total_samples, dtype=bool)
    val_samples_per_class = int(samples_per_class * val_ratio)

    print(f"Validation samples per class: {val_samples_per_class}")
    print(f"Training samples per class: {samples_per_class - val_samples_per_class}")

    # For each class, randomly select validation samples
    for class_idx, sample_indices in class_to_samples.items():
        class_val_indices = np.random.choice(
            sample_indices,
            size=val_samples_per_class,
            replace=False
        )
        val_indices[class_val_indices] = True

    return val_indices


def save_cifar10_indices(data_location, val_indices):
    """Save CIFAR-10 indices with metadata"""
    idx_file = os.path.join(data_location, 'cifar10_train_val_indices.npy')

    print(f"\n💾 Saving indices to: {idx_file}")
    np.save(idx_file, val_indices.astype(np.uint8))

    # Verification
    loaded = np.load(idx_file).astype(bool)
    print(f"✅ Validation samples: {np.sum(loaded)}")
    print(f"✅ Training samples: {np.sum(~loaded)}")

    print(f"\n🔒 INDICES SAVED! Use these for ALL future experiments.")
    print(f"   File: {idx_file}")


if __name__ == "__main__":
    import sys

    # Determine which dataset to generate indices for
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1].lower()
    else:
        print("Usage: python generate_train_val_indices.py [cifar10|tinyimagenet]")
        dataset_name = input("Which dataset? (cifar10/tinyimagenet): ").lower()

    print("🚨 GENERATING PERSISTENT TRAIN/VAL SPLIT INDICES 🚨")
    print("="*60)

    response = input("Are you sure you want to generate NEW indices? (yes/no): ")
    if response.lower() != 'yes':
        print("❌ Cancelled. No indices generated.")
        exit(1)

    if dataset_name == 'cifar10':
        # CIFAR-10: Use current directory as data location
        data_location = os.path.dirname(os.path.abspath(__file__))

        print("\n📊 Generating stratified 98/2 train/val split for CIFAR-10...")
        val_indices = generate_cifar10_indices(data_location, val_ratio=0.02, random_seed=42)
        save_cifar10_indices(data_location, val_indices)

    elif dataset_name == 'tinyimagenet':
        # TinyImageNet: Use specific path
        data_location = "/Users/Yang/Desktop/model-merge/model-soups/clip_TinyImageNet/dataset"

        print("\n📊 Generating stratified 90/10 train/val split for TinyImageNet...")
        val_indices = generate_persistent_indices(data_location, val_ratio=0.1, random_seed=42)
        save_indices(data_location, val_indices)

    else:
        print(f"❌ Unknown dataset: {dataset_name}")
        print("   Supported: cifar10, tinyimagenet")
        exit(1)

    print("\n✅ DONE! Indices generated and saved.")
    print("   These indices will be used for all future experiments.")
    print("   DO NOT run this script again unless you want to change the split!")