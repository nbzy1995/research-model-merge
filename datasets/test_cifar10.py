#!/usr/bin/env python3
"""
Self-contained test suite for CIFAR10 dataset class.

Tests verify:
1. Split ratios are correct (49k train / 1k val / 10k test)
2. No data leakage between train/val/test
3. Stratification: equal samples per class in validation
4. Persistent indices reproducibility
5. Transforms applied correctly
"""

import os
import sys
import numpy as np
import torch

# Add parent directory to path for imports
datasets_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(datasets_dir)
sys.path.insert(0, parent_dir)

from datasets.cifar10 import CIFAR10, load_persistent_indices


def test_split_ratios():
    """Test that split ratios are correct: 98/2 (49k/1k)"""
    print("\n" + "="*60)
    print("TEST 1: Split Ratios")
    print("="*60)

    data_location = os.path.dirname(os.path.abspath(__file__))
    dataset = CIFAR10(data_location, batch_size=100, num_workers=0)

    # Count samples
    train_count = len(dataset.train_sampler)
    val_count = len(dataset.val_sampler)
    test_count = len(dataset.test_dataset)

    print(f"Train samples: {train_count}")
    print(f"Val samples: {val_count}")
    print(f"Test samples: {test_count}")

    # Expected counts
    expected_train = 49000
    expected_val = 1000
    expected_test = 10000

    assert train_count == expected_train, f"Expected {expected_train} train samples, got {train_count}"
    assert val_count == expected_val, f"Expected {expected_val} val samples, got {val_count}"
    assert test_count == expected_test, f"Expected {expected_test} test samples, got {test_count}"

    print("✅ PASSED: Split ratios are correct (49k/1k/10k)")


def test_no_data_leakage():
    """Test that train and val sets have no overlapping indices"""
    print("\n" + "="*60)
    print("TEST 2: No Data Leakage")
    print("="*60)

    data_location = os.path.dirname(os.path.abspath(__file__))
    dataset = CIFAR10(data_location, batch_size=100, num_workers=0)

    # Get indices
    train_indices = set(dataset.train_sampler.indices)
    val_indices = set(dataset.val_sampler.indices)

    # Check for overlap
    overlap = train_indices & val_indices

    print(f"Train indices count: {len(train_indices)}")
    print(f"Val indices count: {len(val_indices)}")
    print(f"Overlapping indices: {len(overlap)}")

    assert len(overlap) == 0, f"Found {len(overlap)} overlapping indices between train and val!"

    # Check total coverage
    total = len(train_indices) + len(val_indices)
    expected_total = 50000

    assert total == expected_total, f"Expected {expected_total} total indices, got {total}"

    print("✅ PASSED: No data leakage between train and val")


def test_stratification():
    """Test that validation set has equal samples per class"""
    print("\n" + "="*60)
    print("TEST 3: Stratification")
    print("="*60)

    data_location = os.path.dirname(os.path.abspath(__file__))
    dataset = CIFAR10(data_location, batch_size=100, num_workers=0)

    # Count samples per class in validation set
    class_counts = [0] * 10

    for idx in dataset.val_sampler.indices:
        _, label = dataset.val_dataset[idx]
        class_counts[label] += 1

    print("Validation samples per class:")
    for i, count in enumerate(class_counts):
        print(f"  Class {i} ({dataset.classnames[i]}): {count}")

    # All classes should have exactly 100 samples (1000 / 10 = 100)
    expected_per_class = 100

    for i, count in enumerate(class_counts):
        assert count == expected_per_class, \
            f"Class {i} has {count} samples, expected {expected_per_class}"

    print(f"✅ PASSED: All classes have {expected_per_class} validation samples")


def test_persistent_indices_reproducibility():
    """Test that loading indices twice gives the same split"""
    print("\n" + "="*60)
    print("TEST 4: Persistent Indices Reproducibility")
    print("="*60)

    data_location = os.path.dirname(os.path.abspath(__file__))

    # Load indices twice
    indices1 = load_persistent_indices(data_location)
    indices2 = load_persistent_indices(data_location)

    # Should be identical
    assert np.array_equal(indices1, indices2), "Indices changed between loads!"

    print(f"Indices shape: {indices1.shape}")
    print(f"Val samples (True): {np.sum(indices1)}")
    print(f"Train samples (False): {np.sum(~indices1)}")

    # Create two dataset instances
    dataset1 = CIFAR10(data_location, batch_size=100, num_workers=0)
    dataset2 = CIFAR10(data_location, batch_size=100, num_workers=0)

    # Compare indices
    train_indices1 = list(dataset1.train_sampler.indices)
    train_indices2 = list(dataset2.train_sampler.indices)

    val_indices1 = list(dataset1.val_sampler.indices)
    val_indices2 = list(dataset2.val_sampler.indices)

    # Val indices should be identical (deterministic)
    assert val_indices1 == val_indices2, "Validation indices changed!"

    print("✅ PASSED: Persistent indices are reproducible")


def test_transforms():
    """Test that transforms are applied correctly"""
    print("\n" + "="*60)
    print("TEST 5: Transforms Applied")
    print("="*60)

    data_location = os.path.dirname(os.path.abspath(__file__))
    dataset = CIFAR10(data_location, batch_size=2, num_workers=0)

    # Get one batch from each loader
    train_batch = next(iter(dataset.train_loader))
    val_batch = next(iter(dataset.val_loader))
    test_batch = next(iter(dataset.test_loader))

    print(f"Train batch shape: {train_batch[0].shape}")
    print(f"Val batch shape: {val_batch[0].shape}")
    print(f"Test batch shape: {test_batch[0].shape}")

    # All should be [batch_size, 3, 32, 32]
    assert train_batch[0].shape == (2, 3, 32, 32), f"Unexpected train shape: {train_batch[0].shape}"
    assert val_batch[0].shape == (2, 3, 32, 32), f"Unexpected val shape: {val_batch[0].shape}"
    assert test_batch[0].shape == (2, 3, 32, 32), f"Unexpected test shape: {test_batch[0].shape}"

    # Check normalization (values should be roughly in range [-3, 3])
    train_min, train_max = train_batch[0].min(), train_batch[0].max()
    val_min, val_max = val_batch[0].min(), val_batch[0].max()

    print(f"Train batch range: [{train_min:.2f}, {train_max:.2f}]")
    print(f"Val batch range: [{val_min:.2f}, {val_max:.2f}]")

    # Normalized values should be roughly in [-3, 3] range
    assert -5 < train_min < 0, f"Train min {train_min} seems unnormalized"
    assert 0 < train_max < 5, f"Train max {train_max} seems unnormalized"
    assert -5 < val_min < 0, f"Val min {val_min} seems unnormalized"
    assert 0 < val_max < 5, f"Val max {val_max} seems unnormalized"

    print("✅ PASSED: Transforms applied correctly")


def test_classnames():
    """Test that classnames are correct"""
    print("\n" + "="*60)
    print("TEST 6: Classnames")
    print("="*60)

    data_location = os.path.dirname(os.path.abspath(__file__))
    dataset = CIFAR10(data_location, batch_size=100, num_workers=0)

    expected_classnames = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    print(f"Classnames: {dataset.classnames}")

    assert dataset.classnames == expected_classnames, \
        f"Classnames mismatch: {dataset.classnames} != {expected_classnames}"

    print("✅ PASSED: Classnames are correct")


def run_all_tests():
    """Run all tests"""
    print("\n" + "#"*60)
    print("# CIFAR10 Dataset Test Suite")
    print("#"*60)

    try:
        test_split_ratios()
        test_no_data_leakage()
        test_stratification()
        test_persistent_indices_reproducibility()
        test_transforms()
        test_classnames()

        print("\n" + "="*60)
        print("ALL TESTS PASSED ✅")
        print("="*60)
        print("\nCIFAR10 dataset implementation is verified and working correctly!")

        return True

    except AssertionError as e:
        print("\n" + "="*60)
        print("TEST FAILED ❌")
        print("="*60)
        print(f"\nError: {e}")
        return False

    except Exception as e:
        print("\n" + "="*60)
        print("TEST ERROR ❌")
        print("="*60)
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
