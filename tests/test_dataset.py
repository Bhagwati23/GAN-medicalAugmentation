"""
Unit tests for dataset loading and preparation utilities.
"""

import pytest
import numpy as np
from src.data.dataset import (
    get_medmnist_info,
    load_breast_mnist,
    load_pneumonia_mnist,
    get_dataset_statistics,
    load_full_dataset,
    get_training_data_only,
    verify_split_separation,
    set_random_seed,
    demonstrate_reproducibility
)


class TestMedMNISTInfo:
    """Tests for MedMNIST metadata retrieval."""

    def test_get_breastmnist_info(self):
        """Test retrieving BreastMNIST metadata."""
        info = get_medmnist_info('breastmnist')
        assert info is not None
        assert 'task' in info
        assert 'n_channels' in info
        assert 'n_samples' in info

    def test_get_pneumoniamnist_info(self):
        """Test retrieving PneumoniaMNIST metadata."""
        info = get_medmnist_info('pneumoniamnist')
        assert info is not None
        assert 'task' in info
        assert 'n_channels' in info
        assert 'n_samples' in info

    def test_invalid_dataset_name(self):
        """Test that invalid dataset names raise ValueError."""
        with pytest.raises(ValueError):
            get_medmnist_info('invalidmnist')


class TestDatasetLoading:
    """Tests for dataset loading functions."""

    def test_load_breast_mnist_train(self):
        """Test loading BreastMNIST training set."""
        images, labels = load_breast_mnist(split='train', download=True)

        assert images is not None
        assert labels is not None
        assert len(images) == len(labels)
        assert len(images.shape) == 4  # (N, H, W, C)
        assert images.shape[1:3] == (28, 28)
        assert labels.dtype in [np.int64, np.int32, np.int_]

    def test_load_breast_mnist_val(self):
        """Test loading BreastMNIST validation set."""
        images, labels = load_breast_mnist(split='val', download=True)

        assert images is not None
        assert labels is not None
        assert len(images) == len(labels)
        assert images.shape[1:3] == (28, 28)

    def test_load_breast_mnist_test(self):
        """Test loading BreastMNIST test set."""
        images, labels = load_breast_mnist(split='test', download=True)

        assert images is not None
        assert labels is not None
        assert len(images) == len(labels)
        assert images.shape[1:3] == (28, 28)

    def test_load_pneumonia_mnist_train(self):
        """Test loading PneumoniaMNIST training set."""
        images, labels = load_pneumonia_mnist(split='train', download=True)

        assert images is not None
        assert labels is not None
        assert len(images) == len(labels)
        assert len(images.shape) == 4  # (N, H, W, C)
        assert images.shape[1:3] == (28, 28)
        assert labels.dtype in [np.int64, np.int32, np.int_]

    def test_load_full_breast_dataset(self):
        """Test loading all splits of BreastMNIST."""
        data = load_full_dataset('breastmnist', download=True)

        assert 'train' in data
        assert 'val' in data
        assert 'test' in data

        train_images, train_labels = data['train']
        val_images, val_labels = data['val']
        test_images, test_labels = data['test']

        assert len(train_images) > 0
        assert len(val_images) > 0
        assert len(test_images) > 0

    def test_load_full_pneumonia_dataset(self):
        """Test loading all splits of PneumoniaMNIST."""
        data = load_full_dataset('pneumoniamnist', download=True)

        assert 'train' in data
        assert 'val' in data
        assert 'test' in data

        train_images, train_labels = data['train']
        val_images, val_labels = data['val']
        test_images, test_labels = data['test']

        assert len(train_images) > 0
        assert len(val_images) > 0
        assert len(test_images) > 0


class TestDatasetStatistics:
    """Tests for dataset statistics calculation."""

    def test_get_statistics(self):
        """Test calculating dataset statistics."""
        images, labels = load_breast_mnist(split='train', download=True)
        stats = get_dataset_statistics(images, labels)

        assert 'num_samples' in stats
        assert 'image_shape' in stats
        assert 'num_channels' in stats
        assert 'num_classes' in stats
        assert 'class_distribution' in stats
        assert 'pixel_min' in stats
        assert 'pixel_max' in stats

        assert stats['num_samples'] == len(images)
        assert stats['num_classes'] > 0

    def test_class_distribution(self):
        """Test that class distribution sums to total samples."""
        images, labels = load_pneumonia_mnist(split='train', download=True)
        stats = get_dataset_statistics(images, labels)

        total_from_dist = sum(stats['class_distribution'].values())
        assert total_from_dist == len(images)


class TestDataLeakagePrevention:
    """Tests to ensure test data isolation."""

    def test_training_data_only_excludes_test(self):
        """Test that get_training_data_only returns only training data."""
        train_only_images, train_only_labels = get_training_data_only(
            'breastmnist',
            download=True,
            include_val=False
        )

        # Load full dataset to compare
        full_data = load_full_dataset('breastmnist', download=True)
        train_images, train_labels = full_data['train']
        test_images, test_labels = full_data['test']

        # Training-only should match the train split size
        assert len(train_only_images) == len(train_images)
        assert len(train_only_labels) == len(train_labels)

        # Verify arrays are actually identical to official training split
        assert np.array_equal(train_only_images, train_images), "Training images don't match official split"
        assert np.array_equal(train_only_labels, train_labels), "Training labels don't match official split"

        # Training-only should NOT include test data
        assert len(train_only_images) != len(train_images) + len(test_images)

    def test_training_data_with_validation(self):
        """Test that include_val adds validation data."""
        train_only = get_training_data_only('breastmnist', download=True, include_val=False)
        train_val = get_training_data_only('breastmnist', download=True, include_val=True)

        # With validation should have more samples
        assert len(train_val[0]) > len(train_only[0])

    def test_split_separation(self):
        """Test that train/val/test splits pass validation.

        This test verifies:
        1. No train-test overlap (CRITICAL - always fails if detected)
        2. No val-test overlap (always fails if detected)
        3. Train-val overlap is within acceptable tolerance (< 1%)

        Note: Small numbers of pixel-identical images between train and val
        may exist in medical imaging datasets without sample IDs. These could be:
        - Different patients with identical imaging patterns
        - Uniform/blank images
        - Common visual artifacts

        The critical requirement is that test data never overlaps with train data.
        """
        data = load_full_dataset('pneumoniamnist', download=True)

        is_separated = verify_split_separation(data, allow_pixel_duplicates=True)
        assert is_separated == True

    def test_splits_are_non_empty(self):
        """Test that train, val, and test splits are non-empty."""
        data = load_full_dataset('breastmnist', download=True)

        train_size = len(data['train'][0])
        val_size = len(data['val'][0])
        test_size = len(data['test'][0])

        # All splits should be non-empty
        assert train_size > 0, "Training split is empty"
        assert val_size > 0, "Validation split is empty"
        assert test_size > 0, "Test split is empty"

    def test_no_train_val_overlap(self):
        """Test that training and validation splits don't overlap."""
        import hashlib

        data = load_full_dataset('breastmnist', download=True)
        train_images, _ = data['train']
        val_images, _ = data['val']

        # Create SHA-256 hash sets
        train_hashes = {hashlib.sha256(img.tobytes()).hexdigest() for img in train_images}
        val_hashes = {hashlib.sha256(img.tobytes()).hexdigest() for img in val_images}

        # Check for overlap
        overlap = train_hashes & val_hashes
        assert len(overlap) == 0, f"Found {len(overlap)} overlapping samples between train and val"

    def test_no_train_test_overlap(self):
        """Test that training and test splits don't overlap (CRITICAL)."""
        import hashlib

        data = load_full_dataset('pneumoniamnist', download=True)
        train_images, _ = data['train']
        test_images, _ = data['test']

        # Create SHA-256 hash sets
        train_hashes = {hashlib.sha256(img.tobytes()).hexdigest() for img in train_images}
        test_hashes = {hashlib.sha256(img.tobytes()).hexdigest() for img in test_images}

        # Check for overlap - THIS MUST BE ZERO
        overlap = train_hashes & test_hashes
        assert len(overlap) == 0, f"CRITICAL: Found {len(overlap)} overlapping samples between train and test"

    def test_no_val_test_overlap(self):
        """Test that validation and test splits don't overlap."""
        import hashlib

        data = load_full_dataset('pneumoniamnist', download=True)
        val_images, _ = data['val']
        test_images, _ = data['test']

        # Create SHA-256 hash sets
        val_hashes = {hashlib.sha256(img.tobytes()).hexdigest() for img in val_images}
        test_hashes = {hashlib.sha256(img.tobytes()).hexdigest() for img in test_images}

        # Check for overlap
        overlap = val_hashes & test_hashes
        assert len(overlap) == 0, f"Found {len(overlap)} overlapping samples between val and test"


class TestReproducibility:
    """Tests for reproducibility of data loading."""

    def test_set_random_seed(self):
        """Test that random seed setting works."""
        set_random_seed(42)
        sample1 = np.random.rand(10)

        set_random_seed(42)
        sample2 = np.random.rand(10)

        assert np.array_equal(sample1, sample2)

    def test_demonstrate_reproducibility_breast(self):
        """Test reproducibility demonstration for BreastMNIST."""
        result = demonstrate_reproducibility('breastmnist', seed=42, n_samples=3)

        assert 'identical' in result
        assert result['identical'] == True
        assert 'indices1' in result
        assert 'indices2' in result
        assert np.array_equal(result['indices1'], result['indices2'])

    def test_demonstrate_reproducibility_pneumonia(self):
        """Test reproducibility demonstration for PneumoniaMNIST."""
        result = demonstrate_reproducibility('pneumoniamnist', seed=123, n_samples=5)

        assert 'identical' in result
        assert result['identical'] == True


class TestImageProperties:
    """Tests for image properties and validity."""

    def test_image_dimensions(self):
        """Test that images have correct dimensions."""
        images, _ = load_breast_mnist(split='train', download=True)

        assert len(images.shape) == 4  # (N, H, W, C)
        assert images.shape[1] == 28
        assert images.shape[2] == 28

    def test_pixel_range(self):
        """Test that pixel values are in valid range."""
        images, _ = load_pneumonia_mnist(split='train', download=True)

        assert images.min() >= 0
        assert images.max() <= 255

    def test_labels_are_valid(self):
        """Test that labels are valid class indices."""
        images, labels = load_breast_mnist(split='train', download=True)

        # BreastMNIST has 2 classes (0 and 1)
        assert labels.min() >= 0
        assert labels.max() < 2

        # All labels should be integers
        assert labels.dtype in [np.int64, np.int32, np.int_]


class TestErrorHandling:
    """Tests for error handling."""

    def test_invalid_dataset_name_in_load_full(self):
        """Test that invalid dataset names are rejected."""
        with pytest.raises(ValueError):
            load_full_dataset('invalidmnist', download=True)

    def test_invalid_dataset_name_in_training_only(self):
        """Test that invalid dataset names are rejected in training_only."""
        with pytest.raises(ValueError):
            get_training_data_only('invalidmnist', download=True)


class TestSplitSeparationImproved:
    """Tests for improved verify_split_separation functionality."""

    def test_identical_images_detected(self):
        """Test that identical images across splits are detected."""
        # Create synthetic data with known duplicates
        # Train: 10 unique images
        # Val: 5 images, where 2 are duplicates from train
        # Test: 5 completely unique images

        np.random.seed(42)
        train_images = np.random.randint(0, 255, (10, 28, 28, 1), dtype=np.uint8)
        train_labels = np.zeros(10, dtype=np.int64)

        # Val contains 2 duplicates from train and 3 unique
        val_images = np.concatenate([
            train_images[[0, 5]],  # 2 duplicates
            np.random.randint(0, 255, (3, 28, 28, 1), dtype=np.uint8)  # 3 unique
        ], axis=0)
        val_labels = np.zeros(5, dtype=np.int64)

        # Test is completely unique
        test_images = np.random.randint(0, 255, (5, 28, 28, 1), dtype=np.uint8)
        test_labels = np.zeros(5, dtype=np.int64)

        data = {
            'train': (train_images, train_labels),
            'val': (val_images, val_labels),
            'test': (test_images, test_labels)
        }

        # Should pass with allow_pixel_duplicates=True (2/10 = 20% but we can increase threshold)
        result = verify_split_separation(data, allow_pixel_duplicates=True, max_pixel_duplicate_ratio=0.25)
        # With 20% overlap, this should pass if threshold is 25%
        assert result == True

        # Should fail with strict mode
        result_strict = verify_split_separation(data, allow_pixel_duplicates=False)
        assert result_strict == False

    def test_multiple_copies_same_image(self):
        """Test that multiple copies of the same image are counted correctly."""
        # Create an image that appears multiple times in each split
        duplicate_img = np.ones((28, 28, 1), dtype=np.uint8) * 128

        # Train: 3 copies of the duplicate + 7 unique
        train_images = np.concatenate([
            np.stack([duplicate_img] * 3),
            np.random.randint(0, 255, (7, 28, 28, 1), dtype=np.uint8)
        ], axis=0)
        train_labels = np.zeros(10, dtype=np.int64)

        # Val: 2 copies of the duplicate + 3 unique
        val_images = np.concatenate([
            np.stack([duplicate_img] * 2),
            np.random.randint(0, 255, (3, 28, 28, 1), dtype=np.uint8)
        ], axis=0)
        val_labels = np.zeros(5, dtype=np.int64)

        # Test: completely unique (no duplicates)
        test_images = np.random.randint(0, 255, (5, 28, 28, 1), dtype=np.uint8)
        test_labels = np.zeros(5, dtype=np.int64)

        data = {
            'train': (train_images, train_labels),
            'val': (val_images, val_labels),
            'test': (test_images, test_labels)
        }

        # Should detect min(3,2) = 2 overlapping samples
        # With large enough threshold, should pass
        result = verify_split_separation(data, allow_pixel_duplicates=True, max_pixel_duplicate_ratio=0.25)
        assert result == True  # 2/10 = 20% < 25%

    def test_different_images_no_overlap(self):
        """Test that completely different images pass validation."""
        # Create completely unique images for each split
        np.random.seed(42)
        train_images = np.random.randint(0, 255, (10, 28, 28, 1), dtype=np.uint8)
        train_labels = np.zeros(10, dtype=np.int64)

        np.random.seed(43)
        val_images = np.random.randint(0, 255, (5, 28, 28, 1), dtype=np.uint8)
        val_labels = np.zeros(5, dtype=np.int64)

        np.random.seed(44)
        test_images = np.random.randint(0, 255, (5, 28, 28, 1), dtype=np.uint8)
        test_labels = np.zeros(5, dtype=np.int64)

        data = {
            'train': (train_images, train_labels),
            'val': (val_images, val_labels),
            'test': (test_images, test_labels)
        }

        # Should pass even with strict mode
        result = verify_split_separation(data, allow_pixel_duplicates=False)
        assert result == True

    def test_train_test_overlap_rejected(self):
        """Test that ANY train-test overlap is rejected (zero tolerance)."""
        # Create data where train and test share 1 image
        shared_img = np.ones((28, 28, 1), dtype=np.uint8) * 200

        train_images = np.concatenate([
            np.stack([shared_img]),
            np.random.randint(0, 255, (9, 28, 28, 1), dtype=np.uint8)
        ], axis=0)
        train_labels = np.zeros(10, dtype=np.int64)

        val_images = np.random.randint(0, 255, (5, 28, 28, 1), dtype=np.uint8)
        val_labels = np.zeros(5, dtype=np.int64)

        test_images = np.concatenate([
            np.stack([shared_img]),  # Same image in test!
            np.random.randint(0, 255, (4, 28, 28, 1), dtype=np.uint8)
        ], axis=0)
        test_labels = np.zeros(5, dtype=np.int64)

        data = {
            'train': (train_images, train_labels),
            'val': (val_images, val_labels),
            'test': (test_images, test_labels)
        }

        # Should ALWAYS fail, regardless of tolerance settings
        result_permissive = verify_split_separation(data, allow_pixel_duplicates=True, max_pixel_duplicate_ratio=1.0)
        assert result_permissive == False

        result_strict = verify_split_separation(data, allow_pixel_duplicates=False)
        assert result_strict == False

    def test_configurable_tolerance(self):
        """Test that train-val tolerance is configurable."""
        # Create data with exactly 10% overlap
        duplicate_img = np.ones((28, 28, 1), dtype=np.uint8) * 150

        train_images = np.concatenate([
            np.stack([duplicate_img]),  # 1 duplicate
            np.random.randint(0, 255, (9, 28, 28, 1), dtype=np.uint8)  # 9 unique
        ], axis=0)
        train_labels = np.zeros(10, dtype=np.int64)

        val_images = np.concatenate([
            np.stack([duplicate_img]),  # Same image
            np.random.randint(0, 255, (4, 28, 28, 1), dtype=np.uint8)
        ], axis=0)
        val_labels = np.zeros(5, dtype=np.int64)

        test_images = np.random.randint(0, 255, (5, 28, 28, 1), dtype=np.uint8)
        test_labels = np.zeros(5, dtype=np.int64)

        data = {
            'train': (train_images, train_labels),
            'val': (val_images, val_labels),
            'test': (test_images, test_labels)
        }

        # 1/10 = 10% overlap

        # Should fail with 5% threshold
        result_low = verify_split_separation(data, allow_pixel_duplicates=True, max_pixel_duplicate_ratio=0.05)
        assert result_low == False

        # Should pass with 15% threshold
        result_high = verify_split_separation(data, allow_pixel_duplicates=True, max_pixel_duplicate_ratio=0.15)
        assert result_high == True

        # Should fail with strict mode
        result_strict = verify_split_separation(data, allow_pixel_duplicates=False)
        assert result_strict == False

    def test_sha256_hash_stability(self):
        """Test that SHA-256 hashing produces stable results."""
        import hashlib

        # Create a known image
        img = np.arange(28*28, dtype=np.uint8).reshape(28, 28, 1)

        # Hash it multiple times
        hash1 = hashlib.sha256(img.tobytes()).hexdigest()
        hash2 = hashlib.sha256(img.tobytes()).hexdigest()

        # Should be identical
        assert hash1 == hash2

        # Different image should have different hash
        img2 = img.copy()
        img2[0, 0, 0] = 255  # Change one pixel
        hash3 = hashlib.sha256(img2.tobytes()).hexdigest()

        assert hash1 != hash3


class TestInputValidation:
    """Tests for input validation in dataset functions."""

    def test_empty_images_rejected(self):
        """Test that empty image arrays are rejected."""
        empty_images = np.array([]).reshape(0, 28, 28, 1)
        labels = np.array([])

        with pytest.raises(ValueError, match="Image array is empty"):
            get_dataset_statistics(empty_images, labels)

    def test_empty_labels_rejected(self):
        """Test that empty label arrays are rejected."""
        images = np.random.randint(0, 255, (5, 28, 28, 1), dtype=np.uint8)
        empty_labels = np.array([])

        with pytest.raises(ValueError, match="Label array is empty"):
            get_dataset_statistics(images, empty_labels)

    def test_mismatched_counts_rejected(self):
        """Test that mismatched image/label counts are rejected."""
        images = np.random.randint(0, 255, (5, 28, 28, 1), dtype=np.uint8)
        labels = np.zeros(3, dtype=np.int64)  # Only 3 labels for 5 images

        with pytest.raises(ValueError, match="Image count .* does not match label count"):
            get_dataset_statistics(images, labels)

    def test_invalid_image_dimensions_rejected(self):
        """Test that images with invalid dimensions are rejected."""
        # 2D array (missing height/width dimensions)
        images_2d = np.random.randint(0, 255, (5, 28), dtype=np.uint8)
        labels = np.zeros(5, dtype=np.int64)

        with pytest.raises(ValueError, match="Images must be 3D .* or 4D"):
            get_dataset_statistics(images_2d, labels)

        # 5D array (too many dimensions)
        images_5d = np.random.randint(0, 255, (5, 28, 28, 1, 1), dtype=np.uint8)

        with pytest.raises(ValueError, match="Images must be 3D .* or 4D"):
            get_dataset_statistics(images_5d, labels)

    def test_invalid_duplicate_ratio_rejected(self):
        """Test that invalid max_pixel_duplicate_ratio values are rejected."""
        data = load_full_dataset('breastmnist', download=True)

        # Negative ratio
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            verify_split_separation(data, max_pixel_duplicate_ratio=-0.1)

        # Ratio > 1
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            verify_split_separation(data, max_pixel_duplicate_ratio=1.5)

    def test_invalid_n_samples_rejected(self):
        """Test that invalid n_samples values are rejected."""
        # Zero samples
        with pytest.raises(ValueError, match="n_samples must be positive"):
            demonstrate_reproducibility('breastmnist', n_samples=0)

        # Negative samples
        with pytest.raises(ValueError, match="n_samples must be positive"):
            demonstrate_reproducibility('breastmnist', n_samples=-5)

        # More samples than available
        with pytest.raises(ValueError, match="exceeds available samples"):
            demonstrate_reproducibility('breastmnist', n_samples=10000)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
