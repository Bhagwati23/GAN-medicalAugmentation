"""
Dataset loading and preparation utilities for MedMNIST datasets.

This module provides functions to load, explore, and prepare BreastMNIST and PneumoniaMNIST
datasets while maintaining strict separation of train/validation/test splits to prevent
data leakage.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
import medmnist
from medmnist import INFO


def get_medmnist_info(dataset_name: str) -> Dict[str, Any]:
    """
    Get metadata information for a MedMNIST dataset.

    Args:
        dataset_name: Name of the dataset (e.g., 'breastmnist', 'pneumoniamnist')

    Returns:
        Dictionary containing dataset metadata
    """
    if dataset_name.lower() not in INFO:
        raise ValueError(f"Dataset '{dataset_name}' not found in MedMNIST INFO")

    return INFO[dataset_name.lower()]


def load_breast_mnist(
    split: str = 'train',
    download: bool = True,
    root: str = './data',
    as_rgb: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load BreastMNIST dataset.

    Args:
        split: Data split to load ('train', 'val', or 'test')
        download: Whether to download the dataset if not present
        root: Root directory to store the dataset
        as_rgb: Whether to convert grayscale images to RGB

    Returns:
        Tuple of (images, labels) as numpy arrays
        - images: shape (N, H, W, C) where C=1 for grayscale or C=3 for RGB
        - labels: shape (N,) with dtype int64
    """
    from medmnist import BreastMNIST

    dataset = BreastMNIST(
        split=split,
        download=download,
        root=root,
        as_rgb=as_rgb
    )

    # Extract images and labels
    images = dataset.imgs
    labels = dataset.labels.flatten().astype(np.int64)

    # Ensure images have 4 dimensions (N, H, W, C)
    if len(images.shape) == 3:
        # Grayscale images without channel dimension - add it
        images = np.expand_dims(images, axis=-1)

    return images, labels


def load_pneumonia_mnist(
    split: str = 'train',
    download: bool = True,
    root: str = './data',
    as_rgb: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load PneumoniaMNIST dataset.

    Args:
        split: Data split to load ('train', 'val', or 'test')
        download: Whether to download the dataset if not present
        root: Root directory to store the dataset
        as_rgb: Whether to convert grayscale images to RGB

    Returns:
        Tuple of (images, labels) as numpy arrays
        - images: shape (N, H, W, C) where C=1 for grayscale or C=3 for RGB
        - labels: shape (N,) with dtype int64
    """
    from medmnist import PneumoniaMNIST

    dataset = PneumoniaMNIST(
        split=split,
        download=download,
        root=root,
        as_rgb=as_rgb
    )

    # Extract images and labels
    images = dataset.imgs
    labels = dataset.labels.flatten().astype(np.int64)

    # Ensure images have 4 dimensions (N, H, W, C)
    if len(images.shape) == 3:
        # Grayscale images without channel dimension - add it
        images = np.expand_dims(images, axis=-1)

    return images, labels


def get_dataset_statistics(
    images: np.ndarray,
    labels: np.ndarray
) -> Dict[str, Any]:
    """
    Calculate statistics for a dataset split.

    Args:
        images: Image array of shape (N, H, W, C) or (N, H, W)
        labels: Label array of shape (N,)

    Returns:
        Dictionary containing dataset statistics

    Raises:
        ValueError: If images or labels are empty, if their counts don't match,
                   or if images have invalid dimensions
    """
    # Validate inputs
    if images.size == 0:
        raise ValueError("Image array is empty")
    if labels.size == 0:
        raise ValueError("Label array is empty")
    if len(images) != len(labels):
        raise ValueError(f"Image count ({len(images)}) does not match label count ({len(labels)})")

    # Validate image dimensions
    if images.ndim < 3 or images.ndim > 4:
        raise ValueError(f"Images must be 3D (N, H, W) or 4D (N, H, W, C), got shape {images.shape}")

    unique_labels, counts = np.unique(labels, return_counts=True)

    stats = {
        'num_samples': len(images),
        'image_shape': images.shape[1:],
        'num_channels': images.shape[-1] if len(images.shape) == 4 else 1,
        'dtype': str(images.dtype),
        'pixel_min': images.min(),
        'pixel_max': images.max(),
        'pixel_mean': images.mean(),
        'pixel_std': images.std(),
        'num_classes': len(unique_labels),
        'class_labels': unique_labels.tolist(),
        'class_counts': counts.tolist(),
        'class_distribution': dict(zip(unique_labels.tolist(), counts.tolist()))
    }

    return stats


def load_full_dataset(
    dataset_name: str,
    download: bool = True,
    root: str = './data',
    as_rgb: bool = False
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Load all splits (train, val, test) of a MedMNIST dataset.

    IMPORTANT: This function returns all three splits separately.
    Test data must NEVER be used for training or augmentation.

    Args:
        dataset_name: Name of the dataset ('breastmnist' or 'pneumoniamnist')
        download: Whether to download the dataset if not present
        root: Root directory to store the dataset
        as_rgb: Whether to convert grayscale images to RGB

    Returns:
        Dictionary with keys 'train', 'val', 'test', each containing
        a tuple of (images, labels)
    """
    dataset_name = dataset_name.lower()

    if dataset_name == 'breastmnist':
        load_func = load_breast_mnist
    elif dataset_name == 'pneumoniamnist':
        load_func = load_pneumonia_mnist
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    data = {}
    for split in ['train', 'val', 'test']:
        images, labels = load_func(
            split=split,
            download=download,
            root=root,
            as_rgb=as_rgb
        )
        data[split] = (images, labels)

    return data


def get_training_data_only(
    dataset_name: str,
    download: bool = True,
    root: str = './data',
    as_rgb: bool = False,
    include_val: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load ONLY the training data for GAN training and augmentation.

    This function explicitly excludes test data to prevent data leakage.
    Use this function when preparing data for GAN training.

    Args:
        dataset_name: Name of the dataset ('breastmnist' or 'pneumoniamnist')
        download: Whether to download the dataset if not present
        root: Root directory to store the dataset
        as_rgb: Whether to convert grayscale images to RGB
        include_val: Whether to include validation data with training data

    Returns:
        Tuple of (images, labels) containing only training (and optionally validation) data
    """
    dataset_name = dataset_name.lower()

    if dataset_name == 'breastmnist':
        load_func = load_breast_mnist
    elif dataset_name == 'pneumoniamnist':
        load_func = load_pneumonia_mnist
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Load training data
    train_images, train_labels = load_func(
        split='train',
        download=download,
        root=root,
        as_rgb=as_rgb
    )

    if include_val:
        # Optionally include validation data
        val_images, val_labels = load_func(
            split='val',
            download=download,
            root=root,
            as_rgb=as_rgb
        )
        images = np.concatenate([train_images, val_images], axis=0)
        labels = np.concatenate([train_labels, val_labels], axis=0)
    else:
        images = train_images
        labels = train_labels

    return images, labels


def verify_split_separation(
    data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    allow_pixel_duplicates: bool = True,
    max_pixel_duplicate_ratio: float = 0.01
) -> bool:
    """
    Verify that train/val/test splits don't have problematic pixel-level overlap.

    This function detects EXACT PIXEL-LEVEL DUPLICATE IMAGES between splits using
    SHA-256 hashing. It counts duplicates at the sample level (not unique image level).

    IMPORTANT DISTINCTIONS:
    - Pixel-level duplicates: Images with identical pixel values (what this detects)
    - Sample-level leakage: Same data sample in multiple splits (cannot verify without IDs)
    - Patient-level leakage: Same patient in multiple splits (cannot verify without patient IDs)

    Without sample or patient identifiers in the dataset, we CANNOT distinguish between:
    - Same scan of the same patient appearing in multiple splits (true leakage)
    - Different scans of different patients with identical pixel patterns (coincidental)
    - Duplicate copies of the same image within a split

    This function detects and counts pixel-level duplicates. Passing this check does NOT
    prove patient-level or sample-level separation - it only verifies pixel-level separation.

    CRITICAL: Train-test pixel-level overlap is STRICTLY FORBIDDEN (zero tolerance).

    Args:
        data: Dictionary containing 'train', 'val', 'test' splits
        allow_pixel_duplicates: If True, allow small numbers of pixel-identical images
                               between train and val only (not train and test)
        max_pixel_duplicate_ratio: Maximum allowed ratio of duplicates (must be between 0 and 1)

    Returns:
        True if splits pass validation, False otherwise

    Raises:
        ValueError: If max_pixel_duplicate_ratio is not in valid range [0, 1]
    """
    import hashlib
    from collections import Counter

    # Validate parameters
    if not 0 <= max_pixel_duplicate_ratio <= 1:
        raise ValueError(f"max_pixel_duplicate_ratio must be between 0 and 1, got {max_pixel_duplicate_ratio}")

    train_images, train_labels = data['train']
    val_images, val_labels = data['val']
    test_images, test_labels = data['test']

    # Validate that splits are not empty
    train_size = len(train_images)
    val_size = len(val_images)
    test_size = len(test_images)

    if train_size == 0 or val_size == 0 or test_size == 0:
        print(f"Error: Empty splits detected - train: {train_size}, val: {val_size}, test: {test_size}")
        print(f"All splits must contain at least one sample.")
        return False

    # Detect pixel-level duplicate images using stable SHA-256 hashing
    def hash_images_with_counts(images):
        """
        Create SHA-256 hash for each image and count occurrences.

        Returns:
            Counter mapping hash -> count of images with that hash
        """
        hash_counts = Counter()
        for img in images:
            # Use SHA-256 for stable, collision-resistant hashing
            img_bytes = img.tobytes()
            img_hash = hashlib.sha256(img_bytes).hexdigest()
            hash_counts[img_hash] += 1
        return hash_counts

    # Get hash counts for each split
    train_hash_counts = hash_images_with_counts(train_images)
    val_hash_counts = hash_images_with_counts(val_images)
    test_hash_counts = hash_images_with_counts(test_images)

    # Find overlapping hashes (pixel-level duplicates across splits)
    train_hashes = set(train_hash_counts.keys())
    val_hashes = set(val_hash_counts.keys())
    test_hashes = set(test_hash_counts.keys())

    train_val_overlapping_hashes = train_hashes & val_hashes
    train_test_overlapping_hashes = train_hashes & test_hashes
    val_test_overlapping_hashes = val_hashes & test_hashes

    # Count total overlapping SAMPLES (not just unique images)
    # If an image appears 3 times in train and 2 times in val, count min(3,2)=2 overlaps
    def count_sample_overlaps(hashes, counts1, counts2):
        """Count minimum occurrences across splits for overlapping hashes."""
        total = 0
        for h in hashes:
            # Minimum count represents confirmed duplicates across splits
            total += min(counts1[h], counts2[h])
        return total

    train_val_overlap = count_sample_overlaps(train_val_overlapping_hashes, train_hash_counts, val_hash_counts)
    train_test_overlap = count_sample_overlaps(train_test_overlapping_hashes, train_hash_counts, test_hash_counts)
    val_test_overlap = count_sample_overlaps(val_test_overlapping_hashes, val_hash_counts, test_hash_counts)

    # Calculate overlap ratios
    train_val_ratio = train_val_overlap / train_size if train_size > 0 else 0
    train_test_ratio = train_test_overlap / train_size if train_size > 0 else 0
    val_test_ratio = val_test_overlap / val_size if val_size > 0 else 0

    # CRITICAL: Train-test pixel-level overlap is NEVER allowed
    if train_test_overlap > 0:
        print(f"CRITICAL ERROR: Train-Test pixel-level overlap detected!")
        print(f"  Overlapping images: {len(train_test_overlapping_hashes)} unique")
        print(f"  Overlapping samples: {train_test_overlap} total ({train_test_ratio*100:.2f}%)")
        print(f"  This represents EXACT pixel-level duplicates between train and test.")
        print(f"  Data leakage detected - investigation required.")
        return False

    # Val-test overlap is also problematic
    if val_test_overlap > 0:
        print(f"ERROR: Val-Test pixel-level overlap detected!")
        print(f"  Overlapping images: {len(val_test_overlapping_hashes)} unique")
        print(f"  Overlapping samples: {val_test_overlap} total ({val_test_ratio*100:.2f}%)")
        print(f"  This represents EXACT pixel-level duplicates between val and test.")
        print(f"  Data leakage detected - investigation required.")
        return False

    # Train-val overlap: check against threshold
    if train_val_overlap > 0:
        if allow_pixel_duplicates and train_val_ratio <= max_pixel_duplicate_ratio:
            print(f"Info: Train-Val pixel-level duplicates detected but within tolerance:")
            print(f"  Overlapping unique images: {len(train_val_overlapping_hashes)}")
            print(f"  Overlapping sample count: {train_val_overlap} ({train_val_ratio*100:.2f}%)")
            print(f"  Tolerance threshold: {max_pixel_duplicate_ratio*100:.1f}%")
            print(f"")
            print(f"  IMPORTANT: These are EXACT pixel-level duplicates.")
            print(f"  Without sample/patient IDs, we CANNOT determine if these represent:")
            print(f"    - Same patient samples in both splits (true data leakage)")
            print(f"    - Different patients with identical images (coincidental duplicates)")
            print(f"    - Blank/uniform images or preprocessing artifacts")
            print(f"")
            print(f"  This function detects pixel-level duplicates only.")
            print(f"  Passing this check does NOT prove patient-level separation.")
            print(f"")
            print(f"  RECOMMENDATION: Use TRAIN SET ONLY for GAN training to avoid any")
            print(f"  potential sample-level or patient-level leakage.")
            return True  # Pass with detailed warning
        else:
            print(f"Warning: Train-Val pixel-level overlap exceeds threshold!")
            print(f"  Overlapping unique images: {len(train_val_overlapping_hashes)}")
            print(f"  Overlapping sample count: {train_val_overlap} ({train_val_ratio*100:.2f}%)")
            print(f"  Tolerance threshold: {max_pixel_duplicate_ratio*100:.1f}%")
            return False

    # No pixel-level overlaps detected
    return True


def set_random_seed(seed: int = 42):
    """
    Set random seed for reproducibility.

    Sets seeds for:
    - Python's random module (via numpy)
    - NumPy random number generator
    - PyTorch (if available)

    Note: This ensures reproducibility of random operations (sampling, shuffling, etc.)
    but does not affect the deterministic loading of dataset splits from MedMNIST.

    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def demonstrate_reproducibility(
    dataset_name: str = 'breastmnist',
    seed: int = 42,
    n_samples: int = 5
) -> Dict[str, Any]:
    """
    Demonstrate that random sampling with the same seed produces identical results.

    Note: This demonstrates reproducibility of random operations (like sampling),
    not reproducibility of the dataset splits themselves. MedMNIST dataset splits
    are deterministic and always load the same official train/val/test splits.

    Args:
        dataset_name: Name of the dataset
        seed: Random seed
        n_samples: Number of samples to randomly extract for comparison (must be > 0
                  and <= available samples)

    Returns:
        Dictionary showing that two runs with the same seed produce identical results

    Raises:
        ValueError: If n_samples is invalid
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    # Load data once for efficiency
    images, labels = get_training_data_only(dataset_name, download=True)

    if n_samples > len(images):
        raise ValueError(f"n_samples ({n_samples}) exceeds available samples ({len(images)})")

    # First run with seed
    set_random_seed(seed)
    indices1 = np.random.choice(len(images), n_samples, replace=False)
    sample1 = images[indices1]
    labels_sample1 = labels[indices1]

    # Second run with same seed
    set_random_seed(seed)
    indices2 = np.random.choice(len(images), n_samples, replace=False)
    sample2 = images[indices2]
    labels_sample2 = labels[indices2]

    # Verify they are identical
    indices_match = np.array_equal(indices1, indices2)
    samples_match = np.array_equal(sample1, sample2)
    labels_match = np.array_equal(labels_sample1, labels_sample2)
    are_identical = indices_match and samples_match and labels_match

    return {
        'identical': are_identical,
        'indices1': indices1,
        'indices2': indices2,
        'indices_match': indices_match,
        'samples_equal': samples_match,
        'labels_equal': labels_match
    }
