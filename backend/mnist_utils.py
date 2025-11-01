"""
Utilities for loading the MNIST dataset via HuggingFace datasets.
"""

from datasets import load_dataset
import numpy as np


def load_mnist_data(use_cache=True):
    """
    Load MNIST dataset (train + test) from HuggingFace.

    Args:
        use_cache: Use local cache if available

    Returns:
        X_train: Training images, shape (60000, 28, 28)
        y_train: Training labels, shape (60000,)
        X_test: Test images, shape (10000, 28, 28)
        y_test: Test labels, shape (10000,)
    """
    print("   Downloading MNIST dataset from HuggingFace...")

    # Load train set
    ds_train = load_dataset("ylecun/mnist", split="train", trust_remote_code=True)
    X_train = np.stack([np.array(im) for im in ds_train["image"]])  # (60000, 28, 28)
    y_train = np.array(ds_train["label"])  # (60000,)

    # Load test set
    ds_test = load_dataset("ylecun/mnist", split="test", trust_remote_code=True)
    X_test = np.stack([np.array(im) for im in ds_test["image"]])  # (10000, 28, 28)
    y_test = np.array(ds_test["label"])  # (10000,)

    print("   Dataset loaded successfully")

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    # Test loading
    print("Testing MNIST loading...")
    X_train, y_train, X_test, y_test = load_mnist_data()

    print(f"\nShapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_test: {y_test.shape}")

    print(f"\nValue range: [{X_train.min()}, {X_train.max()}]")
    print(f"Unique labels: {np.unique(y_train)}")
