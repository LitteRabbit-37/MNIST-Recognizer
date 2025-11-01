"""
Training script for the MLP model on the MNIST dataset.

Usage:
    python train.py
"""

import numpy as np
from model import MLP, one_hot_encode
from mnist_utils import load_mnist_data
from tqdm import tqdm
import time


def train_model(
    epochs=10, batch_size=128, learning_rate=0.01, save_path="saved_model/weights.pkl"
):
    """
    Train the MLP model on MNIST.

    Args:
        epochs: Number of training epochs
        batch_size: Mini-batch size
        learning_rate: Learning rate
        save_path: Path to save the model
    """
    print("=" * 60)
    print("TRAINING MLP MODEL ON MNIST")
    print("=" * 60)

    # Load data
    print("\nLoading MNIST dataset...")
    X_train, y_train, X_test, y_test = load_mnist_data()

    print(f"   - Train set: {X_train.shape[0]} examples")
    print(f"   - Test set:  {X_test.shape[0]} examples")
    print(f"   - Input shape: {X_train.shape[1:]}")

    # Normalize pixels [0, 255] -> [0, 1]
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    # Flatten: (60000, 28, 28) -> (60000, 784)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # One-hot encode labels
    y_train_onehot = one_hot_encode(y_train)
    y_test_onehot = one_hot_encode(y_test)

    print(f"   - Final shape: {X_train.shape}")

    # Create model
    print("\nCreating MLP model...")
    model = MLP(input_size=784, hidden_sizes=[128, 64], output_size=10)

    # Calculate total parameters
    total_params = sum(w.size for w in model.weights) + sum(
        b.size for b in model.biases
    )
    print(f"   - Architecture: {' -> '.join(map(str, model.layer_sizes))}")
    print(f"   - Total parameters: {total_params:,}")

    # Training
    print(
        f"\nTraining ({epochs} epochs, batch_size={batch_size}, lr={learning_rate})..."
    )
    print("-" * 60)

    num_batches = len(X_train) // batch_size
    best_accuracy = 0.0

    for epoch in range(epochs):
        start_time = time.time()

        # Shuffle data at each epoch
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train_onehot[indices]

        epoch_losses = []

        # Mini-batch gradient descent with progress bar
        with tqdm(
            total=num_batches, desc=f"Epoch {epoch + 1}/{epochs}", ncols=80
        ) as pbar:
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size

                X_batch = X_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]

                # One training step
                loss = model.train_step(X_batch, y_batch, learning_rate)
                epoch_losses.append(loss)

                pbar.update(1)
                pbar.set_postfix({"loss": f"{loss:.4f}"})

        # Evaluate on train and test set
        train_accuracy = model.evaluate(X_train, y_train)
        test_accuracy = model.evaluate(X_test, y_test)
        avg_loss = np.mean(epoch_losses)
        epoch_time = time.time() - start_time

        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"Loss: {avg_loss:.4f} - "
            f"Train Acc: {train_accuracy:.4f} - "
            f"Test Acc: {test_accuracy:.4f} - "
            f"Time: {epoch_time:.1f}s"
        )

        # Save best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            model.save(save_path)
            print(f"   New best model saved (acc: {best_accuracy:.4f})")

    print("-" * 60)
    print(f"\nTraining completed!")
    print(f"   - Best accuracy: {best_accuracy:.4f} ({best_accuracy * 100:.2f}%)")
    print(f"   - Model saved: {save_path}")

    # Final test with a few predictions
    print("\nTesting a few predictions:")
    test_samples = 5
    sample_indices = np.random.choice(len(X_test), test_samples, replace=False)

    for idx in sample_indices:
        x_sample = X_test[idx]
        true_label = y_test[idx]

        # Prediction with probabilities
        pred_label = model.predict(x_sample)[0]
        probas = model.predict_proba(x_sample)[0]
        confidence = probas[pred_label]

        status = "✓" if pred_label == true_label else "✗"
        print(
            f"   {status} True: {true_label} | Predicted: {pred_label} | Confidence: {confidence:.2%}"
        )

    return model


if __name__ == "__main__":
    # Training configuration
    config = {
        "epochs": 500,  # More epochs = better learning
        "batch_size": 128,  # Speed/stability trade-off
        "learning_rate": 0.01,  # Standard learning rate
        "save_path": "saved_model/weights.pkl",
    }

    print("\nConfiguration:")
    for key, value in config.items():
        print(f"   - {key}: {value}")

    # Start training
    trained_model = train_model(**config)

    print("\nThe model is ready to be used in the application!")
