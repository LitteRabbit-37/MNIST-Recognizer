import numpy as np
import pickle
import os


class MLP:
    """
    Multi-Layer Perceptron (Neural Network) built from scratch with NumPy.

    Default architecture: 784 -> 128 -> 64 -> 10
    - 784 inputs (28x28 pixels)
    - 2 hidden layers (128 and 64 neurons) with ReLU activation
    - 10 outputs (digits 0-9) with Softmax activation
    """

    def __init__(self, input_size=784, hidden_sizes=[128, 64], output_size=10):
        """
        Initialize the neural network.

        Args:
            input_size: Number of input features (784 for 28x28 images)
            hidden_sizes: List of hidden layer sizes [128, 64]
            output_size: Number of classes (10 for digits 0-9)
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        # Build architecture: [784, 128, 64, 10]
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]

        # Initialize weights and biases
        self.weights = []
        self.biases = []

        # He initialization for ReLU (better convergence)
        for i in range(len(self.layer_sizes) - 1):
            # He: std = sqrt(2 / n_in) to avoid vanishing/exploding gradients
            w = np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * np.sqrt(
                2.0 / self.layer_sizes[i]
            )
            b = np.zeros((1, self.layer_sizes[i + 1]))

            self.weights.append(w)
            self.biases.append(b)

    def relu(self, z):
        """
        ReLU activation function (Rectified Linear Unit).
        ReLU(z) = max(0, z)

        Simple, fast, avoids vanishing gradient.
        """
        return np.maximum(0, z)

    def relu_derivative(self, z):
        """
        Derivative of ReLU for backpropagation.
        ReLU'(z) = 1 if z > 0, else 0
        """
        return (z > 0).astype(float)

    def softmax(self, z):
        """
        Softmax activation function for output layer.
        Converts scores to probabilities that sum to 1.

        Softmax(z_i) = exp(z_i) / sum(exp(z_j))

        Uses numerical stability trick: subtract max
        """
        # Numerical stability: avoid overflow with large values
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, x):
        """
        Forward propagation (forward pass).
        Compute activations for each layer.

        Args:
            x: Input data, shape (batch_size, 784)

        Returns:
            activations: List of activations for each layer
            z_values: List of pre-activation values (for backprop)
        """
        activations = [x]
        z_values = []

        current_activation = x

        # Propagate through hidden layers (ReLU)
        for i in range(len(self.weights) - 1):
            z = np.dot(current_activation, self.weights[i]) + self.biases[i]
            z_values.append(z)
            current_activation = self.relu(z)
            activations.append(current_activation)

        # Last layer (Softmax)
        z = np.dot(current_activation, self.weights[-1]) + self.biases[-1]
        z_values.append(z)
        current_activation = self.softmax(z)
        activations.append(current_activation)

        return activations, z_values

    def backward(self, x, y, activations, z_values, learning_rate=0.01):
        """
        Backpropagation.
        Compute gradients and update weights.

        Args:
            x: Input data
            y: True labels (one-hot encoded)
            activations: Activations from forward pass
            z_values: Pre-activation values
            learning_rate: Learning rate
        """
        m = x.shape[0]  # Number of examples in batch

        # Gradient of last layer (Softmax + Cross-Entropy)
        # Simplified derivative: a - y
        delta = activations[-1] - y

        # Update weights and biases (starting from the end)
        for i in range(len(self.weights) - 1, -1, -1):
            # Gradients
            dw = np.dot(activations[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m

            # Update (gradient descent)
            self.weights[i] -= learning_rate * dw
            self.biases[i] -= learning_rate * db

            # Propagate delta to previous layer (except for first)
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(
                    z_values[i - 1]
                )

    def train_step(self, x, y, learning_rate=0.01):
        """
        One complete training step (forward + backward).

        Args:
            x: Input data, shape (batch_size, 784)
            y: True labels (one-hot encoded), shape (batch_size, 10)
            learning_rate: Learning rate

        Returns:
            loss: Cross-entropy loss
        """
        # Forward pass
        activations, z_values = self.forward(x)

        # Compute loss (Cross-Entropy)
        # Loss = -sum(y * log(predictions))
        predictions = activations[-1]
        # Avoid log(0) with epsilon
        loss = -np.mean(np.sum(y * np.log(predictions + 1e-8), axis=1))

        # Backward pass
        self.backward(x, y, activations, z_values, learning_rate)

        return loss

    def predict(self, x):
        """
        Prediction for new data.

        Args:
            x: Input data, shape (batch_size, 784) or (784,)

        Returns:
            predictions: Predicted class indices
        """
        # Handle single example
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # Forward pass (keep only final activations)
        activations, _ = self.forward(x)

        # Return index of class with highest probability
        return np.argmax(activations[-1], axis=1)

    def predict_proba(self, x):
        """
        Return probabilities for each class.

        Args:
            x: Input data, shape (batch_size, 784) or (784,)

        Returns:
            probabilities: Probabilities for each class, shape (batch_size, 10)
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        activations, _ = self.forward(x)
        return activations[-1]

    def evaluate(self, x, y):
        """
        Evaluate model accuracy.

        Args:
            x: Input data
            y: True labels (indices or one-hot)

        Returns:
            accuracy: Accuracy (0-1)
        """
        predictions = self.predict(x)

        # If y is one-hot encoded, convert to indices
        if y.ndim > 1:
            y = np.argmax(y, axis=1)

        accuracy = np.mean(predictions == y)
        return accuracy

    def save(self, path):
        """
        Save model weights.

        Args:
            path: File path (e.g., 'saved_model/weights.pkl')
        """
        # Create directory if necessary
        os.makedirs(os.path.dirname(path), exist_ok=True)

        model_data = {
            "weights": self.weights,
            "biases": self.biases,
            "layer_sizes": self.layer_sizes,
            "input_size": self.input_size,
            "hidden_sizes": self.hidden_sizes,
            "output_size": self.output_size,
        }

        with open(path, "wb") as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {path}")

    def load(self, path):
        """
        Load model weights.

        Args:
            path: File path
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} does not exist")

        with open(path, "rb") as f:
            model_data = pickle.load(f)

        self.weights = model_data["weights"]
        self.biases = model_data["biases"]
        self.layer_sizes = model_data["layer_sizes"]
        self.input_size = model_data["input_size"]
        self.hidden_sizes = model_data["hidden_sizes"]
        self.output_size = model_data["output_size"]

        print(f"Model loaded from {path}")


# Helper function to create one-hot labels
def one_hot_encode(labels, num_classes=10):
    """
    Convert labels to one-hot encoding.

    Example: 3 -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

    Args:
        labels: Array of labels (indices)
        num_classes: Number of classes

    Returns:
        one_hot: One-hot encoded array
    """
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot
