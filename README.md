# MNIST Handwritten Digit Recognizer

A complete implementation of a Multi-Layer Perceptron (MLP) built from scratch using only NumPy to recognize handwritten digits from the MNIST dataset. This project includes a REST API backend and an interactive web frontend.

## Overview

This project demonstrates a deep understanding of neural networks by implementing all core components from scratch without using high-level frameworks like PyTorch or TensorFlow. The system achieves approximately 97-98% accuracy on the MNIST test set.

**Key Features:**

- Neural network implementation using only NumPy
- Optimized architecture: 784 → 128 → 64 → 10 (~109K parameters)
- FastAPI REST API for predictions
- Interactive web interface with 28×28 drawing canvas
- Side-by-side responsive design
- Real-time digit recognition

## Project Structure

```
mnist_recognizer/
├── backend/
│   ├── main.py              # FastAPI REST API
│   ├── model.py             # MLP implementation (NumPy only)
│   ├── train.py             # Training script
│   ├── mnist_utils.py       # Dataset loading utilities
│   ├── utils.py             # Helper functions
│   ├── requirements.txt     # Python dependencies
│   └── saved_model/         # Trained model weights
│
├── frontend/
│   ├── index.html           # Web interface
│   ├── css/style.css        # Styles
│   └── js/script.js         # Drawing logic and API calls
│
└── README.md
```

## Neural Network Architecture

### Model Details

The MLP uses a simple yet effective architecture:

```
Input Layer:     784 neurons  (28×28 pixels flattened)
                   ↓
Hidden Layer 1:  128 neurons  (ReLU activation)
                   ↓
Hidden Layer 2:   64 neurons  (ReLU activation)
                   ↓
Output Layer:     10 neurons  (Softmax activation, classes 0-9)
```

**Key Techniques:**

- **He Initialization**: Prevents vanishing/exploding gradients
- **ReLU Activation**: Fast and effective for hidden layers
- **Softmax Output**: Converts scores to probabilities
- **Cross-Entropy Loss**: Optimal for classification
- **Mini-batch Gradient Descent**: Efficient training with batch size 128
- **Backpropagation**: Automatic gradient computation

### Performance

- **Model Size**: < 1 MB
- **Parameters**: 109,386
- **Training**: 500 epochs (~30-45 minutes)
- **Accuracy**: 97-98% on MNIST test set
- **Inference**: < 10ms per prediction on CPU

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd mnist_recognizer
```

2. Install dependencies:

```bash
cd backend
pip install -r requirements.txt
```

## Usage

### 1. Train the Model

Before first use, train the model on MNIST:

```bash
cd backend
python train.py
```

This will:

- Download the MNIST dataset from HuggingFace (automatic)
- Train for 500 epochs (configurable in `train.py`)
- Save the best model to `saved_model/weights.pkl`
- Display training progress with loss and accuracy metrics

**Training configuration** (editable in `train.py`):

```python
config = {
    "epochs": 500,
    "batch_size": 128,
    "learning_rate": 0.01,
    "save_path": "saved_model/weights.pkl"
}
```

### 2. Start the Backend API

```bash
cd backend
python main.py
```

The API will be available at `http://localhost:8000`

**Available endpoints:**

- `POST /predict` - Predict a digit from 784 pixels
- `GET /health` - Check API and model status
- `GET /model/info` - Get model architecture details
- `GET /docs` - Interactive API documentation (Swagger)

### 3. Start the Frontend

**Option A: Simple HTTP server (recommended)**

```bash
cd frontend
python -m http.server 3000
```

**Option B: Open directly**

```bash
open frontend/index.html
```

Access the application at `http://localhost:3000`

### 4. Using the Application

1. Draw a digit (0-9) on the canvas using your mouse or touchscreen
   - **Left click + drag**: Draw with smooth Gaussian brush
   - The drawing is automatically centered (like MNIST preprocessing)
2. Click the **"Predict"** button to get a prediction
3. View results:
   - Predicted digit with confidence
   - Probability bars for all digits (0-9)
4. Click **"Clear"** to reset and draw again

## Technical Implementation

### Backend (Python + NumPy)

#### model.py - MLP Implementation

The `MLP` class implements a complete neural network from scratch:

**Forward Propagation:**

```python
z = x · W + b
a = activation(z)
```

**Backpropagation:**

- Computes gradients using the chain rule
- Updates weights via gradient descent
- Simplified derivative for Softmax + Cross-Entropy: `δ = ŷ - y`

**Key Methods:**

- `forward(x)`: Compute predictions
- `backward(...)`: Calculate and apply gradients
- `train_step(x, y, lr)`: One complete training iteration
- `predict(x)`: Return predicted class
- `predict_proba(x)`: Return class probabilities
- `save(path)` / `load(path)`: Model persistence

#### Mathematical Details

**ReLU Activation:**

```
f(x) = max(0, x)
f'(x) = 1 if x > 0, else 0
```

**Softmax Activation:**

```
σ(z)ᵢ = exp(zᵢ) / Σⱼ exp(zⱼ)
```

**Cross-Entropy Loss:**

```
L = -Σᵢ yᵢ · log(ŷᵢ)
```

**Weight Update:**

```
W = W - η · ∂L/∂W
```

### Frontend (HTML/CSS/JavaScript)

#### Drawing Canvas

- **Canvas**: HTML5 Canvas (280×280px) representing 28×28 grid
- **Gaussian Brush**: Smooth drawing with intensity falloff (σ=0.5)
- **Auto-centering**: Images are centered using center of mass (like MNIST)
- **Data Storage**: Grayscale values 0-255 in grid array
- **API Format**: Normalized to [0, 1] before sending

#### Preprocessing Pipeline

The frontend applies MNIST-style preprocessing:

```javascript
// 1. Calculate center of mass of the drawing
// 2. Shift image to center the digit
// 3. Normalize pixel values to [0, 1]
// 4. Flatten to 784-element array (row-major order)
```

## API Reference

### POST /predict

**Request:**

```json
{
  "pixels": [0.0, 0.1, ..., 0.9]  // 784 values, range [0, 1]
}
```

**Response:**

```json
{
  "prediction": 7,
  "probabilities": {
    "0": 0.001, "1": 0.002, ..., "7": 0.982, ..., "9": 0.003
  },
  "confidence": 0.982
}
```

### GET /health

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_architecture": "784 -> 128 -> 64 -> 10"
}
```

## Development

### Running in Development Mode

**Backend with auto-reload:**

```bash
cd backend
uvicorn main:app --reload --port 8000
```

**Frontend:**

```bash
cd frontend
python -m http.server 3000
```

### Modifying the Model

To experiment with different architectures, edit `train.py`:

```python
# Example: Add more hidden layers
model = MLP(
    input_size=784,
    hidden_sizes=[256, 128, 64],  # 3 hidden layers
    output_size=10
)
```

### Training Parameters

Adjust training in `train.py`:

```python
config = {
    "epochs": 500,          # More epochs = better accuracy (diminishing returns)
    "batch_size": 128,      # Larger = faster but less stable
    "learning_rate": 0.01,  # Higher = faster but may overshoot
}
```

## Project Highlights

### Why This Project Stands Out

1. **Built from Scratch**: No PyTorch, TensorFlow, or Keras - pure NumPy implementation
2. **Educational**: Clear, commented code explaining each concept
3. **Complete System**: Not just a model, but a full application with API and UI

### Learning Outcomes

By studying this project, you will understand:

- How neural networks work at a fundamental level
- Forward and backward propagation mathematics
- Gradient descent optimization
- REST API design with FastAPI

## Performance Optimization

The model is optimized for:

**Size:**

- Architecture 784→128→64→10 vs 784→512→256→10 saves ~440K parameters
- Model file < 1 MB vs 2-5 MB for typical MNIST CNNs

**Speed:**

- NumPy vectorization for fast matrix operations
- Mini-batch processing
- CPU inference < 10ms

**Accuracy:**

- 500 epochs ensures convergence
- He initialization for stable training
- Appropriate learning rate (0.01)

## Troubleshooting

**"Model not found" error:**

```bash
cd backend
python train.py  # Train the model first
```

**"Connection refused" in frontend:**

- Ensure backend is running on port 8000
- Check CORS settings in `main.py`

**Poor predictions:**

- Retrain with more epochs (edit `train.py`)
- Ensure proper 28×28 grid (frontend CSS Grid)
- Check that pixels are normalized [0, 1]

**ModuleNotFoundError:**

```bash
cd backend
pip install -r requirements.txt
```

## Future Improvements

Potential enhancements:

- Add data augmentation (rotation, scaling) during training
- Implement dropout for regularization
- Add momentum or Adam optimizer
- Add batch prediction endpoint
- Add model comparison features

## Technologies Used

**Backend:**

- Python 3.12
- NumPy (matrix operations)
- FastAPI (REST API)
- Uvicorn (ASGI server)
- HuggingFace Datasets (MNIST loading)
- tqdm (progress bars)

**Frontend:**

- HTML5
- CSS3 (Grid, Flexbox, CSS variables)
- Vanilla JavaScript (no frameworks)

## License

MIT License - Feel free to use this project for learning and development.

## Acknowledgments

- MNIST dataset by Yann LeCun
- FastAPI framework by Sebastián Ramírez
- Inspired by the goal of understanding neural networks from first principles

---

**Built with Python and NumPy - No deep learning frameworks required.**
