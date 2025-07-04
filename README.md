# MNIST Handwritten Digit Recognizer

This project provides an interactive web application that allows users to draw a digit (0–9) and get a real-time prediction using a deep learning model built from scratch (NumPy only).

## Project Structure

```
mnist_recognizer/
│
├── backend/
│   ├── main.py           # FastAPI backend
│   ├── model.py          # NumPy MLP
│   ├── utils.py          # Utility functions
│   ├── mnist_utils.py    # MNIST loading
│   ├── requirements.txt  # Python dependencies
│   └── saved_model/      # Trained model weights
│
├── frontend/
│   ├── css/
│   ├── js/
│   └── index.html        # Interactive drawing grid
│
├── README.md
└── .gitignore
```

## Main Features
- Draw a digit in the browser (28x28 grid)
- Send pixel data to the backend via API
- Real-time prediction by a NumPy MLP

## Quick Start
(To be completed after implementation)
