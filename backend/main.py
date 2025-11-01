"""
FastAPI API for handwritten digit prediction.

Launch:
    uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import numpy as np
import os

from model import MLP
from utils import normalize_pixels

# Initialize FastAPI application
app = FastAPI(
    title="MNIST Digit Recognizer API",
    description="API to recognize handwritten digits with a NumPy MLP",
    version="1.0.0",
)

# Configure CORS to allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: specify allowed domains !!!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at application startup
MODEL_PATH = "saved_model/weights.pkl"
model = None


@app.on_event("startup")
async def load_model():
    """Load pre-trained model at server startup."""
    global model

    print("=" * 60)
    print("STARTING MNIST RECOGNIZER API")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        print(f"\nWARNING: Model does not exist at {MODEL_PATH}")
        print("   Please train the model first with:")
        print("   python train.py")
        print("\n   The API will start but predictions will fail.")
        model = None
    else:
        print(f"\nLoading model from {MODEL_PATH}...")
        model = MLP()
        model.load(MODEL_PATH)
        print("Model loaded successfully!")
        print(f"   Architecture: {' -> '.join(map(str, model.layer_sizes))}")

    print("\nAPI available at: http://localhost:8000")
    print("Documentation: http://localhost:8000/docs")
    print("=" * 60 + "\n")


# Pydantic models for data validation
class PredictRequest(BaseModel):
    pixels: List[float]  # Flat array of 784 values (28x28) normalized [0, 1]

    class Config:
        json_schema_extra = {
            "example": {
                "pixels": [0.0] * 784  # Example with all pixels at 0
            }
        }


class PredictResponse(BaseModel):
    prediction: int  # Predicted digit (0-9)
    probabilities: Dict[str, float]  # Probabilities for each class
    confidence: float  # Prediction confidence


# Endpoints


@app.get("/")
async def root():
    """Root endpoint - API information."""
    return {
        "message": "MNIST Digit Recognizer API",
        "version": "1.0.0",
        "status": "running" if model is not None else "model not loaded",
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/health (GET)",
            "docs": "/docs",
        },
    }


@app.get("/health")
async def health_check():
    """Check if API and model are ready."""
    if model is None:
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "message": f"Model not found at {MODEL_PATH}. Run 'python train.py' first.",
        }

    return {
        "status": "healthy",
        "model_loaded": True,
        "model_architecture": " -> ".join(map(str, model.layer_sizes)),
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """
    Predict a handwritten digit.

    Args:
        req: Request containing 784 normalized pixels [0, 1]

    Returns:
        Prediction with probabilities and confidence

    Raises:
        HTTPException: If model is not loaded or data is invalid
    """
    # Check model
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded. Please train the model first with 'python train.py'",
        )

    # Validate input size
    if len(req.pixels) != 784:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 784 pixels (28x28), got {len(req.pixels)}",
        )

    try:
        # Convert to NumPy array
        pixels = np.array(req.pixels, dtype=np.float32)

        # Check value range
        if pixels.min() < 0 or pixels.max() > 1:
            # Normalize if necessary
            pixels = normalize_pixels(pixels * 255)  # Assume [0, 255] range

        # Reshape for model
        pixels = pixels.reshape(1, -1)

        # Prediction
        pred_class = model.predict(pixels)[0]
        probas = model.predict_proba(pixels)[0]

        # Format response
        probabilities = {str(i): float(probas[i]) for i in range(10)}
        confidence = float(probas[pred_class])

        return PredictResponse(
            prediction=int(pred_class),
            probabilities=probabilities,
            confidence=confidence,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/model/info")
async def model_info():
    """Detailed model information."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    total_params = sum(w.size for w in model.weights) + sum(
        b.size for b in model.biases
    )

    return {
        "architecture": " -> ".join(map(str, model.layer_sizes)),
        "input_size": model.input_size,
        "hidden_sizes": model.hidden_sizes,
        "output_size": model.output_size,
        "total_parameters": total_params,
        "num_layers": len(model.weights),
        "weights_shapes": [w.shape for w in model.weights],
        "biases_shapes": [b.shape for b in model.biases],
    }


# Entry point to run with uvicorn
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload during development
        log_level="info",
    )
