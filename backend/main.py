from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    pixels: List[float]  # flat array of 28*28 values

class PredictResponse(BaseModel):
    prediction: int

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    # TODO: replace with real model call
    # For now, always returns 0 to test the API
    return PredictResponse(prediction=0)
