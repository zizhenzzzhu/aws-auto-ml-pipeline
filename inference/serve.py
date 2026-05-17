"""Minimal FastAPI inference container entrypoint."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from fastapi import FastAPI
from pydantic import BaseModel

from stages.training.train_pytorch import build_model


MODEL_DIR = Path(os.getenv("MODEL_DIR", "/opt/ml/model"))
app = FastAPI()
model = None
metadata = None


class PredictionRequest(BaseModel):
    features: list[float]


@app.on_event("startup")
def load_model() -> None:
    global model, metadata
    metadata = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
    model = build_model(len(metadata["feature_columns"]), metadata["hidden_dim"])
    model.load_state_dict(torch.load(MODEL_DIR / "model.pt", map_location="cpu"))
    model.eval()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, float]:
    if model is None:
        raise RuntimeError("Model is not loaded.")
    with torch.no_grad():
        features = torch.tensor([request.features], dtype=torch.float32)
        score = float(torch.sigmoid(model(features)).item())
    return {"score": score}
