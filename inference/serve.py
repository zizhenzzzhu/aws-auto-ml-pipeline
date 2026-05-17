"""Minimal FastAPI inference container entrypoint."""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, Request
from pydantic import BaseModel

from stages.training.train_pytorch import build_model


MODEL_DIR = Path(os.getenv("MODEL_DIR", "/opt/ml/model"))
model = None
metadata = None


class PredictionRequest(BaseModel):
    features: list[float]


@asynccontextmanager
async def lifespan(application: FastAPI):
    global model, metadata
    metadata = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
    model = build_model(len(metadata["feature_columns"]), metadata["hidden_dim"])
    model.load_state_dict(torch.load(MODEL_DIR / "model.pt", map_location="cpu", weights_only=True))
    model.eval()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ping")
def ping() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, float]:
    if model is None:
        raise RuntimeError("Model is not loaded.")
    with torch.no_grad():
        features = torch.tensor([request.features], dtype=torch.float32)
        score = float(torch.sigmoid(model(features)).item())
    return {"score": score}


@app.post("/invocations")
async def invocations(request: Request) -> dict[str, float]:
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        payload = await request.json()
        features = payload.get("features", payload)
    else:
        body = (await request.body()).decode("utf-8").strip()
        features = [float(value) for value in body.split(",") if value.strip()]
    return predict(PredictionRequest(features=features))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
