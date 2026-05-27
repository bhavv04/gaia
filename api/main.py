"""
api/main.py

FastAPI application for GAIA ecological cascade prediction.

Endpoints:
  GET  /health                    — liveness check
  POST /predict                   — run cascade prediction for a region
  GET  /graph/{region}            — return current graph snapshot for a region
  GET  /nodes/{region}            — return current EcoNode values for a region

Usage:
    uvicorn api.main:app --reload --port 8000
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import health, predict, graph
from model.gnn import GAIANet

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Model registry — loaded once at startup
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, GAIANet] = {}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_MODEL_PATH = Path("model/experiments/best_model.pt")


def load_model(path: Path = DEFAULT_MODEL_PATH) -> Optional[GAIANet]:
    if not path.exists():
        logger.warning(
            "No trained model found at %s — predictions will use untrained weights. "
            "Run model/train.py to train the model first.",
            path,
        )
        return GAIANet()  # untrained, for dev/demo purposes

    model = GAIANet()
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    logger.info("Loaded GAIA model from %s on %s", path, DEVICE)
    return model


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    MODEL_REGISTRY["gaia"] = load_model()
    logger.info("GAIA API ready")
    yield
    MODEL_REGISTRY.clear()
    logger.info("GAIA API shutdown")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title       = "GAIA — Ecological Cascade Prediction API",
    description = (
        "Predicts cross-domain ecological cascade failures using a "
        "temporal graph neural network. Given a geographic region, "
        "GAIA fetches live environmental data, builds an ecological "
        "indicator graph, and returns a cascade prediction sequence."
    ),
    version     = "0.1.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["http://localhost:3000"],  # Next.js dev server
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

app.include_router(health.router)
app.include_router(predict.router, prefix="/predict")
app.include_router(graph.router,   prefix="/graph")


# Make model registry accessible to routes via app state
@app.on_event("startup")
async def attach_state():
    app.state.model  = MODEL_REGISTRY.get("gaia")
    app.state.device = DEVICE