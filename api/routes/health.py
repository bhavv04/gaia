"""
api/routes/health.py

Liveness and readiness endpoints.
"""

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health", tags=["meta"])
async def health(request: Request):
    model_loaded = getattr(request.app.state, "model", None) is not None
    return {
        "status":       "ok",
        "model_loaded": model_loaded,
        "version":      "0.1.0",
    }