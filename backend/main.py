"""
backend/main.py

Gaia API — FastAPI entry point.
Run with:
    uvicorn backend.main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import json
from pathlib import Path

from backend.routers import temperature, anomalies

# ── Data loading ────────────────────────────────────────────────────────────
PROCESSED_DIR = Path(__file__).parent / "data" / "processed"

def load_data() -> tuple[pd.DataFrame, list[dict]]:
    parquet_path = PROCESSED_DIR / "era5_temperature.parquet"
    summary_path = PROCESSED_DIR / "era5_summary.json"

    if not parquet_path.exists():
        raise RuntimeError("Processed data not found — run era5_loader.py first.")

    df = pd.read_parquet(parquet_path)

    with open(summary_path) as f:
        summary = json.load(f)

    print(f"Loaded {len(df):,} records covering years {df['year'].min()}–{df['year'].max()}")
    return df, summary


# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(title="Gaia API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Load data once at startup — routers read from app.state
@app.on_event("startup")
async def startup():
    app.state.df, app.state.summary = load_data()

# ── Routes ───────────────────────────────────────────────────────────────────
app.include_router(temperature.router, prefix="/api/temperature")
app.include_router(anomalies.router, prefix="/api/anomalies")

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.get("/api/years")
def years(request: "Request"):
    from fastapi import Request
    df: pd.DataFrame = request.app.state.df
    return {"years": sorted(df["year"].unique().tolist())}