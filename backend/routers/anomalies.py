"""
backend/routers/anomalies.py

GET /api/anomalies?year=2014
Returns ML-flagged anomalous grid cells for a given year.
Isolation Forest is trained once at startup and stored in app.state.
"""

from fastapi import APIRouter, Request, HTTPException
from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np

router = APIRouter()

CONTAMINATION = 0.03  # flag ~3% of grid cells as anomalies


def train_isolation_forest(df: pd.DataFrame) -> IsolationForest:
    """Train Isolation Forest on the full dataset at startup."""
    print("Training Isolation Forest...")
    features = df[["lat", "lon", "anomaly_c"]].dropna()
    model = IsolationForest(
        n_estimators=100,
        contamination=CONTAMINATION,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(features)
    print("Isolation Forest ready.")
    return model


@router.on_event("startup")
async def startup():
    pass  # model is trained via main.py startup


@router.get("")
def get_anomalies(year: int, request: Request):
    df: pd.DataFrame = request.app.state.df

    available_years = df["year"].unique().tolist()
    if year not in available_years:
        raise HTTPException(
            status_code=404,
            detail=f"Year {year} not available."
        )

    # Train model lazily on first request and cache it
    if not hasattr(request.app.state, "isolation_forest"):
        request.app.state.isolation_forest = train_isolation_forest(df)

    model: IsolationForest = request.app.state.isolation_forest
    year_df = df[df["year"] == year].copy()

    features = year_df[["lat", "lon", "anomaly_c"]].fillna(0)
    preds = model.predict(features)          # -1 = anomaly, 1 = normal
    scores = model.decision_function(features)  # lower = more anomalous

    year_df = year_df.copy()
    year_df["is_anomaly"] = preds == -1
    year_df["anomaly_score"] = scores

    anomalies = year_df[year_df["is_anomaly"]].sort_values("anomaly_score")

    features_out = []
    for row in anomalies.itertuples(index=False):
        features_out.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [row.lon, row.lat]
            },
            "properties": {
                "temp_c": row.temp_c,
                "anomaly_c": row.anomaly_c,
                "anomaly_score": round(float(row.anomaly_score), 4),
            }
        })

    return {
        "type": "FeatureCollection",
        "year": year,
        "anomaly_count": len(features_out),
        "features": features_out,
    }