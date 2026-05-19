"""
backend/routers/temperature.py

GET /api/temperature?year=2014
Returns grid-cell temperature and anomaly data for a given year
as a lightweight GeoJSON FeatureCollection.
"""

from fastapi import APIRouter, Request, HTTPException
import pandas as pd

router = APIRouter()


@router.get("")
def get_temperature(year: int, request: Request):
    df: pd.DataFrame = request.app.state.df

    available_years = df["year"].unique().tolist()
    if year not in available_years:
        raise HTTPException(
            status_code=404,
            detail=f"Year {year} not available. Range: {min(available_years)}–{max(available_years)}"
        )

    year_df = df[df["year"] == year][["lat", "lon", "temp_c", "anomaly_c"]]

    # Build a compact GeoJSON FeatureCollection
    features = []
    for row in year_df.itertuples(index=False):
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [row.lon, row.lat]
            },
            "properties": {
                "temp_c": row.temp_c,
                "anomaly_c": row.anomaly_c,
            }
        })

    return {
        "type": "FeatureCollection",
        "year": year,
        "features": features,
    }


@router.get("/summary")
def get_summary(request: Request):
    """Returns per-year global stats for the frontend stats panel."""
    return {"summary": request.app.state.summary}