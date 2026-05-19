"""
backend/data_processing/era5_loader.py

Loads raw ERA5 NetCDF files and processes them into:
  - Annual mean temperature per grid cell
  - Temperature anomaly relative to 1951-1980 baseline
  - GeoJSON-ready output saved as parquet for fast API serving

Run from project root after fetch_era5.py:
    python -m backend.data_processing.era5_loader

Outputs land in backend/data/processed/
"""

import numpy as np
import pandas as pd
import xarray as xr
import json
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw" / "era5"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Baseline period for anomaly calculation (standard climatological baseline)
BASELINE_START = 1951
BASELINE_END = 1980

# Kelvin to Celsius
KELVIN_OFFSET = 273.15


def load_year(year: int) -> xr.DataArray:
    """Load a single year's NetCDF and return annual mean in Celsius."""
    path = RAW_DIR / f"era5_t2m_{year}.nc"
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path.name} — run fetch_era5.py first")

    ds = xr.open_dataset(path)

    # ERA5 variable name is either 't2m' or 'VAR_2T' depending on format
    var_name = "t2m" if "t2m" in ds else "VAR_2T"
    da = ds[var_name]

    # Average across all 12 months → annual mean
    annual_mean = da.mean(dim="valid_time" if "valid_time" in da.dims else "time")

    # Convert Kelvin → Celsius
    annual_mean = annual_mean - KELVIN_OFFSET

    ds.close()
    return annual_mean


def build_baseline(years: list[int]) -> xr.DataArray:
    """Compute the mean temperature across baseline years at each grid cell."""
    print(f"Building baseline ({BASELINE_START}–{BASELINE_END})...")
    arrays = []
    for year in years:
        try:
            arrays.append(load_year(year))
        except FileNotFoundError as e:
            print(f"  [warn] {e}")

    if not arrays:
        raise RuntimeError("No baseline years found — check your raw data directory.")

    stacked = xr.concat(arrays, dim="year")
    baseline = stacked.mean(dim="year")
    print(f"  Baseline built from {len(arrays)} years.")
    return baseline


def dataarray_to_records(da: xr.DataArray, year: int, anomaly: xr.DataArray) -> list[dict]:
    """Convert xarray DataArrays to a flat list of records for parquet storage."""
    # Coarsen to 2-degree grid to reduce output size (optional — remove for full res)
    da_coarse = da.coarsen(latitude=2, longitude=2, boundary="trim").mean()
    anom_coarse = anomaly.coarsen(latitude=2, longitude=2, boundary="trim").mean()

    lats = da_coarse.latitude.values
    lons = da_coarse.longitude.values

    records = []
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            temp = float(da_coarse.values[i, j])
            anom = float(anom_coarse.values[i, j])

            if np.isnan(temp):
                continue

            records.append({
                "year": year,
                "lat": round(float(lat), 1),
                "lon": round(float(lon), 1),
                "temp_c": round(temp, 3),
                "anomaly_c": round(anom, 3),
            })

    return records


def process_all(start: int = 1950, end: int = 2023):
    """Process all years and save to parquet."""
    years = list(range(start, end + 1))
    available = [y for y in years if (RAW_DIR / f"era5_t2m_{y}.nc").exists()]

    print(f"\nGaia — ERA5 Loader")
    print(f"Available years: {available[0]}–{available[-1]} ({len(available)} files)")

    # Build baseline
    baseline_years = [y for y in range(BASELINE_START, BASELINE_END + 1) if y in available]
    baseline = build_baseline(baseline_years)

    # Process each year
    all_records = []
    for year in available:
        print(f"  Processing {year}...", end=" ")
        try:
            annual = load_year(year)
            anomaly = annual - baseline
            records = dataarray_to_records(annual, year, anomaly)
            all_records.extend(records)
            print(f"{len(records)} grid cells")
        except Exception as e:
            print(f"error — {e}")

    # Save to parquet
    df = pd.DataFrame(all_records)
    out_path = PROCESSED_DIR / "era5_temperature.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nSaved {len(df):,} records → {out_path}")
    print(f"File size: {out_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Also save a summary JSON (min/max anomaly per year) for the frontend stats panel
    summary = (
        df.groupby("year")
        .agg(
            mean_anomaly=("anomaly_c", "mean"),
            max_anomaly=("anomaly_c", "max"),
            min_anomaly=("anomaly_c", "min"),
        )
        .round(3)
        .reset_index()
        .to_dict(orient="records")
    )
    summary_path = PROCESSED_DIR / "era5_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f)
    print(f"Saved summary → {summary_path}")

    return df


if __name__ == "__main__":
    process_all()