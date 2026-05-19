"""
scripts/fetch_era5.py

Downloads ERA5 monthly mean 2m temperature for every year 1950-2023.
Saves one NetCDF file per year into backend/data/raw/era5/.

Run from the project root:
    python scripts/fetch_era5.py

Or a specific year range:
    python scripts/fetch_era5.py --start 2000 --end 2023
"""

import argparse
import cdsapi
import os
import sys
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "backend" / "data" / "raw" / "era5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_START = 1950
DEFAULT_END = 2023


def fetch_year(client: cdsapi.Client, year: int) -> Path:
    out_path = OUTPUT_DIR / f"era5_t2m_{year}.nc"

    if out_path.exists():
        print(f"[skip] {year} already downloaded → {out_path.name}")
        return out_path

    print(f"[fetch] Requesting ERA5 2m temperature for {year}...")

    client.retrieve(
        "reanalysis-era5-single-levels-monthly-means",
        {
            "product_type": "monthly_averaged_reanalysis",
            "variable": "2m_temperature",
            "year": str(year),
            "month": [f"{m:02d}" for m in range(1, 13)],
            "time": "00:00",
            "data_format": "netcdf",
            "grid": "1.0/1.0",  # 1-degree resolution — ~180x360 grid, manageable size
        },
        str(out_path),
    )

    print(f"[done]  {year} saved → {out_path.name}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Fetch ERA5 2m temperature data.")
    parser.add_argument("--start", type=int, default=DEFAULT_START)
    parser.add_argument("--end", type=int, default=DEFAULT_END)
    args = parser.parse_args()

    if args.start < 1950 or args.end > 2023:
        print("ERA5 monthly means are available 1950–2023.")
        sys.exit(1)

    years = list(range(args.start, args.end + 1))
    print(f"\nGaia — ERA5 fetch")
    print(f"Years: {args.start}–{args.end} ({len(years)} files)")
    print(f"Output: {OUTPUT_DIR}\n")

    client = cdsapi.Client()

    failed = []
    for year in years:
        try:
            fetch_year(client, year)
        except Exception as e:
            print(f"[error] {year}: {e}")
            failed.append(year)

    print(f"\nDone. {len(years) - len(failed)}/{len(years)} years fetched.")
    if failed:
        print(f"Failed years: {failed}")
        print("Re-run the script — it will skip already-downloaded years.")


if __name__ == "__main__":
    main()