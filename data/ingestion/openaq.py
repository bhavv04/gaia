"""
data/ingestion/openaq.py

Fetches and normalises OpenAQ air quality data into EcoNode instances.

Covers:
  - PM2.5 and PM10 particulate matter
  - NO2, SO2, O3 concentrations
  - Composite AQI stress score

Data source:
  - OpenAQ v3 API (free, no auth required for basic queries)
  - Docs: https://docs.openaq.org/
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
import os
import requests

from graph.schema import EcoNode, NodeType

logger  = logging.getLogger(__name__)
BASE    = "https://api.openaq.org/v3"
TIMEOUT = 30
OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY", "")

# WHO annual guideline limits (μg/m³) used as normalisation ceiling
WHO_LIMITS = {
    "pm25": 15.0,
    "pm10": 45.0,
    "no2":  10.0,
    "so2":  40.0,
    "o3":   60.0,
}


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def _normalise_pollutant(value: float, parameter: str) -> float:
    """
    Normalise a pollutant concentration to [0, 1] stress scale
    relative to WHO annual guideline limits.
    1.0 = at or above 3× the WHO limit (severe)
    0.0 = at or below the WHO limit (healthy)
    """
    limit = WHO_LIMITS.get(parameter.lower(), 1.0)
    return float(max(0.0, min(1.0, value / (limit * 3.0))))


def _composite_aqi_stress(measurements: dict[str, float]) -> float:
    """Weighted composite stress from multiple pollutant readings."""
    weights = {"pm25": 0.40, "pm10": 0.20, "no2": 0.20, "so2": 0.10, "o3": 0.10}
    total_w, score = 0.0, 0.0
    for param, val in measurements.items():
        w = weights.get(param.lower(), 0.0)
        if w > 0:
            score   += w * _normalise_pollutant(val, param)
            total_w += w
    return score / total_w if total_w > 0 else 0.0


# ---------------------------------------------------------------------------
# Location lookup
# ---------------------------------------------------------------------------

def _find_location_id(lat: float, lon: float, radius_m: int = 25000) -> Optional[int]:
    """Find the nearest OpenAQ location ID for a given lat/lon."""
    url    = f"{BASE}/locations"
    params = {
        "coordinates": f"{lat},{lon}",
        "radius":      radius_m,
        "limit":       5,
        "order_by":    "distance",
    }
    try:
        resp = requests.get(url, params=params, timeout=TIMEOUT, headers={"X-API-Key": OPENAQ_API_KEY})
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if results:
            return results[0]["id"]
    except requests.RequestException as e:
        logger.error("OpenAQ location lookup failed: %s", e)
    return None


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------

def fetch_air_quality(
    lat:    float,
    lon:    float,
    region: str,
    days_back: int = 7,
) -> Optional[EcoNode]:
    """
    Fetch recent air quality measurements for a location.
    Averages readings over the past `days_back` days.
    Returns an AIR_QUALITY EcoNode or None if fetch fails.
    """
    location_id = _find_location_id(lat, lon)
    if location_id is None:
        logger.warning("No OpenAQ location found near lat=%.2f lon=%.2f", lat, lon)
        return None

    date_from = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%dT%H:%M:%SZ")
    url       = f"{BASE}/measurements"
    params    = {
        "location_id": location_id,
        "date_from":   date_from,
        "limit":       1000,
    }

    try:
        resp = requests.get(url, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        results = resp.json().get("results", [])

        if not results:
            logger.warning("No OpenAQ measurements for location_id=%d", location_id)
            return None

        # Aggregate by parameter
        param_values: dict[str, list[float]] = {}
        for r in results:
            param = r.get("parameter", "").lower()
            val   = r.get("value")
            if param and val is not None and val >= 0:
                param_values.setdefault(param, []).append(val)

        mean_values = {p: sum(v) / len(v) for p, v in param_values.items() if v}
        if not mean_values:
            return None

        stress = _composite_aqi_stress(mean_values)

        return EcoNode(
            node_id   = f"air_{region.lower().replace(' ', '_')}",
            node_type = NodeType.AIR_QUALITY,
            region    = region,
            latitude  = lat,
            longitude = lon,
            value     = stress,
            confidence= 0.88,
            metadata  = {
                "source":      "openaq_v3",
                "location_id": location_id,
                "days_back":   days_back,
                "mean_values": {p: round(v, 3) for p, v in mean_values.items()},
            },
        )

    except requests.RequestException as e:
        logger.error("OpenAQ measurements fetch failed: %s", e)
        return None


def fetch_precipitation_proxy(
    lat:    float,
    lon:    float,
    region: str,
) -> Optional[EcoNode]:
    """
    OpenAQ doesn't provide precipitation directly.
    This fetches SO2 as a proxy for industrial/atmospheric stress
    and returns a PRECIPITATION node with low confidence,
    flagged for replacement by a dedicated weather API.

    In production, replace with Open-Meteo or NOAA precipitation endpoint.
    """
    location_id = _find_location_id(lat, lon)
    if location_id is None:
        return None

    url    = f"{BASE}/measurements"
    params = {"location_id": location_id, "parameter": "so2", "limit": 100}

    try:
        resp  = requests.get(url, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        rows  = resp.json().get("results", [])
        vals  = [r["value"] for r in rows if r.get("value", -1) >= 0]

        if not vals:
            return None

        mean_so2 = sum(vals) / len(vals)
        stress   = _normalise_pollutant(mean_so2, "so2")

        return EcoNode(
            node_id   = f"precip_proxy_{region.lower().replace(' ', '_')}",
            node_type = NodeType.PRECIPITATION,
            region    = region,
            latitude  = lat,
            longitude = lon,
            value     = stress,
            confidence= 0.40,   # low — this is a proxy, not actual precipitation
            metadata  = {
                "source":  "openaq_so2_proxy",
                "note":    "Replace with Open-Meteo precipitation endpoint",
                "so2_mean":round(mean_so2, 3),
            },
        )

    except requests.RequestException as e:
        logger.error("OpenAQ SO2 proxy fetch failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def fetch_all_openaq_nodes(
    lat:    float,
    lon:    float,
    region: str,
) -> list[EcoNode]:
    """Fetch all available OpenAQ EcoNodes for a given region."""
    nodes = []
    for fetcher, kwargs in [
        (fetch_air_quality,          {"lat": lat, "lon": lon, "region": region}),
        (fetch_precipitation_proxy,  {"lat": lat, "lon": lon, "region": region}),
    ]:
        node = fetcher(**kwargs)
        if node:
            nodes.append(node)
    logger.info("OpenAQ: fetched %d nodes for region=%s", len(nodes), region)
    return nodes