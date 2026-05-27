"""
data/ingestion/noaa.py

Fetches and normalises NOAA data into EcoNode instances.

Covers:
  - Hypoxic zone measurements (dissolved oxygen, Gulf of Mexico dead zone)
  - Sea surface temperature anomalies
  - NOAA Fisheries stock assessment indices

Data sources:
  - NOAA ERDDAP REST API (oceanographic time series)
  - NOAA CoastWatch (SST)
  - NOAA Fisheries (stock indices)

Docs: https://www.ncdc.noaa.gov/cdo-web/webservices/v2
      https://coastwatch.pfeg.noaa.gov/erddap/
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import requests

from graph.schema import EcoNode, NodeType

logger = logging.getLogger(__name__)

ERDDAP_BASE = "https://coastwatch.pfeg.noaa.gov/erddap/tabledap"
TIMEOUT     = 30


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _normalise_dissolved_oxygen(do_mg_l: float) -> float:
    """
    Normalise dissolved oxygen to [0, 1] stress scale.
    DO < 2 mg/L = hypoxic (value → 1.0)
    DO > 8 mg/L = healthy (value → 0.0)
    """
    return float(max(0.0, min(1.0, (8.0 - do_mg_l) / 6.0)))


def _normalise_sst_anomaly(anomaly_c: float) -> float:
    """
    Normalise SST anomaly to [0, 1] stress scale.
    +3°C anomaly → 1.0, 0°C → 0.0, negative anomalies clamped to 0.
    """
    return float(max(0.0, min(1.0, anomaly_c / 3.0)))


def _normalise_fish_biomass(index: float, baseline: float) -> float:
    """
    Normalise fish biomass index relative to a historical baseline.
    Returns stress: 0 = at or above baseline, 1 = fully depleted.
    """
    if baseline <= 0:
        return 0.0
    return float(max(0.0, min(1.0, 1.0 - (index / baseline))))


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------

def fetch_gulf_hypoxia(
    year: Optional[int] = None,
    lat:  float = 29.0,
    lon:  float = -90.5,
) -> Optional[EcoNode]:
    """
    Fetch Gulf of Mexico dissolved oxygen data from NOAA ERDDAP.
    Returns a HYPOXIA_ZONE EcoNode or None if fetch fails.

    Uses the NOAA/MBARI MBTS dataset as primary source.
    Falls back to the annual dead zone size estimate from LUMCON.
    """
    year = year or datetime.now(timezone.utc).year
    url  = (
        f"{ERDDAP_BASE}/erdNOAA_dissolved_oxygen.json"
        f"?time,latitude,longitude,dissolved_oxygen"
        f"&time>={year}-06-01&time<={year}-08-31"
        f"&latitude>={lat - 1.0}&latitude<={lat + 1.0}"
        f"&longitude>={lon - 1.0}&longitude<={lon + 1.0}"
        f"&orderByMean(\"time/1month\")"
    )

    try:
        resp = requests.get(url, timeout=TIMEOUT)
        resp.raise_for_status()
        rows = resp.json().get("table", {}).get("rows", [])

        if not rows:
            logger.warning("NOAA ERDDAP returned no hypoxia rows for %d", year)
            return _fallback_gulf_hypoxia(year, lat, lon)

        # Average DO across returned measurements
        do_values = [r[3] for r in rows if r[3] is not None]
        if not do_values:
            return _fallback_gulf_hypoxia(year, lat, lon)

        mean_do    = sum(do_values) / len(do_values)
        stress_val = _normalise_dissolved_oxygen(mean_do)

        return EcoNode(
            node_id   = f"gulf_hypoxia_{year}",
            node_type = NodeType.HYPOXIA_ZONE,
            region    = "Gulf of Mexico",
            latitude  = lat,
            longitude = lon,
            value     = stress_val,
            confidence= 0.9,
            metadata  = {
                "source":        "noaa_erddap",
                "mean_do_mg_l":  round(mean_do, 3),
                "year":          year,
                "n_measurements":len(do_values),
            },
        )

    except requests.RequestException as e:
        logger.error("NOAA hypoxia fetch failed: %s", e)
        return _fallback_gulf_hypoxia(year, lat, lon)


def _fallback_gulf_hypoxia(year: int, lat: float, lon: float) -> EcoNode:
    """
    Fallback using LUMCON annual dead zone size estimates.
    Normalised against the historical max (~22,720 km² in 2002).
    """
    # Historical annual dead zone sizes (km²) — LUMCON data
    lumcon = {
        2015: 16770, 2016: 17729, 2017: 22720, 2018: 2720,
        2019: 18006, 2020: 4880,  2021: 6334,  2022: 6597,
        2023: 5408,  2024: 6800,
    }
    area    = lumcon.get(year, 8000)
    stress  = min(1.0, area / 22720.0)

    return EcoNode(
        node_id   = f"gulf_hypoxia_{year}",
        node_type = NodeType.HYPOXIA_ZONE,
        region    = "Gulf of Mexico",
        latitude  = lat,
        longitude = lon,
        value     = stress,
        confidence= 0.75,
        metadata  = {
            "source":    "lumcon_fallback",
            "area_km2":  area,
            "year":      year,
        },
    )


def fetch_sst_anomaly(
    lat:    float = 29.0,
    lon:    float = -90.5,
    region: str   = "Gulf of Mexico",
) -> Optional[EcoNode]:
    """
    Fetch sea surface temperature anomaly from NOAA CoastWatch ERDDAP.
    Returns an OCEAN_TEMPERATURE EcoNode.
    """
    url = (
        f"{ERDDAP_BASE}/ncdcOisst21Agg_LonPM180.json"
        f"?time,latitude,longitude,sst,anom"
        f"&latitude>={lat - 0.5}&latitude<={lat + 0.5}"
        f"&longitude>={lon - 0.5}&longitude<={lon + 0.5}"
        f"&orderByMax(\"time\")"
    )

    try:
        resp = requests.get(url, timeout=TIMEOUT)
        resp.raise_for_status()
        rows = resp.json().get("table", {}).get("rows", [])

        if not rows:
            logger.warning("No SST data returned for lat=%.2f lon=%.2f", lat, lon)
            return None

        latest  = rows[-1]
        sst     = latest[2]
        anomaly = latest[3] if latest[3] is not None else 0.0
        stress  = _normalise_sst_anomaly(anomaly)

        return EcoNode(
            node_id   = f"sst_{region.lower().replace(' ', '_')}",
            node_type = NodeType.OCEAN_TEMPERATURE,
            region    = region,
            latitude  = lat,
            longitude = lon,
            value     = stress,
            confidence= 0.92,
            metadata  = {
                "source":        "noaa_coastwatch_oisst",
                "sst_c":         sst,
                "anomaly_c":     anomaly,
                "timestamp":     latest[0],
            },
        )

    except requests.RequestException as e:
        logger.error("NOAA SST fetch failed: %s", e)
        return None


def fetch_fish_population(
    species:  str   = "menhaden",
    region:   str   = "Gulf of Mexico",
    lat:      float = 29.5,
    lon:      float = -89.8,
    baseline: float = 1.0,
) -> Optional[EcoNode]:
    """
    Fetch NOAA Fisheries stock assessment index.
    Returns a FISH_POPULATION EcoNode.

    Note: Full NOAA Fisheries API requires registration.
    This fetcher targets the publicly available stock assessment
    summary endpoints. Baseline should be set to the species'
    maximum observed index value for normalisation.
    """
    url = (
        "https://apps-st.fisheries.noaa.gov/ods/foss/trade_data/"
        f"?species={species}&region={region.replace(' ', '+')}&format=json&limit=1"
    )

    try:
        resp  = requests.get(url, timeout=TIMEOUT)
        resp.raise_for_status()
        data  = resp.json()
        items = data.get("items", [])

        if not items:
            logger.warning("No fisheries data for species=%s region=%s", species, region)
            return None

        index  = float(items[0].get("value", baseline))
        stress = _normalise_fish_biomass(index, baseline)

        return EcoNode(
            node_id   = f"fish_{species}_{region.lower().replace(' ', '_')}",
            node_type = NodeType.FISH_POPULATION,
            region    = region,
            latitude  = lat,
            longitude = lon,
            value     = stress,
            confidence= 0.80,
            metadata  = {
                "source":  "noaa_fisheries",
                "species": species,
                "index":   index,
                "baseline":baseline,
            },
        )

    except requests.RequestException as e:
        logger.error("NOAA fisheries fetch failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Convenience: fetch all NOAA nodes for a region
# ---------------------------------------------------------------------------

def fetch_all_noaa_nodes(
    lat:    float = 29.0,
    lon:    float = -90.5,
    region: str   = "Gulf of Mexico",
    year:   Optional[int] = None,
) -> list[EcoNode]:
    """Fetch all available NOAA EcoNodes for a given region."""
    nodes = []
    for fetcher, kwargs in [
        (fetch_gulf_hypoxia,  {"year": year, "lat": lat, "lon": lon}),
        (fetch_sst_anomaly,   {"lat": lat, "lon": lon, "region": region}),
        (fetch_fish_population, {"lat": lat, "lon": lon, "region": region}),
    ]:
        node = fetcher(**kwargs)
        if node:
            nodes.append(node)
    logger.info("NOAA: fetched %d nodes for region=%s", len(nodes), region)
    return nodes