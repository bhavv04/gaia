"""
data/ingestion/inaturalist.py

Fetches and normalises iNaturalist observation data into EcoNode instances.

Covers:
  - Predator population density (apex / mesopredators)
  - Biodiversity index (species richness anomaly)
  - Benthic species presence as proxy for habitat health

Data source:
  - iNaturalist API v1 (free, no auth required for read operations)
  - Docs: https://api.inaturalist.org/v1/docs/

Note on limitations:
  iNaturalist is citizen science data — observation density is biased
  toward populated/accessible areas. Confidence values reflect this.
  All nodes from this source should be used alongside other indicators,
  not as standalone ground truth.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

from graph.schema import EcoNode, NodeType

logger  = logging.getLogger(__name__)
BASE    = "https://api.inaturalist.org/v1"
TIMEOUT = 30


# ---------------------------------------------------------------------------
# Taxon IDs for key indicator species groups
# ---------------------------------------------------------------------------

# iNaturalist taxon IDs for relevant species groups
TAXON_IDS = {
    # Marine predators
    "bottlenose_dolphin":    41573,
    "brown_pelican":         4849,
    "osprey":                5305,
    "great_blue_heron":      4956,

    # Benthic indicator species
    "blue_crab":             127397,
    "eastern_oyster":        127347,
    "fiddler_crab":          126576,

    # Terrestrial coastal predators
    "bald_eagle":            4849,
    "river_otter":           42418,

    # Algae / bloom indicators
    "cyanobacteria":         67333,
}

# Groups for each node type
PREDATOR_TAXA   = ["bottlenose_dolphin", "brown_pelican", "osprey", "great_blue_heron", "bald_eagle"]
BENTHIC_TAXA    = ["blue_crab", "eastern_oyster", "fiddler_crab"]
BLOOM_TAXA      = ["cyanobacteria"]


# ---------------------------------------------------------------------------
# Core observation fetcher
# ---------------------------------------------------------------------------

def _fetch_observation_count(
    taxon_id:  int,
    lat:       float,
    lon:       float,
    radius_km: float = 50.0,
    days_back: int   = 90,
) -> int:
    """
    Fetch observation count for a taxon within a geographic radius
    over the past `days_back` days.
    """
    date_since = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")
    url        = f"{BASE}/observations"
    params     = {
        "taxon_id":  taxon_id,
        "lat":       lat,
        "lng":       lon,
        "radius":    radius_km,
        "d1":        date_since,
        "quality_grade": "research",
        "per_page":  0,   # we only need total_results
    }
    try:
        resp = requests.get(url, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json().get("total_results", 0)
    except requests.RequestException as e:
        logger.error("iNaturalist observation fetch failed for taxon %d: %s", taxon_id, e)
        return 0


def _fetch_species_count(
    lat:       float,
    lon:       float,
    radius_km: float = 50.0,
    days_back: int   = 365,
) -> int:
    """
    Fetch total species count (richness) observed in a region over past year.
    Used as a biodiversity index.
    """
    date_since = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")
    url        = f"{BASE}/observations/species_counts"
    params     = {
        "lat":    lat,
        "lng":    lon,
        "radius": radius_km,
        "d1":     date_since,
        "quality_grade": "research",
        "per_page": 0,
    }
    try:
        resp = requests.get(url, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json().get("total_results", 0)
    except requests.RequestException as e:
        logger.error("iNaturalist species count fetch failed: %s", e)
        return 0


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def _normalise_observation_decline(current: int, baseline: int) -> float:
    """
    Normalise population stress from observation count vs baseline.
    0.0 = at or above baseline (healthy)
    1.0 = zero observations (locally absent)
    """
    if baseline <= 0:
        return 0.5  # no baseline, assume moderate uncertainty
    return float(max(0.0, min(1.0, 1.0 - (current / baseline))))


def _normalise_species_richness(current: int, baseline: int) -> float:
    """
    Normalise biodiversity stress from species count vs historical baseline.
    """
    if baseline <= 0:
        return 0.5
    return float(max(0.0, min(1.0, 1.0 - (current / baseline))))


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------

def fetch_predator_population(
    lat:               float,
    lon:               float,
    region:            str,
    radius_km:         float         = 75.0,
    days_back:         int           = 90,
    baseline_per_taxon: dict[str, int] = None,
) -> Optional[EcoNode]:
    """
    Fetch predator population indicator from iNaturalist observation counts.
    Aggregates across PREDATOR_TAXA and normalises against a baseline.

    Returns a PREDATOR_POPULATION EcoNode.

    Args:
        baseline_per_taxon: Historical baseline observation counts per taxon.
                            If None, uses hardcoded Gulf Coast baselines.
    """
    # Gulf Coast baselines (90-day research-grade observation counts)
    default_baselines = {
        "bottlenose_dolphin": 45,
        "brown_pelican":      120,
        "osprey":             85,
        "great_blue_heron":   200,
        "bald_eagle":         30,
    }
    baselines = baseline_per_taxon or default_baselines

    counts: dict[str, int] = {}
    for taxon_name in PREDATOR_TAXA:
        taxon_id = TAXON_IDS.get(taxon_name)
        if taxon_id:
            counts[taxon_name] = _fetch_observation_count(
                taxon_id, lat, lon, radius_km, days_back
            )

    if not counts:
        return None

    # Weighted stress across taxa
    stress_scores = [
        _normalise_observation_decline(counts[t], baselines.get(t, 50))
        for t in counts
    ]
    mean_stress = sum(stress_scores) / len(stress_scores)

    return EcoNode(
        node_id   = f"predator_{region.lower().replace(' ', '_')}",
        node_type = NodeType.PREDATOR_POPULATION,
        region    = region,
        latitude  = lat,
        longitude = lon,
        value     = mean_stress,
        confidence= 0.55,   # citizen science bias acknowledged
        metadata  = {
            "source":     "inaturalist_v1",
            "taxa":       list(counts.keys()),
            "counts":     counts,
            "baselines":  {t: baselines.get(t) for t in counts},
            "radius_km":  radius_km,
            "days_back":  days_back,
            "note":       "Citizen science data — observation density biased toward accessible areas",
        },
    )


def fetch_benthic_health_proxy(
    lat:       float,
    lon:       float,
    region:    str,
    radius_km: float = 50.0,
    days_back: int   = 180,
    baselines: dict[str, int] = None,
) -> Optional[EcoNode]:
    """
    Fetch benthic habitat health from iNaturalist benthic indicator species.
    Oysters, crabs, and fiddler crabs are sensitive hypoxia indicators.
    Returns a BENTHIC_HABITAT EcoNode.
    """
    default_baselines = {
        "blue_crab":    300,
        "eastern_oyster": 150,
        "fiddler_crab": 200,
    }
    baselines = baselines or default_baselines

    counts: dict[str, int] = {}
    for taxon_name in BENTHIC_TAXA:
        taxon_id = TAXON_IDS.get(taxon_name)
        if taxon_id:
            counts[taxon_name] = _fetch_observation_count(
                taxon_id, lat, lon, radius_km, days_back
            )

    if not counts:
        return None

    stress_scores = [
        _normalise_observation_decline(counts[t], default_baselines.get(t, 100))
        for t in counts
    ]
    mean_stress = sum(stress_scores) / len(stress_scores)

    return EcoNode(
        node_id   = f"benthic_{region.lower().replace(' ', '_')}",
        node_type = NodeType.BENTHIC_HABITAT,
        region    = region,
        latitude  = lat,
        longitude = lon,
        value     = mean_stress,
        confidence= 0.50,
        metadata  = {
            "source":    "inaturalist_v1_benthic",
            "taxa":      list(counts.keys()),
            "counts":    counts,
            "radius_km": radius_km,
            "days_back": days_back,
        },
    )


def fetch_bloom_indicator(
    lat:       float,
    lon:       float,
    region:    str,
    radius_km: float = 30.0,
    days_back: int   = 60,
) -> Optional[EcoNode]:
    """
    Fetch cyanobacteria observation count as a citizen-science bloom indicator.
    Low confidence — supplement with NASA chlorophyll data.
    Returns an ALGAE_BLOOM EcoNode.
    """
    taxon_id = TAXON_IDS["cyanobacteria"]
    count    = _fetch_observation_count(taxon_id, lat, lon, radius_km, days_back)

    # Normalise: 0 obs = 0 stress, 50+ obs = high stress
    stress = float(min(1.0, count / 50.0))

    return EcoNode(
        node_id   = f"bloom_citizen_{region.lower().replace(' ', '_')}",
        node_type = NodeType.ALGAE_BLOOM,
        region    = region,
        latitude  = lat,
        longitude = lon,
        value     = stress,
        confidence= 0.40,
        metadata  = {
            "source":    "inaturalist_v1_cyanobacteria",
            "count":     count,
            "radius_km": radius_km,
            "days_back": days_back,
            "note":      "Supplement with NASA MODIS chlorophyll for higher confidence",
        },
    )


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def fetch_all_inaturalist_nodes(
    lat:       float,
    lon:       float,
    region:    str,
    radius_km: float = 75.0,
) -> list[EcoNode]:
    """Fetch all available iNaturalist EcoNodes for a given region."""
    nodes = []
    for fetcher, kwargs in [
        (fetch_predator_population,  {"lat": lat, "lon": lon, "region": region, "radius_km": radius_km}),
        (fetch_benthic_health_proxy, {"lat": lat, "lon": lon, "region": region, "radius_km": radius_km}),
        (fetch_bloom_indicator,      {"lat": lat, "lon": lon, "region": region, "radius_km": radius_km}),
    ]:
        node = fetcher(**kwargs)
        if node:
            nodes.append(node)
    logger.info("iNaturalist: fetched %d nodes for region=%s", len(nodes), region)
    return nodes