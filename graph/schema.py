"""
graph/schema.py

Defines the typed node and edge schema for GAIA's ecological graph.
All cascade propagation logic operates on this structure.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Domain taxonomy
# ---------------------------------------------------------------------------

class Domain(str, Enum):
    """Ecological domain a node belongs to."""
    MARINE      = "marine"
    ATMOSPHERIC = "atmospheric"
    TERRESTRIAL = "terrestrial"
    FRESHWATER  = "freshwater"


# ---------------------------------------------------------------------------
# Node types
# ---------------------------------------------------------------------------

class NodeType(str, Enum):
    """
    Categorical type of an ecological indicator node.
    Each type belongs to one canonical domain but cross-domain
    interactions are captured via edges, not node membership.
    """
    # Marine
    HYPOXIA_ZONE        = "hypoxia_zone"        # dissolved oxygen deficit region
    FISH_POPULATION     = "fish_population"     # biomass estimate for a species group
    ALGAE_BLOOM         = "algae_bloom"         # chlorophyll-a / cyanobacteria density
    BENTHIC_HABITAT     = "benthic_habitat"     # seafloor community health index
    OCEAN_TEMPERATURE   = "ocean_temperature"   # SST anomaly

    # Atmospheric
    AIR_QUALITY         = "air_quality"         # AQI / PM2.5 composite
    CO2_CONCENTRATION   = "co2_concentration"   # ppm at surface level
    PRECIPITATION       = "precipitation"       # mm/day anomaly

    # Terrestrial
    SOIL_HEALTH         = "soil_health"         # organic matter / pH composite
    VEGETATION_INDEX    = "vegetation_index"    # NDVI anomaly
    PREDATOR_POPULATION = "predator_population" # apex/mesopredator density

    # Freshwater
    NUTRIENT_RUNOFF     = "nutrient_runoff"     # nitrogen / phosphorus load
    WATER_TEMPERATURE   = "water_temperature"   # freshwater thermal anomaly


NODE_DOMAIN: dict[NodeType, Domain] = {
    NodeType.HYPOXIA_ZONE:        Domain.MARINE,
    NodeType.FISH_POPULATION:     Domain.MARINE,
    NodeType.ALGAE_BLOOM:         Domain.MARINE,
    NodeType.BENTHIC_HABITAT:     Domain.MARINE,
    NodeType.OCEAN_TEMPERATURE:   Domain.MARINE,
    NodeType.AIR_QUALITY:         Domain.ATMOSPHERIC,
    NodeType.CO2_CONCENTRATION:   Domain.ATMOSPHERIC,
    NodeType.PRECIPITATION:       Domain.ATMOSPHERIC,
    NodeType.SOIL_HEALTH:         Domain.TERRESTRIAL,
    NodeType.VEGETATION_INDEX:    Domain.TERRESTRIAL,
    NodeType.PREDATOR_POPULATION: Domain.TERRESTRIAL,
    NodeType.NUTRIENT_RUNOFF:     Domain.FRESHWATER,
    NodeType.WATER_TEMPERATURE:   Domain.FRESHWATER,
}


# ---------------------------------------------------------------------------
# Edge types
# ---------------------------------------------------------------------------

class EdgeType(str, Enum):
    """
    Directed causal relationship between two ecological indicator nodes.
    Direction encodes: source drives / influences target.
    """
    # Marine internal
    HYPOXIA_DRIVES_FISH_DISPLACEMENT    = "hypoxia_drives_fish_displacement"
    FISH_GRAZING_CONTROLS_ALGAE         = "fish_grazing_controls_algae"
    ALGAE_BLOOM_DEPLETES_OXYGEN         = "algae_bloom_depletes_oxygen"
    HYPOXIA_DEGRADES_BENTHIC            = "hypoxia_degrades_benthic"
    SST_ANOMALY_TRIGGERS_BLOOM          = "sst_anomaly_triggers_bloom"

    # Cross-domain: marine → terrestrial
    FISH_DISPLACEMENT_SHIFTS_PREDATOR   = "fish_displacement_shifts_predator"
    BENTHIC_LOSS_REDUCES_SHOREBIRD      = "benthic_loss_reduces_shorebird"

    # Cross-domain: freshwater → marine
    NUTRIENT_RUNOFF_FEEDS_BLOOM         = "nutrient_runoff_feeds_bloom"
    NUTRIENT_RUNOFF_DRIVES_HYPOXIA      = "nutrient_runoff_drives_hypoxia"

    # Cross-domain: atmospheric → marine / freshwater
    PRECIPITATION_INCREASES_RUNOFF      = "precipitation_increases_runoff"
    CO2_DRIVES_OCEAN_TEMPERATURE        = "co2_drives_ocean_temperature"

    # Cross-domain: terrestrial → freshwater
    SOIL_DEGRADATION_INCREASES_RUNOFF   = "soil_degradation_increases_runoff"
    VEGETATION_LOSS_INCREASES_RUNOFF    = "vegetation_loss_increases_runoff"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EcoNode:
    """
    A single ecological indicator node in the GAIA graph.

    Attributes:
        node_id:    Unique identifier, e.g. "gulf_mexico_hypoxia_2024"
        node_type:  Categorical type from NodeType enum
        region:     ISO 3166-1 alpha-2 country code or named region string
        latitude:   Centroid latitude of the monitored area
        longitude:  Centroid longitude of the monitored area
        value:      Current normalised indicator value in [0, 1]
                    where 1.0 = maximum observed degradation / stress
        confidence: Model or sensor confidence in the current value [0, 1]
        metadata:   Arbitrary key-value store for source attribution,
                    units, raw values, timestamps, etc.
    """
    node_id:    str
    node_type:  NodeType
    region:     str
    latitude:   float
    longitude:  float
    value:      float = 0.0 # Normalised to [0, 1] where 1.0 = max observed degradation, somethign like that 
    confidence: float = 1.0 # Confidence in the value itself changed to be between 0 and 1, where 1 is high confidence and 0 is low confidence. kill me
    metadata:   dict  = field(default_factory=dict)

    @property
    def domain(self) -> Domain:
        return NODE_DOMAIN[self.node_type]

    def __repr__(self) -> str:
        return (
            f"EcoNode({self.node_id!r}, type={self.node_type.value}, "
            f"domain={self.domain.value}, value={self.value:.3f})"
        )


@dataclass
class EcoEdge:
    """
    A directed causal edge between two EcoNodes.

    Attributes:
        source_id:   node_id of the driving indicator
        target_id:   node_id of the affected indicator
        edge_type:   Categorical relationship type from EdgeType enum
        weight:      Learned or empirically-derived influence strength [0, 1]
        lag_hours:   Expected propagation delay in hours (e.g. 72 = 3 days)
        confidence:  Confidence in the causal relationship itself [0, 1]
        cross_domain: True if source and target belong to different domains
    """
    source_id:    str
    target_id:    str
    edge_type:    EdgeType
    weight:       float = 1.0
    lag_hours:    int   = 0
    confidence:   float = 1.0
    cross_domain: bool  = False

    def __repr__(self) -> str:
        return (
            f"EcoEdge({self.source_id!r} → {self.target_id!r}, "
            f"type={self.edge_type.value}, weight={self.weight:.3f}, "
            f"lag={self.lag_hours}h)"
        )


# ---------------------------------------------------------------------------
# Graph-level metadata
# ---------------------------------------------------------------------------

@dataclass
class GraphMetadata:
    """
    Top-level metadata attached to a GAIA graph snapshot.

    Attributes:
        snapshot_id:   Unique identifier for this graph state
        timestamp:     ISO 8601 UTC timestamp of the snapshot
        region_bounds: Bounding box as (min_lat, max_lat, min_lon, max_lon)
        node_count:    Total nodes in graph
        edge_count:    Total directed edges in graph
        source_apis:   List of data source identifiers used to build snapshot
    """
    snapshot_id:   str
    timestamp:     str
    region_bounds: tuple[float, float, float, float]
    node_count:    int = 0
    edge_count:    int = 0
    source_apis:   list[str] = field(default_factory=list)