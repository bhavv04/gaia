"""
graph/edges.py

Encodes empirically-derived causal relationships between ecological indicators.
These edges form the prior knowledge structure of the GAIA graph before
the GNN learns to refine edge weights from observed data.

Each EdgeDefinition captures:
  - The directed causal relationship (source → target node types)
  - A literature-derived base weight and propagation lag
  - The ecological mechanism driving the relationship
  - Source citations for auditability

References:
  [1] Rabalais et al. (2002) - Gulf of Mexico hypoxia and benthic community response
  [2] Breitburg et al. (2018) - Declining oxygen in the global ocean and coastal waters
  [3] Diaz & Rosenberg (2008) - Spreading dead zones and consequences for marine ecosystems
  [4] Scheffer et al. (2001) - Catastrophic shifts in ecosystems (trophic cascades)
  [5] Rabotyagov et al. (2014) - The economics of dead zones
"""

from dataclasses import dataclass, field
from typing import Optional

from graph.schema import EdgeType, NodeType


# ---------------------------------------------------------------------------
# Edge definition structure
# ---------------------------------------------------------------------------

@dataclass
class EdgeDefinition:
    """
    A prototype causal edge between two node types.
    Instantiated into EcoEdge objects by graph/builder.py when
    matching node pairs exist in the graph.

    Attributes:
        source_type:   NodeType of the driving indicator
        target_type:   NodeType of the affected indicator
        edge_type:     Categorical relationship label
        base_weight:   Literature-derived influence strength [0, 1]
                       Higher = stronger causal coupling
        lag_hours:     Expected propagation delay in hours
        mechanism:     Plain-language description of the causal pathway
        citations:     Reference indices from module docstring
        conditions:    Optional dict of threshold conditions that must
                       be met for this edge to activate, e.g.
                       {"source_value_min": 0.4}
    """
    source_type:  NodeType
    target_type:  NodeType
    edge_type:    EdgeType
    base_weight:  float
    lag_hours:    int
    mechanism:    str
    citations:    list[int]         = field(default_factory=list)
    conditions:   dict              = field(default_factory=dict)

    @property
    def cross_domain(self) -> bool:
        from graph.schema import NODE_DOMAIN
        return NODE_DOMAIN[self.source_type] != NODE_DOMAIN[self.target_type]


# ---------------------------------------------------------------------------
# Marine internal edges
# ---------------------------------------------------------------------------

MARINE_EDGES: list[EdgeDefinition] = [
    EdgeDefinition(
        source_type = NodeType.HYPOXIA_ZONE,
        target_type = NodeType.FISH_POPULATION,
        edge_type   = EdgeType.HYPOXIA_DRIVES_FISH_DISPLACEMENT,
        base_weight = 0.85,
        lag_hours   = 72,
        mechanism   = (
            "Dissolved oxygen depletion below 2 mg/L creates uninhabitable benthic "
            "conditions, forcing demersal fish into shallower pelagic zones. "
            "Biomass in affected grid cells drops as fish redistribute."
        ),
        citations   = [2, 3],
        conditions  = {"source_value_min": 0.4},  # only triggers above moderate hypoxia
    ),
    EdgeDefinition(
        source_type = NodeType.FISH_POPULATION,
        target_type = NodeType.ALGAE_BLOOM,
        edge_type   = EdgeType.FISH_GRAZING_CONTROLS_ALGAE,
        base_weight = 0.60,
        lag_hours   = 168,  # ~1 week
        mechanism   = (
            "Planktivorous fish suppress phytoplankton and zooplankton grazers. "
            "When fish populations decline or displace, grazing pressure drops "
            "and algae proliferate unchecked."
        ),
        citations   = [4],
    ),
    EdgeDefinition(
        source_type = NodeType.ALGAE_BLOOM,
        target_type = NodeType.HYPOXIA_ZONE,
        edge_type   = EdgeType.ALGAE_BLOOM_DEPLETES_OXYGEN,
        base_weight = 0.80,
        lag_hours   = 120,  # ~5 days decomposition lag
        mechanism   = (
            "Algal biomass sinks and decomposes via bacterial respiration, "
            "consuming dissolved oxygen in bottom waters and expanding or "
            "intensifying existing hypoxic zones. Core positive feedback loop."
        ),
        citations   = [1, 2],
    ),
    EdgeDefinition(
        source_type = NodeType.HYPOXIA_ZONE,
        target_type = NodeType.BENTHIC_HABITAT,
        edge_type   = EdgeType.HYPOXIA_DEGRADES_BENTHIC,
        base_weight = 0.90,
        lag_hours   = 240,  # ~10 days for community-level response
        mechanism   = (
            "Prolonged hypoxia causes mass mortality of benthic invertebrates "
            "(polychaetes, bivalves, crustaceans), degrading seafloor habitat "
            "complexity and reducing food availability for demersal species."
        ),
        citations   = [1, 3],
        conditions  = {"source_value_min": 0.5},
    ),
    EdgeDefinition(
        source_type = NodeType.OCEAN_TEMPERATURE,
        target_type = NodeType.ALGAE_BLOOM,
        edge_type   = EdgeType.SST_ANOMALY_TRIGGERS_BLOOM,
        base_weight = 0.65,
        lag_hours   = 96,
        mechanism   = (
            "Positive SST anomalies increase thermal stratification, trapping "
            "nutrients in the photic zone and accelerating cyanobacterial and "
            "dinoflagellate growth rates."
        ),
        citations   = [2],
    ),
]


# ---------------------------------------------------------------------------
# Cross-domain: marine → terrestrial
# ---------------------------------------------------------------------------

MARINE_TO_TERRESTRIAL_EDGES: list[EdgeDefinition] = [
    EdgeDefinition(
        source_type = NodeType.FISH_POPULATION,
        target_type = NodeType.PREDATOR_POPULATION,
        edge_type   = EdgeType.FISH_DISPLACEMENT_SHIFTS_PREDATOR,
        base_weight = 0.55,
        lag_hours   = 336,  # ~2 weeks for predator behavioural response
        mechanism   = (
            "Coastal apex predators (osprey, brown pelican, bottlenose dolphin) "
            "that rely on demersal fish concentrate in areas of displaced fish "
            "aggregations, altering foraging territory and population dynamics "
            "in adjacent terrestrial/coastal zones."
        ),
        citations   = [4],
    ),
    EdgeDefinition(
        source_type = NodeType.BENTHIC_HABITAT,
        target_type = NodeType.PREDATOR_POPULATION,
        edge_type   = EdgeType.BENTHIC_LOSS_REDUCES_SHOREBIRD,
        base_weight = 0.45,
        lag_hours   = 720,  # ~1 month, slower community shift
        mechanism   = (
            "Shorebird species dependent on benthic invertebrates (sandpipers, "
            "dunlins, knots) experience reduced foraging success during hypoxic "
            "events, affecting breeding success and local population counts."
        ),
        citations   = [3],
    ),
]


# ---------------------------------------------------------------------------
# Cross-domain: freshwater → marine
# ---------------------------------------------------------------------------

FRESHWATER_TO_MARINE_EDGES: list[EdgeDefinition] = [
    EdgeDefinition(
        source_type = NodeType.NUTRIENT_RUNOFF,
        target_type = NodeType.ALGAE_BLOOM,
        edge_type   = EdgeType.NUTRIENT_RUNOFF_FEEDS_BLOOM,
        base_weight = 0.88,
        lag_hours   = 48,
        mechanism   = (
            "Nitrogen (NO3-) and phosphorus (PO4-) from agricultural runoff "
            "enter coastal waters via river discharge, providing the primary "
            "nutrient subsidy that drives eutrophication and algal blooms. "
            "Mississippi River discharge is the dominant driver of Gulf hypoxia."
        ),
        citations   = [1, 5],
    ),
    EdgeDefinition(
        source_type = NodeType.NUTRIENT_RUNOFF,
        target_type = NodeType.HYPOXIA_ZONE,
        edge_type   = EdgeType.NUTRIENT_RUNOFF_DRIVES_HYPOXIA,
        base_weight = 0.82,
        lag_hours   = 168,
        mechanism   = (
            "Excess nutrients stimulate phytoplankton blooms whose decomposition "
            "consumes bottom-water oxygen. Runoff volume in spring correlates "
            "directly with summer dead zone extent."
        ),
        citations   = [1, 5],
    ),
]


# ---------------------------------------------------------------------------
# Cross-domain: atmospheric → marine / freshwater
# ---------------------------------------------------------------------------

ATMOSPHERIC_EDGES: list[EdgeDefinition] = [
    EdgeDefinition(
        source_type = NodeType.PRECIPITATION,
        target_type = NodeType.NUTRIENT_RUNOFF,
        edge_type   = EdgeType.PRECIPITATION_INCREASES_RUNOFF,
        base_weight = 0.75,
        lag_hours   = 24,
        mechanism   = (
            "Heavy precipitation events increase river discharge volume and "
            "surface runoff velocity, mobilising nitrogen and phosphorus from "
            "agricultural soils into waterways at accelerated rates."
        ),
        citations   = [5],
    ),
    EdgeDefinition(
        source_type = NodeType.CO2_CONCENTRATION,
        target_type = NodeType.OCEAN_TEMPERATURE,
        edge_type   = EdgeType.CO2_DRIVES_OCEAN_TEMPERATURE,
        base_weight = 0.70,
        lag_hours   = 8760,  # ~1 year, slow forcing
        mechanism   = (
            "Elevated atmospheric CO2 drives radiative forcing that increases "
            "sea surface temperatures over seasonal to annual timescales, "
            "amplifying stratification and bloom conditions."
        ),
        citations   = [2],
    ),
]


# ---------------------------------------------------------------------------
# Cross-domain: terrestrial → freshwater
# ---------------------------------------------------------------------------

TERRESTRIAL_TO_FRESHWATER_EDGES: list[EdgeDefinition] = [
    EdgeDefinition(
        source_type = NodeType.SOIL_HEALTH,
        target_type = NodeType.NUTRIENT_RUNOFF,
        edge_type   = EdgeType.SOIL_DEGRADATION_INCREASES_RUNOFF,
        base_weight = 0.65,
        lag_hours   = 72,
        mechanism   = (
            "Degraded soils with low organic matter and poor structure have "
            "reduced nutrient retention capacity, increasing leaching of "
            "nitrogen and phosphorus into groundwater and surface runoff."
        ),
        citations   = [5],
    ),
    EdgeDefinition(
        source_type = NodeType.VEGETATION_INDEX,
        target_type = NodeType.NUTRIENT_RUNOFF,
        edge_type   = EdgeType.VEGETATION_LOSS_INCREASES_RUNOFF,
        base_weight = 0.60,
        lag_hours   = 48,
        mechanism   = (
            "Vegetation loss reduces root uptake of soil nutrients and increases "
            "surface runoff velocity. Riparian buffer loss is especially "
            "impactful as it removes the last interception layer before waterways."
        ),
        citations   = [5],
    ),
]


# ---------------------------------------------------------------------------
# Master edge registry
# ---------------------------------------------------------------------------

ALL_EDGE_DEFINITIONS: list[EdgeDefinition] = (
    MARINE_EDGES
    + MARINE_TO_TERRESTRIAL_EDGES
    + FRESHWATER_TO_MARINE_EDGES
    + ATMOSPHERIC_EDGES
    + TERRESTRIAL_TO_FRESHWATER_EDGES
)

# Cross-domain edges only 
CROSS_DOMAIN_EDGES: list[EdgeDefinition] = [
    e for e in ALL_EDGE_DEFINITIONS if e.cross_domain
]


#what is this 
def get_edges_from(source_type: NodeType) -> list[EdgeDefinition]:
    """Return all edge definitions originating from a given node type."""
    return [e for e in ALL_EDGE_DEFINITIONS if e.source_type == source_type]


def get_edges_to(target_type: NodeType) -> list[EdgeDefinition]:
    """Return all edge definitions targeting a given node type."""
    return [e for e in ALL_EDGE_DEFINITIONS if e.target_type == target_type]