"""
graph/builder.py

Assembles EcoNode instances and EdgeDefinitions into a live NetworkX DiGraph.
This is the entry point for all downstream cascade propagation and GNN training.

Workflow:
    1. Register EcoNode instances (from data ingestion or manually)
    2. Call build() to match nodes against EdgeDefinition registry
    3. Edges are only instantiated when:
       - Both source and target node types exist in the graph
       - All conditions on the EdgeDefinition are satisfied
    4. Returns a NetworkX DiGraph with node/edge attributes intact

Usage:
    from graph.builder import GAIAGraphBuilder
    from graph.schema import EcoNode, NodeType

    builder = GAIAGraphBuilder()
    builder.add_node(EcoNode(
        node_id="gulf_hypoxia_2024",
        node_type=NodeType.HYPOXIA_ZONE,
        region="Gulf of Mexico",
        latitude=29.0,
        longitude=-90.0,
        value=0.78,
    ))
    G = builder.build()
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import networkx as nx

from graph.edges import ALL_EDGE_DEFINITIONS, EdgeDefinition
from graph.schema import (
    Domain,
    EcoEdge,
    EcoNode,
    GraphMetadata,
    NODE_DOMAIN,
    NodeType,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Condition evaluator
# ---------------------------------------------------------------------------

def _evaluate_conditions(node: EcoNode, conditions: dict) -> bool:
    """
    Evaluate whether an EcoNode satisfies the activation conditions
    defined on an EdgeDefinition.

    Supported condition keys:
        source_value_min (float): node.value must be >= this threshold
        source_value_max (float): node.value must be <= this threshold
        confidence_min   (float): node.confidence must be >= this threshold

    Returns True if all conditions pass or conditions dict is empty.
    """
    if not conditions:
        return True

    if "source_value_min" in conditions:
        if node.value < conditions["source_value_min"]:
            logger.debug(
                "Edge condition failed: %s value %.3f < source_value_min %.3f",
                node.node_id, node.value, conditions["source_value_min"],
            )
            return False

    if "source_value_max" in conditions:
        if node.value > conditions["source_value_max"]:
            logger.debug(
                "Edge condition failed: %s value %.3f > source_value_max %.3f",
                node.node_id, node.value, conditions["source_value_max"],
            )
            return False

    if "confidence_min" in conditions:
        if node.confidence < conditions["confidence_min"]:
            logger.debug(
                "Edge condition failed: %s confidence %.3f < confidence_min %.3f",
                node.node_id, node.confidence, conditions["confidence_min"],
            )
            return False

    return True


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

class GAIAGraphBuilder:
    """
    Builds and maintains the GAIA ecological DiGraph.

    The builder holds a registry of EcoNode instances indexed by node_id
    and node_type. On build(), it iterates ALL_EDGE_DEFINITIONS and
    instantiates EcoEdge objects for every valid (source, target) node pair
    that satisfies edge conditions.

    Multiple nodes of the same type are supported — e.g. two HYPOXIA_ZONE
    nodes in different regions will each independently drive edges to any
    FISH_POPULATION nodes in their vicinity (proximity filtering optional).
    """

    def __init__(self, proximity_km: Optional[float] = None):
        """
        Args:
            proximity_km: If set, only instantiate edges between nodes
                          within this distance of each other (haversine).
                          None disables proximity filtering.
        """
        self._nodes: dict[str, EcoNode] = {}
        self._nodes_by_type: dict[NodeType, list[EcoNode]] = {t: [] for t in NodeType}
        self.proximity_km = proximity_km

    # ------------------------------------------------------------------
    # Node registration
    # ------------------------------------------------------------------

    def add_node(self, node: EcoNode) -> None:
        """Register a single EcoNode."""
        if node.node_id in self._nodes:
            logger.warning("Node %r already registered — overwriting.", node.node_id)
        self._nodes[node.node_id] = node
        self._nodes_by_type[node.node_type].append(node)
        logger.debug("Registered node: %r", node)

    def add_nodes(self, nodes: list[EcoNode]) -> None:
        """Register multiple EcoNodes."""
        for node in nodes:
            self.add_node(node)

    def remove_node(self, node_id: str) -> None:
        """Deregister a node by ID."""
        if node_id not in self._nodes:
            raise KeyError(f"Node {node_id!r} not found in builder registry.")
        node = self._nodes.pop(node_id)
        self._nodes_by_type[node.node_type].remove(node)

    def update_node_value(self, node_id: str, value: float, confidence: float = 1.0) -> None:
        """Update the value and confidence of a registered node in place."""
        if node_id not in self._nodes:
            raise KeyError(f"Node {node_id!r} not found.")
        self._nodes[node_id].value = value
        self._nodes[node_id].confidence = confidence

    # ------------------------------------------------------------------
    # Proximity filter
    # ------------------------------------------------------------------

    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Return great-circle distance in km between two lat/lon points."""
        import math
        R = 6371.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def _within_proximity(self, source: EcoNode, target: EcoNode) -> bool:
        if self.proximity_km is None:
            return True
        dist = self._haversine_km(
            source.latitude, source.longitude,
            target.latitude, target.longitude,
        )
        return dist <= self.proximity_km

    # ------------------------------------------------------------------
    # Graph assembly
    # ------------------------------------------------------------------

    def build(self, snapshot_id: Optional[str] = None) -> nx.DiGraph:
        """
        Assemble and return a NetworkX DiGraph from registered nodes
        and the global EdgeDefinition registry.

        Each graph node stores the full EcoNode dataclass as attribute "data".
        Each graph edge stores the instantiated EcoEdge as attribute "data"
        plus flattened weight and lag_hours for convenience.

        Returns:
            nx.DiGraph with node attributes {data: EcoNode} and
            edge attributes {data: EcoEdge, weight: float, lag_hours: int}
        """
        G = nx.DiGraph()

        # Add all nodes
        for node in self._nodes.values():
            G.add_node(
                node.node_id,
                data=node,
                domain=node.domain.value,
                node_type=node.node_type.value,
                value=node.value,
                confidence=node.confidence,
                latitude=node.latitude,
                longitude=node.longitude,
                region=node.region,
            )

        edges_added = 0
        edges_skipped_conditions = 0
        edges_skipped_proximity = 0

        for edge_def in ALL_EDGE_DEFINITIONS:
            sources = self._nodes_by_type.get(edge_def.source_type, [])
            targets = self._nodes_by_type.get(edge_def.target_type, [])

            if not sources or not targets:
                continue

            for source in sources:
                # Evaluate activation conditions on the source node
                if not _evaluate_conditions(source, edge_def.conditions):
                    edges_skipped_conditions += len(targets)
                    continue

                for target in targets:
                    if source.node_id == target.node_id:
                        continue

                    # Proximity filter
                    if not self._within_proximity(source, target):
                        edges_skipped_proximity += 1
                        continue

                    eco_edge = EcoEdge(
                        source_id   = source.node_id,
                        target_id   = target.node_id,
                        edge_type   = edge_def.edge_type,
                        weight      = edge_def.base_weight,
                        lag_hours   = edge_def.lag_hours,
                        confidence  = 1.0,
                        cross_domain= edge_def.cross_domain,
                    )

                    G.add_edge(
                        source.node_id,
                        target.node_id,
                        data        = eco_edge,
                        weight      = eco_edge.weight,
                        lag_hours   = eco_edge.lag_hours,
                        edge_type   = eco_edge.edge_type.value,
                        cross_domain= eco_edge.cross_domain,
                    )
                    edges_added += 1

        logger.info(
            "Graph built: %d nodes, %d edges "
            "(%d skipped by conditions, %d by proximity)",
            G.number_of_nodes(), edges_added,
            edges_skipped_conditions, edges_skipped_proximity,
        )

        # Attach metadata
        ts = datetime.now(timezone.utc).isoformat()
        lats = [n.latitude for n in self._nodes.values()]
        lons = [n.longitude for n in self._nodes.values()]
        G.graph["metadata"] = GraphMetadata(
            snapshot_id   = snapshot_id or f"snapshot_{ts}",
            timestamp     = ts,
            region_bounds = (min(lats), max(lats), min(lons), max(lons)) if lats else (0,0,0,0),
            node_count    = G.number_of_nodes(),
            edge_count    = G.number_of_edges(),
            source_apis   = [],
        )

        return G

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def summary(self) -> str:
        lines = [
            f"GAIAGraphBuilder — {len(self._nodes)} nodes registered",
            f"  proximity_filter: {self.proximity_km} km",
        ]
        for ntype, nodes in self._nodes_by_type.items():
            if nodes:
                lines.append(f"  {ntype.value}: {len(nodes)} node(s)")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"GAIAGraphBuilder(nodes={len(self._nodes)}, proximity_km={self.proximity_km})"


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_gulf_mexico_graph() -> nx.DiGraph:
    """
    Builds a minimal example graph seeded with Gulf of Mexico indicator
    nodes for development and testing purposes.
    """
    from graph.schema import EcoNode, NodeType

    builder = GAIAGraphBuilder(proximity_km=800)

    builder.add_nodes([
        EcoNode(
            node_id   = "gulf_hypoxia_2024",
            node_type = NodeType.HYPOXIA_ZONE,
            region    = "Gulf of Mexico",
            latitude  = 29.0,
            longitude = -90.5,
            value     = 0.78,
            metadata  = {"source": "noaa", "area_km2": 15906},
        ),
        EcoNode(
            node_id   = "gulf_menhaden_2024",
            node_type = NodeType.FISH_POPULATION,
            region    = "Gulf of Mexico",
            latitude  = 29.5,
            longitude = -89.8,
            value     = 0.55,
            metadata  = {"species": "Atlantic menhaden", "source": "noaa_fisheries"},
        ),
        EcoNode(
            node_id   = "gulf_algae_2024",
            node_type = NodeType.ALGAE_BLOOM,
            region    = "Gulf of Mexico",
            latitude  = 28.8,
            longitude = -90.2,
            value     = 0.62,
            metadata  = {"source": "nasa_modis", "chlorophyll_mg_m3": 12.4},
        ),
        EcoNode(
            node_id   = "gulf_benthic_2024",
            node_type = NodeType.BENTHIC_HABITAT,
            region    = "Gulf of Mexico",
            latitude  = 29.1,
            longitude = -90.6,
            value     = 0.70,
            metadata  = {"source": "noaa_benthic_survey"},
        ),
        EcoNode(
            node_id   = "mississippi_runoff_2024",
            node_type = NodeType.NUTRIENT_RUNOFF,
            region    = "Mississippi River Delta",
            latitude  = 29.9,
            longitude = -89.9,
            value     = 0.80,
            metadata  = {"source": "usgs_waterwatch", "nitrogen_kg_day": 420000},
        ),
        EcoNode(
            node_id   = "gulf_coast_predators_2024",
            node_type = NodeType.PREDATOR_POPULATION,
            region    = "Gulf Coast",
            latitude  = 30.2,
            longitude = -89.5,
            value     = 0.45,
            metadata  = {"species": "brown pelican / bottlenose dolphin", "source": "inaturalist"},
        ),
    ])

    return builder.build(snapshot_id="gulf_mexico_example")