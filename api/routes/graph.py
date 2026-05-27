"""
api/routes/graph.py

Graph inspection endpoints — returns current node values and
graph structure without running a full cascade prediction.

GET /graph/{region}  — full graph snapshot (nodes + edges)
GET /graph/{region}/nodes — nodes only
"""

import logging
from typing import Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel

from data.ingestion.inaturalist import fetch_all_inaturalist_nodes
from data.ingestion.nasa import fetch_all_nasa_nodes
from data.ingestion.noaa import fetch_all_noaa_nodes
from data.ingestion.openaq import fetch_all_openaq_nodes
from graph.builder import GAIAGraphBuilder

logger = logging.getLogger(__name__)
router = APIRouter()


class GraphSnapshotResponse(BaseModel):
    region:      str
    snapshot_id: str
    node_count:  int
    edge_count:  int
    nodes:       list[dict]
    edges:       list[dict]


@router.get("/{region}", response_model=GraphSnapshotResponse, tags=["graph"])
async def get_graph_snapshot(
    region:       str,
    lat:          float = Query(...),
    lon:          float = Query(...),
    proximity_km: float = Query(default=800.0),
    year:         Optional[int] = Query(default=None),
):
    """Return the current ecological graph snapshot for a region."""
    nodes = []
    for fetcher, kwargs in [
        (fetch_all_noaa_nodes,        {"lat": lat, "lon": lon, "region": region, "year": year}),
        (fetch_all_openaq_nodes,      {"lat": lat, "lon": lon, "region": region}),
        (fetch_all_nasa_nodes,        {"lat": lat, "lon": lon, "region": region, "year": year}),
        (fetch_all_inaturalist_nodes, {"lat": lat, "lon": lon, "region": region}),
    ]:
        try:
            nodes.extend(fetcher(**kwargs))
        except Exception as e:
            logger.warning("Ingestion failed: %s", e)

    builder = GAIAGraphBuilder(proximity_km=proximity_km)
    builder.add_nodes(nodes)
    G    = builder.build()
    meta = G.graph.get("metadata")

    node_list = [
        {
            "node_id":    G.nodes[n]["data"].node_id,
            "node_type":  G.nodes[n]["data"].node_type.value,
            "domain":     G.nodes[n]["data"].domain.value,
            "value":      round(G.nodes[n]["data"].value, 4),
            "confidence": round(G.nodes[n]["data"].confidence, 4),
            "latitude":   G.nodes[n]["data"].latitude,
            "longitude":  G.nodes[n]["data"].longitude,
        }
        for n in G.nodes()
    ]

    edge_list = [
        {
            "source":      u,
            "target":      v,
            "edge_type":   d.get("edge_type"),
            "weight":      round(d.get("weight", 1.0), 4),
            "lag_hours":   d.get("lag_hours", 0),
            "cross_domain":d.get("cross_domain", False),
        }
        for u, v, d in G.edges(data=True)
    ]

    return GraphSnapshotResponse(
        region      = region,
        snapshot_id = meta.snapshot_id if meta else "unknown",
        node_count  = G.number_of_nodes(),
        edge_count  = G.number_of_edges(),
        nodes       = node_list,
        edges       = edge_list,
    )