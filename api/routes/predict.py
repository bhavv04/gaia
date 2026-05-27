"""
api/routes/predict.py

Core cascade prediction endpoint.

POST /predict
  Body: PredictRequest
  Returns: CascadeResponse

The endpoint:
  1. Fetches live EcoNode data for the requested region from all ingestion sources
  2. Builds a GAIA ecological graph via GAIAGraphBuilder
  3. Converts graph to PyG Data object
  4. Runs GAIANet.predict_cascade_sequence() for N steps
  5. Returns structured cascade prediction with per-node confidence and timing
"""

import logging
import time
from typing import Optional

import networkx as nx
import torch
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

from data.ingestion.inaturalist import fetch_all_inaturalist_nodes
from data.ingestion.nasa import fetch_all_nasa_nodes
from data.ingestion.noaa import fetch_all_noaa_nodes
from data.ingestion.openaq import fetch_all_openaq_nodes
from graph.builder import GAIAGraphBuilder
from graph.schema import NODE_DOMAIN
from model.gnn import NODE_FEAT_DIM, NUM_DOMAINS, NUM_NODE_TYPES, NodeType

logger = logging.getLogger(__name__)
router = APIRouter()

# Domain and node type index maps for feature encoding
DOMAIN_INDEX    = {d: i for i, d in enumerate(["marine", "atmospheric", "terrestrial", "freshwater"])}
NODETYPE_INDEX  = {nt: i for i, nt in enumerate(NodeType)}


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    lat:          float = Field(..., description="Region centroid latitude",  example=29.0)
    lon:          float = Field(..., description="Region centroid longitude", example=-90.5)
    region:       str   = Field(..., description="Human-readable region name", example="Gulf of Mexico")
    steps:        int   = Field(default=5, ge=1, le=20, description="Cascade simulation steps")
    proximity_km: float = Field(default=800.0, description="Max edge distance in km")
    year:         Optional[int] = Field(default=None, description="Data year (defaults to current)")


class NodeState(BaseModel):
    node_id:      str
    node_type:    str
    domain:       str
    region:       str
    latitude:     float
    longitude:    float
    current_value:float
    confidence:   float


class CascadeStep(BaseModel):
    step:             int
    cascading_nodes:  list[str]         # node_ids that crossed cascade threshold
    node_predictions: dict[str, float]  # node_id → predicted value
    node_cascade_probs: dict[str, float] # node_id → cascade probability


class CascadeResponse(BaseModel):
    region:           str
    snapshot_id:      str
    nodes:            list[NodeState]
    edges:            list[dict]
    cascade_sequence: list[CascadeStep]
    fetch_time_ms:    float
    predict_time_ms:  float
    model_loaded:     bool


# ---------------------------------------------------------------------------
# Graph → PyG conversion
# ---------------------------------------------------------------------------

def _encode_node_features(G: nx.DiGraph) -> torch.Tensor:
    """
    Encode EcoNode attributes into a fixed-size feature tensor.
    Layout: [value, confidence, domain_onehot (4), node_type_onehot (13)]
    Total: NODE_FEAT_DIM = 19
    """
    features = []
    for node_id in G.nodes():
        data    = G.nodes[node_id]
        eco     = data["data"]

        value      = eco.value
        confidence = eco.confidence

        domain_vec = [0.0] * NUM_DOMAINS
        domain_idx = DOMAIN_INDEX.get(eco.domain.value, 0)
        domain_vec[domain_idx] = 1.0

        type_vec  = [0.0] * NUM_NODE_TYPES
        type_idx  = NODETYPE_INDEX.get(eco.node_type, 0)
        type_vec[type_idx] = 1.0

        features.append([value, confidence] + domain_vec + type_vec)

    return torch.tensor(features, dtype=torch.float)


def _encode_edge_features(G: nx.DiGraph) -> torch.Tensor:
    """
    Encode EcoEdge attributes into edge feature tensor.
    Layout: [weight, lag_hours_norm, cross_domain]
    """
    MAX_LAG = 8760.0  # normalise against 1 year
    features = []
    for u, v, data in G.edges(data=True):
        weight      = data.get("weight", 1.0)
        lag_norm    = min(1.0, data.get("lag_hours", 0) / MAX_LAG)
        cross       = float(data.get("cross_domain", False))
        features.append([weight, lag_norm, cross])

    if not features:
        return torch.zeros((0, 3), dtype=torch.float)
    return torch.tensor(features, dtype=torch.float)


def networkx_to_pyg(G: nx.DiGraph) -> tuple[Data, list[str]]:
    """
    Convert a GAIA NetworkX DiGraph to a PyG Data object.

    Returns:
        data:     PyG Data with x, edge_index, edge_attr
        node_ids: Ordered list of node_ids matching tensor row indices
    """
    node_ids  = list(G.nodes())
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    x          = _encode_node_features(G)
    edge_attr  = _encode_edge_features(G)

    if G.number_of_edges() > 0:
        edges      = list(G.edges())
        edge_index = torch.tensor(
            [[id_to_idx[u], id_to_idx[v]] for u, v in edges],
            dtype=torch.long,
        ).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr), node_ids


# ---------------------------------------------------------------------------
# Predict endpoint
# ---------------------------------------------------------------------------

@router.post("", response_model=CascadeResponse, tags=["prediction"])
async def predict_cascade(req: PredictRequest, request: Request):
    model        = getattr(request.app.state, "model", None)
    device       = getattr(request.app.state, "device", torch.device("cpu"))
    model_loaded = model is not None

    # ------------------------------------------------------------------
    # 1. Fetch live data from all ingestion sources
    # ------------------------------------------------------------------
    t0 = time.time()
    nodes = []
    for fetcher, kwargs in [
        (fetch_all_noaa_nodes,        {"lat": req.lat, "lon": req.lon, "region": req.region, "year": req.year}),
        (fetch_all_openaq_nodes,      {"lat": req.lat, "lon": req.lon, "region": req.region}),
        (fetch_all_nasa_nodes,        {"lat": req.lat, "lon": req.lon, "region": req.region, "year": req.year}),
        (fetch_all_inaturalist_nodes, {"lat": req.lat, "lon": req.lon, "region": req.region}),
    ]:
        try:
            fetched = fetcher(**kwargs)
            nodes.extend(fetched)
        except Exception as e:
            logger.warning("Ingestion source failed: %s", e)

    if not nodes:
        raise HTTPException(status_code=503, detail="All data sources failed — no nodes available")

    fetch_time_ms = (time.time() - t0) * 1000
    logger.info("Fetched %d nodes in %.0fms for region=%s", len(nodes), fetch_time_ms, req.region)

    # ------------------------------------------------------------------
    # 2. Build ecological graph
    # ------------------------------------------------------------------
    builder = GAIAGraphBuilder(proximity_km=req.proximity_km)
    builder.add_nodes(nodes)
    G = builder.build(snapshot_id=f"{req.region}_{req.year or 'live'}")

    meta = G.graph.get("metadata")

    # ------------------------------------------------------------------
    # 3. Convert to PyG and run prediction
    # ------------------------------------------------------------------
    t1 = time.time()
    data, node_ids = networkx_to_pyg(G)
    data = data.to(device)

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train.py first.")

    cascade_sequence_raw = model.predict_cascade_sequence(data, steps=req.steps)
    predict_time_ms = (time.time() - t1) * 1000

    # ------------------------------------------------------------------
    # 4. Serialise response
    # ------------------------------------------------------------------
    node_states = [
        NodeState(
            node_id       = n.node_id,
            node_type     = n.node_type.value,
            domain        = n.domain.value,
            region        = n.region,
            latitude      = n.latitude,
            longitude     = n.longitude,
            current_value = round(n.value, 4),
            confidence    = round(n.confidence, 4),
        )
        for n in nodes
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

    cascade_steps = []
    for step_data in cascade_sequence_raw:
        pv   = step_data["predicted_value"].squeeze().cpu().tolist()
        cp   = step_data["cascade_prob"].squeeze().cpu().tolist()
        mask = step_data["cascading_nodes"]

        # Handle single-node graphs
        if isinstance(pv, float):
            pv = [pv]
        if isinstance(cp, float):
            cp = [cp]

        cascade_steps.append(CascadeStep(
            step             = step_data["step"],
            cascading_nodes  = [node_ids[i] for i in mask if i < len(node_ids)],
            node_predictions = {node_ids[i]: round(float(pv[i]), 4) for i in range(len(node_ids))},
            node_cascade_probs = {node_ids[i]: round(float(cp[i]), 4) for i in range(len(node_ids))},
        ))

    return CascadeResponse(
        region           = req.region,
        snapshot_id      = meta.snapshot_id if meta else "unknown",
        nodes            = node_states,
        edges            = edge_list,
        cascade_sequence = cascade_steps,
        fetch_time_ms    = round(fetch_time_ms, 1),
        predict_time_ms  = round(predict_time_ms, 1),
        model_loaded     = model_loaded,
    )