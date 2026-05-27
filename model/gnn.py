"""
model/gnn.py

Graph Neural Network for ecological cascade failure prediction.

Architecture: Temporal Graph Convolutional Network (T-GCN)
  - Graph Attention Network (GAT) layers for spatial message passing
  - GRU cells for temporal state propagation
  - Per-node regression head predicting indicator value at t+lag

The model takes a snapshot graph at time t and predicts the value
of each node at t + horizon_hours, effectively simulating how a
disturbance propagates through the ecological graph over time.

Input:
    - Node features: [value, confidence, domain_onehot (4), node_type_onehot (13)]
    - Edge features: [weight, lag_hours_normalised, cross_domain]

Output:
    - Per-node predicted value at t + horizon_hours (regression, [0, 1])
    - Per-node cascade probability (binary: does this node get significantly affected?)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, global_mean_pool


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_DOMAINS    = 4   # marine, atmospheric, terrestrial, freshwater
NUM_NODE_TYPES = 13  # matches NodeType enum length in schema.py
EDGE_FEAT_DIM  = 3   # weight, lag_hours_norm, cross_domain

# Base node feature dim: value + confidence + domain onehot + type onehot
NODE_FEAT_DIM  = 2 + NUM_DOMAINS + NUM_NODE_TYPES  # = 19


# ---------------------------------------------------------------------------
# Temporal Graph Attention layer
# ---------------------------------------------------------------------------

class TemporalGATLayer(nn.Module):
    """
    Single layer combining GATv2 spatial message passing with a GRU
    cell for temporal state updates.

    The GRU hidden state carries information about how node values
    have evolved over previous timesteps, allowing the model to
    distinguish a node that is newly stressed from one in prolonged decline.
    """

    def __init__(self, in_channels: int, out_channels: int, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.gat = GATv2Conv(
            in_channels  = in_channels,
            out_channels = out_channels,
            heads        = heads,
            dropout      = dropout,
            edge_dim     = EDGE_FEAT_DIM,
            concat       = True,
        )
        self.gru = nn.GRUCell(
            input_size  = out_channels * heads,
            hidden_size = out_channels * heads,
        )
        self.norm = nn.LayerNorm(out_channels * heads)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, h: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            x:          Node features [N, in_channels]
            edge_index: Graph connectivity [2, E]
            edge_attr:  Edge features [E, EDGE_FEAT_DIM]
            h:          GRU hidden state [N, out_channels * heads]

        Returns:
            out: Updated node embeddings [N, out_channels * heads]
            h:   Updated GRU hidden state [N, out_channels * heads]
        """
        spatial = self.gat(x, edge_index, edge_attr)   # [N, out_channels * heads]
        spatial = self.dropout(F.elu(spatial))
        h_new   = self.gru(spatial, h)                 # temporal update
        out     = self.norm(h_new)
        return out, h_new


# ---------------------------------------------------------------------------
# GAIA GNN
# ---------------------------------------------------------------------------

class GAIANet(nn.Module):
    """
    Full cascade prediction network.

    Takes a single graph snapshot and outputs per-node predictions for:
      1. predicted_value:  Continuous indicator value at t + horizon [0, 1]
      2. cascade_prob:     Probability node crosses cascade threshold (>0.6 value)

    Architecture:
        Input projection → 2x TemporalGATLayer → Cascade head + Regression head

    Args:
        hidden_dim:         Hidden embedding dimension per attention head
        heads:              Number of GAT attention heads
        num_layers:         Number of TemporalGAT layers (2 recommended)
        dropout:            Dropout rate
        cascade_threshold:  Value above which a node is considered "in cascade"
    """

    def __init__(
        self,
        hidden_dim:         int   = 64,
        heads:              int   = 4,
        num_layers:         int   = 2,
        dropout:            float = 0.2,
        cascade_threshold:  float = 0.6,
    ):
        super().__init__()
        self.hidden_dim        = hidden_dim
        self.heads             = heads
        self.num_layers        = num_layers
        self.cascade_threshold = cascade_threshold
        self.embed_dim         = hidden_dim * heads

        # Project raw node features into hidden space
        self.input_proj = nn.Sequential(
            nn.Linear(NODE_FEAT_DIM, self.embed_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        # Stack of temporal GAT layers
        self.tgat_layers = nn.ModuleList([
            TemporalGATLayer(
                in_channels  = self.embed_dim,
                out_channels = hidden_dim,
                heads        = heads,
                dropout      = dropout,
            )
            for _ in range(num_layers)
        ])

        # Regression head: predicts indicator value at t + horizon
        self.regression_head = nn.Sequential(
            nn.Linear(self.embed_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),   # output in [0, 1]
        )

        # Cascade head: binary probability this node enters cascade
        self.cascade_head = nn.Sequential(
            nn.Linear(self.embed_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, data: Data, h_states: list[Tensor] | None = None) -> dict[str, Tensor]:
        """
        Forward pass.

        Args:
            data:     PyG Data object with x, edge_index, edge_attr
            h_states: Optional list of GRU hidden states per layer,
                      one tensor of shape [N, embed_dim] per layer.
                      Pass None to initialise with zeros (first timestep).

        Returns:
            dict with keys:
                predicted_value  [N, 1]  — indicator value at t + horizon
                cascade_prob     [N, 1]  — cascade probability per node
                hidden_states    list    — updated GRU states for next timestep
        """
        x          = data.x           # [N, NODE_FEAT_DIM]
        edge_index = data.edge_index  # [2, E]
        edge_attr  = data.edge_attr   # [E, EDGE_FEAT_DIM]

        # Initialise hidden states if not provided
        N = x.size(0)
        if h_states is None:
            h_states = [torch.zeros(N, self.embed_dim, device=x.device)] * self.num_layers

        x = self.input_proj(x)

        new_h_states = []
        for i, layer in enumerate(self.tgat_layers):
            x, h_new = layer(x, edge_index, edge_attr, h_states[i])
            new_h_states.append(h_new)

        predicted_value = self.regression_head(x)   # [N, 1]
        cascade_prob    = self.cascade_head(x)       # [N, 1]

        return {
            "predicted_value": predicted_value,
            "cascade_prob":    cascade_prob,
            "hidden_states":   new_h_states,
        }

    def predict_cascade_sequence(
        self,
        data:       Data,
        steps:      int = 5,
        threshold:  float | None = None,
    ) -> list[dict]:
        """
        Roll the model forward for `steps` timesteps, propagating
        predicted values back into the graph at each step.

        This is the core cascade simulation — each step's output
        becomes the next step's input, letting disturbances ripple
        through the graph iteratively.

        Args:
            data:      Initial graph snapshot
            steps:     Number of forward steps to simulate
            threshold: Cascade threshold override (default: self.cascade_threshold)

        Returns:
            List of step dicts, each containing:
                step:            int
                predicted_value: Tensor [N, 1]
                cascade_prob:    Tensor [N, 1]
                cascading_nodes: list of node indices above threshold
        """
        threshold  = threshold or self.cascade_threshold
        h_states   = None
        sequence   = []
        current_x  = data.x.clone()

        self.eval()
        with torch.no_grad():
            for step in range(steps):
                step_data         = data.clone()
                step_data.x       = current_x
                out               = self.forward(step_data, h_states)
                h_states          = out["hidden_states"]

                pv   = out["predicted_value"]
                cp   = out["cascade_prob"]
                mask = (cp.squeeze() >= threshold).nonzero(as_tuple=True)[0].tolist()

                sequence.append({
                    "step":            step + 1,
                    "predicted_value": pv,
                    "cascade_prob":    cp,
                    "cascading_nodes": mask,
                })

                # Feed predicted values back as next input
                # Update only the value feature (index 0) in node features
                current_x = current_x.clone()
                current_x[:, 0] = pv.squeeze().clamp(0, 1)

        return sequence


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class GAIALoss(nn.Module):
    """
    Combined loss for joint regression + cascade classification.

    L = alpha * MSE(predicted_value, true_value)
      + (1 - alpha) * BCE(cascade_prob, cascade_label)

    cascade_label is derived automatically from true_value >= cascade_threshold.
    """

    def __init__(self, alpha: float = 0.6, cascade_threshold: float = 0.6):
        super().__init__()
        self.alpha             = alpha
        self.cascade_threshold = cascade_threshold
        self.mse               = nn.MSELoss()
        self.bce               = nn.BCELoss()

    def forward(
        self,
        predicted_value: Tensor,
        cascade_prob:    Tensor,
        true_value:      Tensor,
    ) -> dict[str, Tensor]:
        cascade_label = (true_value >= self.cascade_threshold).float()

        loss_regression = self.mse(predicted_value, true_value)
        loss_cascade    = self.bce(cascade_prob, cascade_label)
        total           = self.alpha * loss_regression + (1 - self.alpha) * loss_cascade

        return {
            "total":      total,
            "regression": loss_regression,
            "cascade":    loss_cascade,
        }