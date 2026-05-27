"""
model/train.py

Training loop for GAIANet with MLflow experiment tracking.

Handles:
  - Dataset construction from graph snapshots
  - Train / validation split
  - Training loop with early stopping
  - MLflow logging of metrics, params, and model artifacts
  - Checkpoint saving

Usage:
    python -m model.train --snapshots data/processed/snapshots.pt --epochs 100
"""

import argparse
import logging
import os
import random
from pathlib import Path
from typing import Optional

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

from model.gnn import GAIALoss, GAIANet

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CascadeSnapshotDataset(Dataset):
    """
    Dataset of (graph_t, graph_t+horizon) snapshot pairs.

    Each item is a PyG Data object representing the graph at time t,
    with target node values y representing ground truth at t + horizon.

    Expects a list of PyG Data objects saved as a .pt file where
    consecutive snapshots are temporally ordered.

    Args:
        snapshot_path: Path to .pt file containing list[Data]
        horizon:       Number of snapshot steps ahead to predict
        transform:     Optional PyG transform
    """

    def __init__(self, snapshot_path: str, horizon: int = 1, transform=None):
        super().__init__(transform=transform)
        self.horizon = horizon
        snapshots: list[Data] = torch.load(snapshot_path, weights_only=False)
        # Build (input, target) pairs
        self.pairs: list[tuple[Data, Data]] = [
            (snapshots[i], snapshots[i + horizon])
            for i in range(len(snapshots) - horizon)
        ]
        logger.info(
            "Loaded %d snapshot pairs from %s (horizon=%d)",
            len(self.pairs), snapshot_path, horizon,
        )

    def len(self) -> int:
        return len(self.pairs)

    def get(self, idx: int) -> Data:
        input_graph, target_graph = self.pairs[idx]
        data       = input_graph.clone()
        data.y     = target_graph.x[:, 0].unsqueeze(1)  # target = value feature at t+horizon
        return data


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def train_epoch(
    model:      GAIANet,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer,
    criterion:  GAIALoss,
    device:     torch.device,
) -> dict[str, float]:
    model.train()
    total, reg, cas = 0.0, 0.0, 0.0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out    = model(batch)
        losses = criterion(out["predicted_value"], out["cascade_prob"], batch.y)
        losses["total"].backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total += losses["total"].item()
        reg   += losses["regression"].item()
        cas   += losses["cascade"].item()

    n = len(loader)
    return {"loss": total / n, "regression_loss": reg / n, "cascade_loss": cas / n}


@torch.no_grad()
def eval_epoch(
    model:     GAIANet,
    loader:    DataLoader,
    criterion: GAIALoss,
    device:    torch.device,
) -> dict[str, float]:
    model.eval()
    total, reg, cas = 0.0, 0.0, 0.0
    mae_sum = 0.0

    for batch in loader:
        batch  = batch.to(device)
        out    = model(batch)
        losses = criterion(out["predicted_value"], out["cascade_prob"], batch.y)

        total   += losses["total"].item()
        reg     += losses["regression"].item()
        cas     += losses["cascade"].item()
        mae_sum += torch.mean(torch.abs(out["predicted_value"] - batch.y)).item()

    n = len(loader)
    return {
        "val_loss":            total / n,
        "val_regression_loss": reg / n,
        "val_cascade_loss":    cas / n,
        "val_mae":             mae_sum / n,
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    snapshot_path:  str,
    epochs:         int         = 100,
    lr:             float       = 1e-3,
    hidden_dim:     int         = 64,
    heads:          int         = 4,
    num_layers:     int         = 2,
    dropout:        float       = 0.2,
    alpha:          float       = 0.6,
    horizon:        int         = 1,
    batch_size:     int         = 16,
    val_split:      float       = 0.2,
    patience:       int         = 15,
    checkpoint_dir: str         = "model/experiments",
    seed:           int         = 42,
    run_name:       Optional[str] = None,
) -> GAIANet:

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on %s", device)

    # Dataset
    dataset   = CascadeSnapshotDataset(snapshot_path, horizon=horizon)
    val_size  = int(len(dataset) * val_split)
    train_size= len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    # Model
    model = GAIANet(
        hidden_dim  = hidden_dim,
        heads       = heads,
        num_layers  = num_layers,
        dropout     = dropout,
    ).to(device)

    optimizer  = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler  = ReduceLROnPlateau(optimizer, patience=7, factor=0.5, verbose=True)
    criterion  = GAIALoss(alpha=alpha)

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_val_loss  = float("inf")
    epochs_no_improve = 0
    best_model_path   = Path(checkpoint_dir) / "best_model.pt"

    # MLflow
    mlflow.set_experiment("gaia_cascade_prediction")
    with mlflow.start_run(run_name=run_name or f"gaia_h{horizon}_hd{hidden_dim}"):
        mlflow.log_params({
            "epochs":     epochs,
            "lr":         lr,
            "hidden_dim": hidden_dim,
            "heads":      heads,
            "num_layers": num_layers,
            "dropout":    dropout,
            "alpha":      alpha,
            "horizon":    horizon,
            "batch_size": batch_size,
            "seed":       seed,
        })

        for epoch in range(1, epochs + 1):
            train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
            val_metrics   = eval_epoch(model, val_loader, criterion, device)

            scheduler.step(val_metrics["val_loss"])
            mlflow.log_metrics({**train_metrics, **val_metrics}, step=epoch)

            logger.info(
                "Epoch %3d | loss %.4f | val_loss %.4f | val_mae %.4f",
                epoch,
                train_metrics["loss"],
                val_metrics["val_loss"],
                val_metrics["val_mae"],
            )

            # Checkpoint best model
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                epochs_no_improve = 0
                torch.save(model.state_dict(), best_model_path)
                logger.info("  ✓ New best model saved (val_loss=%.4f)", best_val_loss)
            else:
                epochs_no_improve += 1

            # Early stopping
            if epochs_no_improve >= patience:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch, patience)
                break

        # Load best weights and log to MLflow
        model.load_state_dict(torch.load(best_model_path, weights_only=True))
        mlflow.pytorch.log_model(model, artifact_path="gaia_model")
        mlflow.log_metric("best_val_loss", best_val_loss)
        logger.info("Training complete. Best val_loss: %.4f", best_val_loss)

    return model


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GAIANet cascade prediction model")
    parser.add_argument("--snapshots",      type=str,   required=True,       help="Path to snapshot .pt file")
    parser.add_argument("--epochs",         type=int,   default=100)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--hidden-dim",     type=int,   default=64)
    parser.add_argument("--heads",          type=int,   default=4)
    parser.add_argument("--num-layers",     type=int,   default=2)
    parser.add_argument("--dropout",        type=float, default=0.2)
    parser.add_argument("--alpha",          type=float, default=0.6)
    parser.add_argument("--horizon",        type=int,   default=1)
    parser.add_argument("--batch-size",     type=int,   default=16)
    parser.add_argument("--patience",       type=int,   default=15)
    parser.add_argument("--checkpoint-dir", type=str,   default="model/experiments")
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--run-name",       type=str,   default=None)
    args = parser.parse_args()

    train(
        snapshot_path  = args.snapshots,
        epochs         = args.epochs,
        lr             = args.lr,
        hidden_dim     = args.hidden_dim,
        heads          = args.heads,
        num_layers     = args.num_layers,
        dropout        = args.dropout,
        alpha          = args.alpha,
        horizon        = args.horizon,
        batch_size     = args.batch_size,
        patience       = args.patience,
        checkpoint_dir = args.checkpoint_dir,
        seed           = args.seed,
        run_name       = args.run_name,
    )