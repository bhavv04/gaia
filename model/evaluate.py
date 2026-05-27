"""
model/evaluate.py

Evaluation metrics for GAIANet cascade prediction.

Two evaluation modes:

1. Node-level regression metrics
   Standard ML metrics on predicted vs true indicator values:
   MAE, RMSE, R², per-domain breakdowns

2. Cascade sequence metrics
   Novel metrics for evaluating cross-domain cascade prediction quality:
   - Cascade Precision / Recall / F1 (did the right nodes enter cascade?)
   - Cascade Order Score (did they enter cascade in the right sequence?)
   - Cross-Domain Cascade Accuracy (specifically for cross-domain edges)
   - Mean Time-to-Cascade Error (was the timing prediction accurate?)

Usage:
    from model.evaluate import CascadeEvaluator
    evaluator = CascadeEvaluator(model, device)
    report = evaluator.evaluate(loader, dataset)
    evaluator.print_report(report)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from scipy.stats import kendalltau

from model.gnn import GAIANet

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RegressionMetrics:
    mae:            float
    rmse:           float
    r2:             float
    per_domain_mae: dict[str, float] = field(default_factory=dict)


@dataclass
class CascadeMetrics:
    precision:              float   # of predicted cascade nodes, how many truly cascaded
    recall:                 float   # of true cascade nodes, how many were predicted
    f1:                     float
    order_score:            float   # Kendall tau of predicted vs true cascade order [−1, 1]
    cross_domain_accuracy:  float   # cascade prediction accuracy on cross-domain edges only
    mean_timing_error_h:    float   # mean absolute error in time-to-cascade (hours)


@dataclass
class EvaluationReport:
    regression:     RegressionMetrics
    cascade:        CascadeMetrics
    num_samples:    int
    cascade_threshold: float


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class CascadeEvaluator:
    """
    Evaluates GAIANet on a held-out dataset.

    Args:
        model:             Trained GAIANet instance
        device:            torch.device
        cascade_threshold: Value above which a node is considered in cascade
        horizon_hours:     Hours per prediction step (used for timing error)
    """

    def __init__(
        self,
        model:             GAIANet,
        device:            torch.device,
        cascade_threshold: float = 0.6,
        horizon_hours:     int   = 72,
    ):
        self.model             = model
        self.device            = device
        self.cascade_threshold = cascade_threshold
        self.horizon_hours     = horizon_hours

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> EvaluationReport:
        """
        Run full evaluation over a DataLoader.

        Returns an EvaluationReport with regression and cascade metrics.
        """
        self.model.eval()

        all_pred   = []
        all_true   = []
        all_domain = []
        all_cross  = []  # cross_domain flag per node (approximated via edge presence)

        cascade_precision_scores = []
        cascade_recall_scores    = []
        cascade_order_scores     = []
        cross_domain_accs        = []
        timing_errors            = []

        for batch in loader:
            batch = batch.to(self.device)
            out   = self.model(batch)

            pred = out["predicted_value"].squeeze().cpu().numpy()
            prob = out["cascade_prob"].squeeze().cpu().numpy()
            true = batch.y.squeeze().cpu().numpy()

            all_pred.append(pred)
            all_true.append(true)

            # Domain index is encoded in node features positions 2:6
            domain_idx = batch.x[:, 2:6].argmax(dim=1).cpu().numpy()
            all_domain.append(domain_idx)

            # --- Cascade node metrics ---
            pred_cascade = (prob >= self.cascade_threshold)
            true_cascade = (true >= self.cascade_threshold)

            tp = np.sum(pred_cascade & true_cascade)
            fp = np.sum(pred_cascade & ~true_cascade)
            fn = np.sum(~pred_cascade & true_cascade)

            precision = tp / (tp + fp + 1e-8)
            recall    = tp / (tp + fn + 1e-8)
            cascade_precision_scores.append(precision)
            cascade_recall_scores.append(recall)

            # --- Cascade order score (Kendall tau) ---
            # Rank nodes by cascade probability (predicted) vs true value (ground truth)
            if len(prob) > 1:
                tau, _ = kendalltau(
                    np.argsort(-prob),   # descending rank by predicted cascade prob
                    np.argsort(-true),   # descending rank by true value
                )
                cascade_order_scores.append(tau if not np.isnan(tau) else 0.0)

            # --- Cross-domain cascade accuracy ---
            # Identify cross-domain edges from edge_attr (index 2 = cross_domain flag)
            if batch.edge_attr is not None and batch.edge_attr.shape[1] > 2:
                cross_mask = batch.edge_attr[:, 2].bool().cpu().numpy()
                if cross_mask.any():
                    # Get target nodes of cross-domain edges
                    target_nodes = batch.edge_index[1][
                        torch.from_numpy(cross_mask).to(self.device)
                    ].cpu().numpy()
                    unique_targets = np.unique(target_nodes)

                    cd_pred = pred_cascade[unique_targets]
                    cd_true = true_cascade[unique_targets]
                    acc = np.mean(cd_pred == cd_true)
                    cross_domain_accs.append(acc)

            # --- Timing error ---
            # For nodes that truly cascaded, how far off was the probability rank
            # as a proxy for timing? (True timing requires multi-step rollout)
            true_cascade_idx = np.where(true_cascade)[0]
            if len(true_cascade_idx) > 0:
                pred_ranks = np.argsort(-prob)  # lower rank = predicted to cascade sooner
                true_ranks = {n: i for i, n in enumerate(np.argsort(-true))}
                errors = [
                    abs(np.where(pred_ranks == n)[0][0] - true_ranks[n]) * self.horizon_hours
                    for n in true_cascade_idx
                    if n in true_ranks
                ]
                if errors:
                    timing_errors.extend(errors)

        # ------------------------------------------------------------------
        # Aggregate
        # ------------------------------------------------------------------
        all_pred   = np.concatenate(all_pred)
        all_true   = np.concatenate(all_true)
        all_domain = np.concatenate(all_domain)

        # Regression
        mae  = float(np.mean(np.abs(all_pred - all_true)))
        rmse = float(np.sqrt(np.mean((all_pred - all_true) ** 2)))
        ss_res = np.sum((all_true - all_pred) ** 2)
        ss_tot = np.sum((all_true - np.mean(all_true)) ** 2)
        r2   = float(1 - ss_res / (ss_tot + 1e-8))

        domain_names = ["marine", "atmospheric", "terrestrial", "freshwater"]
        per_domain_mae = {}
        for i, name in enumerate(domain_names):
            mask = all_domain == i
            if mask.any():
                per_domain_mae[name] = float(np.mean(np.abs(all_pred[mask] - all_true[mask])))

        regression = RegressionMetrics(
            mae            = mae,
            rmse           = rmse,
            r2             = r2,
            per_domain_mae = per_domain_mae,
        )

        # Cascade
        mean_prec  = float(np.mean(cascade_precision_scores))
        mean_rec   = float(np.mean(cascade_recall_scores))
        f1         = 2 * mean_prec * mean_rec / (mean_prec + mean_rec + 1e-8)
        order_score= float(np.mean(cascade_order_scores)) if cascade_order_scores else 0.0
        cd_acc     = float(np.mean(cross_domain_accs))     if cross_domain_accs    else 0.0
        timing_err = float(np.mean(timing_errors))         if timing_errors        else 0.0

        cascade = CascadeMetrics(
            precision             = mean_prec,
            recall                = mean_rec,
            f1                    = f1,
            order_score           = order_score,
            cross_domain_accuracy = cd_acc,
            mean_timing_error_h   = timing_err,
        )

        return EvaluationReport(
            regression        = regression,
            cascade           = cascade,
            num_samples       = len(all_pred),
            cascade_threshold = self.cascade_threshold,
        )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @staticmethod
    def print_report(report: EvaluationReport) -> None:
        r = report.regression
        c = report.cascade

        print("\n" + "=" * 52)
        print("  GAIA Cascade Evaluation Report")
        print("=" * 52)
        print(f"  Samples evaluated : {report.num_samples}")
        print(f"  Cascade threshold : {report.cascade_threshold}")
        print()
        print("  Regression metrics")
        print(f"    MAE              : {r.mae:.4f}")
        print(f"    RMSE             : {r.rmse:.4f}")
        print(f"    R²               : {r.r2:.4f}")
        if r.per_domain_mae:
            print("    MAE per domain")
            for domain, mae in r.per_domain_mae.items():
                print(f"      {domain:<14}: {mae:.4f}")
        print()
        print("  Cascade metrics")
        print(f"    Precision        : {c.precision:.4f}")
        print(f"    Recall           : {c.recall:.4f}")
        print(f"    F1               : {c.f1:.4f}")
        print(f"    Order score (τ)  : {c.order_score:.4f}  (Kendall tau, 1.0 = perfect)")
        print(f"    Cross-domain acc : {c.cross_domain_accuracy:.4f}")
        print(f"    Timing error     : {c.mean_timing_error_h:.1f}h")
        print("=" * 52 + "\n")

    @staticmethod
    def to_dict(report: EvaluationReport) -> dict:
        """Flatten report to a dict for MLflow logging."""
        r, c = report.regression, report.cascade
        out = {
            "eval_mae":              r.mae,
            "eval_rmse":             r.rmse,
            "eval_r2":               r.r2,
            "eval_cascade_precision":c.precision,
            "eval_cascade_recall":   c.recall,
            "eval_cascade_f1":       c.f1,
            "eval_order_score":      c.order_score,
            "eval_cross_domain_acc": c.cross_domain_accuracy,
            "eval_timing_error_h":   c.mean_timing_error_h,
        }
        for domain, mae in r.per_domain_mae.items():
            out[f"eval_mae_{domain}"] = mae
        return out