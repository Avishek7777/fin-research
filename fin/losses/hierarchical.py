"""
fin/losses/hierarchical.py

Hierarchical Loss — L_FIN (Definition 1.6)
===========================================

The total training objective of a FIN is:

    L_FIN = sum_{k=0}^{K} lambda_k * O_k(z_k)
          + sum_{k=1}^{K} gamma_k * max(0, KL(N(mu_k,sigma_k)||N(0,I)) - beta_k)

Where:
    O_k     : per-level local objective (reconstruction, fine cls, coarse cls)
    lambda_k : level weight — how much each level's objective contributes
    gamma_k  : bandwidth constraint tightness (lives in the channel, pulled here)
    beta_k   : bandwidth budget in nats (also lives in the channel)

The second sum is L_BW — the bandwidth penalty already computed by each
BandwidthBottleneck channel and returned as kl_loss scalars.

This module:
  1. Receives all per-level losses and kl_losses from a FIN forward pass
  2. Applies lambda_k weights
  3. Assembles the total loss
  4. Tracks a detailed breakdown for TensorBoard logging
  5. Computes accuracy metrics for both fine and coarse classification

Design philosophy
-----------------
The loss module is intentionally decoupled from the network modules.
Level0/1/2 compute their own local objectives (reconstruction, classification)
and return the raw loss values. This module applies the weights and combines
them. This separation means:
  - Ablations are clean: set lambda_k=0 to disable a level's objective
  - Loss balance can be tuned without touching network code
  - The full loss breakdown is always visible for debugging

Ablation support
----------------
All ablations from the config map cleanly to this module:
  no_bandwidth        : kl_losses are all 0.0 (channels have gamma=0)
  no_per_level_obj    : lambda_0=0, lambda_2=0 (only L1 fine cls remains)
  These are config-level changes — this module needs no special ablation code.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, Optional


# =============================================================================
# Loss Breakdown Dataclass
# =============================================================================

@dataclass
class FINLossBreakdown:
    """
    Detailed breakdown of all loss components for a single training step.
    Returned by HierarchicalLoss.forward() alongside the total loss scalar.

    Used for:
      - TensorBoard logging (log each field separately)
      - Debugging (if total loss spikes, identify which component)
      - Paper tables (report per-component contributions)

    Fields:
        total       : L_FIN — the scalar used for optimizer.step()
        loss_0      : lambda_0 * O_0 (reconstruction, weighted)
        loss_1      : lambda_1 * O_1 (fine classification, weighted)
        loss_2      : lambda_2 * O_2 (coarse classification, weighted)
        bw_01       : bandwidth penalty for channel L0->L1
        bw_12       : bandwidth penalty for channel L1->L2
        raw_loss_0  : O_0 unweighted (for monitoring)
        raw_loss_1  : O_1 unweighted (for monitoring)
        raw_loss_2  : O_2 unweighted (for monitoring)
        fine_acc    : top-1 accuracy on fine labels (%)
        coarse_acc  : top-1 accuracy on coarse labels (%)
    """
    total      : torch.Tensor = None # type: ignore
    loss_0     : torch.Tensor = None # type: ignore
    loss_1     : torch.Tensor = None # type: ignore
    loss_2     : torch.Tensor = None # type: ignore
    bw_01      : torch.Tensor = None # type: ignore
    bw_12      : torch.Tensor = None # type: ignore
    raw_loss_0 : torch.Tensor = None # type: ignore
    raw_loss_1 : torch.Tensor = None # type: ignore
    raw_loss_2 : torch.Tensor = None # type: ignore
    fine_acc   : float        = 0.0
    coarse_acc : float        = 0.0

    def to_dict(self) -> Dict[str, float]:
        """
        Convert to a flat dict of Python floats for TensorBoard logging.
        Skips None fields gracefully.
        """
        out = {}
        for key, val in self.__dict__.items():
            if val is None:
                continue
            if isinstance(val, torch.Tensor):
                out[key] = val.item()
            else:
                out[key] = float(val)
        return out

    def __repr__(self) -> str:
        d = self.to_dict()
        lines = [f"  {k:<15}: {v:.4f}" for k, v in d.items()]
        return "FINLossBreakdown(\n" + "\n".join(lines) + "\n)"


# =============================================================================
# Hierarchical Loss Module
# =============================================================================

class HierarchicalLoss(nn.Module):
    """
    Assembles L_FIN from per-level losses and bandwidth penalties.

    L_FIN = lambda_0*O_0 + lambda_1*O_1 + lambda_2*O_2 + bw_01 + bw_12

    The bandwidth penalties (bw_01, bw_12) are already weighted by their
    respective gamma_k inside the BandwidthBottleneck channels. This module
    doesn't re-weight them — it just adds them to the total.

    Args:
        lambda_0 (float): weight for L0 reconstruction loss
        lambda_1 (float): weight for L1 fine classification loss
        lambda_2 (float): weight for L2 coarse classification loss

    These map directly to loss.lambda_{0,1,2} in fin_cifar100.yaml.
    """

    def __init__(
        self,
        lambda_0: float = 0.1,
        lambda_1: float = 1.0,
        lambda_2: float = 0.5,
    ):
        super().__init__()

        assert lambda_0 >= 0, f"lambda_0 must be >= 0, got {lambda_0}"
        assert lambda_1 >= 0, f"lambda_1 must be >= 0, got {lambda_1}"
        assert lambda_2 >= 0, f"lambda_2 must be >= 0, got {lambda_2}"
        assert lambda_1 > 0,  "lambda_1 (primary task) must be > 0"

        self.lambda_0 = lambda_0
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------

    def forward(
        self,
        # Per-level raw losses (unweighted scalars from each level's objective)
        loss_0      : torch.Tensor,           # O_0: reconstruction loss
        loss_1      : torch.Tensor,           # O_1: fine classification loss
        loss_2      : torch.Tensor,           # O_2: coarse classification loss
        # Bandwidth penalties (already gamma-weighted from channels)
        kl_loss_01  : torch.Tensor,           # gamma_1 * max(0, KL_01 - beta_1)
        kl_loss_12  : torch.Tensor,           # gamma_2 * max(0, KL_12 - beta_2)
        # Logits and labels for accuracy computation
        fine_logits  : torch.Tensor,          # (B, 100)
        coarse_logits: torch.Tensor,          # (B, 20)
        fine_labels  : torch.Tensor,          # (B,)
        coarse_labels: torch.Tensor,          # (B,)
    ) -> tuple:
        """
        Assemble total L_FIN and return detailed breakdown.

        Returns:
            total_loss (torch.Tensor): scalar, used for optimizer.step()
            breakdown  (FINLossBreakdown): detailed per-component breakdown
        """
        # --- Weighted per-level objectives ---
        weighted_0 = self.lambda_0 * loss_0
        weighted_1 = self.lambda_1 * loss_1
        weighted_2 = self.lambda_2 * loss_2

        # --- Total loss (Definition 1.6) ---
        # L_FIN = sum_k lambda_k * O_k + L_BW
        total = weighted_0 + weighted_1 + weighted_2 + kl_loss_01 + kl_loss_12

        # --- Accuracy metrics ---
        fine_acc   = self._top1_accuracy(fine_logits,   fine_labels)
        coarse_acc = self._top1_accuracy(coarse_logits, coarse_labels)

        # --- Assemble breakdown ---
        breakdown = FINLossBreakdown(
            total      = total,
            loss_0     = weighted_0,
            loss_1     = weighted_1,
            loss_2     = weighted_2,
            bw_01      = kl_loss_01,
            bw_12      = kl_loss_12,
            raw_loss_0 = loss_0,
            raw_loss_1 = loss_1,
            raw_loss_2 = loss_2,
            fine_acc   = fine_acc,
            coarse_acc = coarse_acc,
        )

        return total, breakdown

    # -------------------------------------------------------------------------
    # Accuracy helper
    # -------------------------------------------------------------------------

    @staticmethod
    @torch.no_grad()
    def _top1_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Top-1 accuracy as a percentage.

        Args:
            logits: shape (B, num_classes)
            labels: shape (B,) — integer ground truth indices
        Returns:
            accuracy: float in [0, 100]
        """
        preds   = logits.argmax(dim=-1)
        correct = (preds == labels).float().sum().item()
        return 100.0 * correct / labels.size(0)

    # -------------------------------------------------------------------------
    # Loss weight summary (useful for logging at experiment start)
    # -------------------------------------------------------------------------

    def summary(self) -> str:
        return (
            f"HierarchicalLoss(\n"
            f"  lambda_0 (reconstruction) : {self.lambda_0}\n"
            f"  lambda_1 (fine cls)       : {self.lambda_1}\n"
            f"  lambda_2 (coarse cls)     : {self.lambda_2}\n"
            f"  bandwidth penalties       : from channels (gamma-weighted)\n"
            f")"
        )


# =============================================================================
# Running Metrics Tracker
# =============================================================================

class MetricsTracker:
    """
    Accumulates FINLossBreakdown values over an epoch and computes averages.

    Usage:
        tracker = MetricsTracker()
        for batch in dataloader:
            total_loss, breakdown = loss_fn(...)
            tracker.update(breakdown)
        epoch_metrics = tracker.average()
        tracker.reset()

    This avoids storing all breakdowns in memory — just accumulates
    running sums and counts.
    """

    def __init__(self):
        self._sums : Dict[str, float] = {}
        self._count: int = 0

    def update(self, breakdown: FINLossBreakdown):
        """Add one batch's breakdown to the running totals."""
        d = breakdown.to_dict()
        for key, val in d.items():
            self._sums[key] = self._sums.get(key, 0.0) + val
        self._count += 1

    def average(self) -> Dict[str, float]:
        """Return average of all tracked metrics over accumulated batches."""
        if self._count == 0:
            return {}
        return {k: v / self._count for k, v in self._sums.items()}

    def reset(self):
        """Reset for next epoch."""
        self._sums  = {}
        self._count = 0

    def pretty_print(self, prefix: str = "Train") -> str:
        """Format averaged metrics for console output."""
        avg = self.average()
        if not avg:
            return f"{prefix}: no data"
        lines = [f"{prefix} metrics (avg over {self._count} batches):"]
        for key, val in sorted(avg.items()):
            lines.append(f"  {key:<20}: {val:.4f}")
        return "\n".join(lines)


# =============================================================================
# Factory
# =============================================================================

def build_loss(loss_cfg: dict) -> HierarchicalLoss:
    """
    Build HierarchicalLoss from config dictionary.

    Expected keys:
        lambda_0 (float)
        lambda_1 (float)
        lambda_2 (float)

    Example (from fin_cifar100.yaml):
        loss_cfg = {"lambda_0": 0.1, "lambda_1": 1.0, "lambda_2": 0.5}
    """
    return HierarchicalLoss(
        lambda_0 = loss_cfg["lambda_0"],
        lambda_1 = loss_cfg["lambda_1"],
        lambda_2 = loss_cfg["lambda_2"],
    )