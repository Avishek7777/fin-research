"""
fin/network/fin.py

Fractal Intelligence Network — Full Assembly (Definition 1.2)
=============================================================

FIN_K = (L_0, L_1, ..., L_K)

This module assembles all previously built components into a single
nn.Module that can be trained end-to-end. It orchestrates the complete
forward pass in the correct order as specified by Definitions 1.2 and 1.4.

Forward Pass Order
------------------
A naive implementation would call each level sequentially top-to-bottom.
But FIN requires a two-phase pass to implement bidirectional message passing:

PHASE 1 — Bottom-Up (upward encoding + channel compression)
    x -> L0.encode() -> channel_01 -> L1.encode() -> channel_12 -> L2.encode()

PHASE 2 — Top-Down (feedback refinement, apex to base)
    L2 (apex, no refinement)
    -> L2.downward_message() -> L1.refine()
    -> L1.downward_message() -> L0.refine()

PHASE 3 — Objectives (compute all losses on refined representations)
    L0: decode(z0_refined) -> reconstruction loss
    L1: classify(z1_refined) -> fine classification loss
    L2: classify(z2) -> coarse classification loss

This three-phase structure is what makes FIN fundamentally different
from a standard deep network. The representations at each level are
shaped by BOTH bottom-up signal AND top-down constraint before any
objective is computed on them.

Real-life analogy — the three phases as a newsroom:
    Phase 1 (bottom-up): reporters file raw stories up the chain
    Phase 2 (top-down): editors send back briefs — "focus on this angle"
    Phase 3 (objectives): each level produces its deliverable
                          (raw notes, article, front page) and is graded

Self-Similarity Check (Definition 1.5)
---------------------------------------
The same structural rule repeats at every level boundary:
    encode -> [channel] -> compress -> refine -> communicate
This is verified in the forward pass structure — each level boundary
executes identical steps regardless of what L_k actually is internally.

Ablation Support
----------------
All four ablations from the config are handled here:
    no_bandwidth      : channels have gamma=0, kl_losses are ~0
    no_per_level_obj  : lambda weights in loss module handle this
    no_top_down       : feedback_enabled=False in all levels
    no_self_similarity: use_self_similar=False builds bespoke level configs
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from fin.levels.level0 import Level0
from fin.levels.level1 import Level1
from fin.levels.level2 import Level2
from fin.channels.bottleneck import BandwidthBottleneck, build_channel
from fin.losses.hierarchical import HierarchicalLoss, FINLossBreakdown, build_loss
from fin.utils.annealing import BetaAnnealingScheduler, build_annealing_scheduler


# =============================================================================
# FIN Forward Output
# =============================================================================

class FINOutput:
    """
    Structured output from a single FIN forward pass.

    Bundles all representations, logits, losses, and diagnostics
    into one object. This keeps the training loop clean — it unpacks
    exactly what it needs rather than juggling tuples.

    Fields:
        # Representations (after top-down refinement)
        z0          : (B, d0) — refined L0 representation
        z1          : (B, d1) — refined L1 representation
        z2          : (B, d2) — L2 apex representation (no refinement)

        # Classification logits
        fine_logits  : (B, 100) — from L1
        coarse_logits: (B, 20)  — from L2

        # Reconstruction
        x_recon     : (B, 3, 32, 32) — L0 decoder output

        # Losses (raw, before lambda weighting)
        loss_0      : reconstruction loss scalar
        loss_1      : fine classification loss scalar
        loss_2      : coarse classification loss scalar
        kl_loss_01  : bandwidth penalty, channel L0->L1
        kl_loss_12  : bandwidth penalty, channel L1->L2

        # Assembled total loss and breakdown (set after HierarchicalLoss.forward)
        total_loss  : L_FIN scalar
        breakdown   : FINLossBreakdown
    """

    def __init__(self):
        self.z0           = None
        self.z1           = None
        self.z2           = None
        self.fine_logits  = None
        self.coarse_logits= None
        self.x_recon      = None
        self.loss_0       = None
        self.loss_1       = None
        self.loss_2       = None
        self.kl_loss_01   = None
        self.kl_loss_12   = None
        self.total_loss   = None
        self.breakdown    = None


# =============================================================================
# Fractal Intelligence Network
# =============================================================================

class FIN(nn.Module):
    """
    Fractal Intelligence Network (Definition 1.2).

    FIN_3 = (L_0, L_1, L_2)

    with d_0=512 > d_1=128 > d_2=32 (strictly decreasing, Definition 1.2).

    Args:
        arch_cfg      (dict): architecture section from fin_cifar100.yaml
        bandwidth_cfg (dict): bandwidth section from fin_cifar100.yaml
        feedback_cfg  (dict): feedback section from fin_cifar100.yaml
        loss_cfg      (dict): loss section from fin_cifar100.yaml
        data_cfg      (dict): data section (for num_classes)
        annealing_cfg (dict): annealing section from fin_cifar100.yaml
        training_cfg  (dict): training section (for total_epochs)
    """

    def __init__(
        self,
        arch_cfg     : dict,
        bandwidth_cfg: dict,
        feedback_cfg : dict,
        loss_cfg     : dict,
        data_cfg     : dict,
        annealing_cfg: dict,
        training_cfg : dict,
    ):
        super().__init__()

        # ── Dimensionalities ──────────────────────────────────────────────
        d0 = arch_cfg["level0"]["out_dim"]   # 512
        d1 = arch_cfg["level1"]["out_dim"]   # 128
        d2 = arch_cfg["level2"]["out_dim"]   # 32

        assert d0 > d1 > d2, (
            f"FIN requires strictly decreasing dims: d0({d0}) > d1({d1}) > d2({d2})"
        )

        feedback_enabled = feedback_cfg.get("enabled", True)
        alpha_init       = feedback_cfg.get("alpha_init", 0.1)
        num_fine         = data_cfg["num_fine_classes"]    # 100
        num_coarse       = data_cfg["num_coarse_classes"]  # 20

        # ── Level 0 — CNN ─────────────────────────────────────────────────
        self.level0 = Level0(
            num_channels     = arch_cfg["level0"]["num_channels"],
            out_dim          = d0,
            feedback_dim     = d1,        # receives top-down from L1
            alpha_init       = alpha_init,
            feedback_enabled = feedback_enabled,
        )

        # ── Channel L0 -> L1 ──────────────────────────────────────────────
        self.channel_01 = build_channel(
            in_dim      = d0,
            out_dim     = d1,
            channel_cfg = bandwidth_cfg["channel_01"],
        )

        # ── Level 1 — Transformer ─────────────────────────────────────────
        self.level1 = Level1(
            in_dim           = d1,         # receives z0 AFTER channel compression
            out_dim          = d1,         # stays at d1 (channel already compressed)
            n_patches        = arch_cfg["level1"].get("n_patches", 8),
            embed_dim        = arch_cfg["level1"].get("embed_dim", 64),
            n_heads          = arch_cfg["level1"]["num_heads"],
            num_layers       = arch_cfg["level1"]["num_layers"],
            dropout          = arch_cfg["level1"]["dropout"],
            num_fine_classes = num_fine,
            feedback_dim     = d2,         # receives top-down from L2
            alpha_init       = alpha_init,
            feedback_enabled = feedback_enabled,
            mlp_mode         = arch_cfg["level1"].get("mlp_mode", False),
        )

        # ── Channel L1 -> L2 ──────────────────────────────────────────────
        self.channel_12 = build_channel(
            in_dim      = d1,
            out_dim     = d2,
            channel_cfg = bandwidth_cfg["channel_12"],
        )

        # ── Level 2 — MLP Apex ────────────────────────────────────────────
        # channel_12 already compressed d1->d2, so Level2 receives and
        # operates within d2-dimensional space (refines rather than compresses)
        self.level2 = Level2(
            in_dim             = d2,
            out_dim            = d2,
            hidden_dim         = arch_cfg["level2"]["hidden_dim"],
            num_coarse_classes = num_coarse,
        )

        # ── Hierarchical Loss ─────────────────────────────────────────────
        self.loss_fn = build_loss(loss_cfg)

        # ── Beta Annealing Scheduler ──────────────────────────────────────
        self.annealing = build_annealing_scheduler(
            channels = {
                "channel_01": self.channel_01,
                "channel_12": self.channel_12,
            },
            annealing_cfg = annealing_cfg,
            total_epochs  = training_cfg["epochs"],
        )

        # Store current betas — updated each epoch via update_betas()
        self._current_betas: Dict[str, float] = {
            "channel_01": bandwidth_cfg["channel_01"]["beta"],
            "channel_12": bandwidth_cfg["channel_12"]["beta"],
        }

        # Store dims for external access (e.g. evaluation, t-SNE)
        self.d0, self.d1, self.d2 = d0, d1, 

        # ── Single-level mode (no_hierarchy_flat ablation) ────────────────
        self.single_level = arch_cfg.get("single_level", False)
        if self.single_level:
            # Replace full hierarchy with a direct fine+coarse classifier on z0
            self.flat_fine_head   = nn.Linear(d0, num_fine)
            self.flat_coarse_head = nn.Linear(d0, num_coarse)

    # =========================================================================
    # Forward Pass
    # =========================================================================

    def forward(
        self,
        x            : torch.Tensor,
        fine_labels  : torch.Tensor,
        coarse_labels: torch.Tensor,
    ) -> FINOutput:
        out = FINOutput()

        # ── Single-level flat mode (no_hierarchy_flat ablation) ───────────
        if self.single_level:
            z0 = self.level0.encode(x)
            fine_logits   = self.flat_fine_head(z0)
            coarse_logits = self.flat_coarse_head(z0)

            loss_1 = self.level1.objective(fine_logits, fine_labels)
            loss_2 = self.level2.objective(coarse_logits, coarse_labels)
            loss_0 = torch.tensor(0.0, device=x.device)
            kl_01  = torch.tensor(0.0, device=x.device)
            kl_12  = torch.tensor(0.0, device=x.device)

            out.z0            = z0
            out.z1            = z0   # same rep, no hierarchy
            out.z2            = z0
            out.fine_logits   = fine_logits
            out.coarse_logits = coarse_logits
            out.loss_0        = loss_0
            out.loss_1        = loss_1
            out.loss_2        = loss_2
            out.kl_loss_01    = kl_01
            out.kl_loss_12    = kl_12

            total_loss, breakdown = self.loss_fn(
                loss_0        = loss_0,
                loss_1        = loss_1,
                loss_2        = loss_2,
                kl_loss_01    = kl_01,
                kl_loss_12    = kl_12,
                fine_logits   = fine_logits,
                coarse_logits = coarse_logits,
                fine_labels   = fine_labels,
                coarse_labels = coarse_labels,
            )
            out.total_loss = total_loss
            out.breakdown  = breakdown
            return out

        # ── Normal 3-level forward ────────────────────────────────────────
        z0_raw = self.level0.encode(x)

        z0_compressed, kl_01 = self.channel_01(
            z0_raw,
            beta_override=self._current_betas["channel_01"],
        )

        z1_raw = self.level1.encode(z0_compressed)

        z1_compressed, kl_12 = self.channel_12(
            z1_raw,
            beta_override=self._current_betas["channel_12"],
        )

        z2 = self.level2.encode(z1_compressed)

        out.kl_loss_01 = kl_01
        out.kl_loss_12 = kl_12

        down_msg_to_l1 = self.level2.downward_message(z2)
        z1_refined     = self.level1.refine(z1_raw, top_down_msg=down_msg_to_l1)
        down_msg_to_l0 = self.level1.downward_message(z1_refined)
        z0_refined     = self.level0.refine(z0_raw, top_down_msg=down_msg_to_l0)

        out.z0 = z0_refined
        out.z1 = z1_refined
        out.z2 = z2

        x_recon       = self.level0.decode(z0_refined)
        loss_0        = self.level0.objective(x, x_recon)
        out.x_recon   = x_recon
        out.loss_0    = loss_0

        fine_logits   = self.level1.classify(z1_refined)
        loss_1        = self.level1.objective(fine_logits, fine_labels)
        out.fine_logits = fine_logits
        out.loss_1    = loss_1

        coarse_logits  = self.level2.classify(z2)
        loss_2         = self.level2.objective(coarse_logits, coarse_labels)
        out.coarse_logits = coarse_logits
        out.loss_2     = loss_2

        total_loss, breakdown = self.loss_fn(
            loss_0        = loss_0,
            loss_1        = loss_1,
            loss_2        = loss_2,
            kl_loss_01    = kl_01,
            kl_loss_12    = kl_12,
            fine_logits   = fine_logits,
            coarse_logits = coarse_logits,
            fine_labels   = fine_labels,
            coarse_labels = coarse_labels,
        )

        out.total_loss = total_loss
        out.breakdown  = breakdown
        return out

    # =========================================================================
    # Inference Forward (deterministic — for evaluation and t-SNE)
    # =========================================================================

    @torch.no_grad()
    def encode_deterministic(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Deterministic forward pass for evaluation and representation analysis.
        Uses channel means (no sampling) for stable, reproducible embeddings.

        Args:
            x: raw pixels (B, 3, 32, 32)

        Returns:
            z0, z1, z2: representations at each level
                        shapes: (B, d0), (B, d1), (B, d2)
        """
        # Phase 1: bottom-up (deterministic channels)
        z0_raw = self.level0.encode(x)
        z0_compressed = self.channel_01.forward_deterministic(z0_raw)
        z1_raw = self.level1.encode(z0_compressed)
        z1_compressed = self.channel_12.forward_deterministic(z1_raw)
        z2 = self.level2.encode(z1_compressed)

        # Phase 2: top-down
        down_msg_to_l1 = self.level2.downward_message(z2)
        z1_refined     = self.level1.refine(z1_raw, top_down_msg=down_msg_to_l1)
        down_msg_to_l0 = self.level1.downward_message(z1_refined)
        z0_refined     = self.level0.refine(z0_raw, top_down_msg=down_msg_to_l0)

        return z0_refined, z1_refined, z2

    # =========================================================================
    # Beta Annealing Interface
    # =========================================================================

    def update_betas(self, epoch: int) -> Dict[str, float]:
        """
        Update current beta values for this epoch via the annealing scheduler.
        Call once per epoch BEFORE the training loop for that epoch.

        Args:
            epoch: current epoch (0-indexed)

        Returns:
            current_betas: dict of {channel_name: beta_value} for logging
        """
        self._current_betas = self.annealing.step(epoch)
        return self._current_betas

    # =========================================================================
    # Diagnostics
    # =========================================================================

    @torch.no_grad()
    def channel_diagnostics(self, x: torch.Tensor) -> dict:
        """
        Returns bandwidth usage diagnostics for both channels.
        Useful for monitoring how much of the budget each channel is using.

        Args:
            x: raw pixels (B, 3, 32, 32)

        Returns:
            dict with channel_01 and channel_12 diagnostics
        """
        z0_raw = self.level0.encode(x)
        z0_comp = self.channel_01.forward_deterministic(z0_raw)

        return {
            "channel_01": self.channel_01.diagnostics(z0_raw),
            "channel_12": self.channel_12.diagnostics(z0_comp),
        }

    def param_count(self) -> dict:
        """
        Full parameter breakdown across all components.
        Used to populate the parameter column in the paper's results table.
        """
        l0   = self.level0.param_count()
        l1   = self.level1.param_count()
        l2   = self.level2.param_count()
        ch01 = sum(p.numel() for p in self.channel_01.parameters())
        ch12 = sum(p.numel() for p in self.channel_12.parameters())

        total = l0["total"] + l1["total"] + l2["total"] + ch01 + ch12

        return {
            "level0"     : l0["total"],
            "channel_01" : ch01,
            "level1"     : l1["total"],
            "channel_12" : ch12,
            "level2"     : l2["total"],
            "total"      : total,
            "total_M"    : round(total / 1e6, 2),  # in millions
        }

    def architecture_summary(self) -> str:
        """
        Print a human-readable summary of the FIN architecture.
        Maps directly to Table in Section 4.1 of the paper.
        """
        pc = self.param_count()
        lines = [
            "=" * 60,
            "Fractal Intelligence Network (FIN-3)",
            "=" * 60,
            f"  L0  CNN Encoder/Decoder    d0={self.d0}   params={pc['level0']:,}",
            f"  |",
            f"  Channel 01 (beta={self._current_betas['channel_01']:.2f})       params={pc['channel_01']:,}",
            f"  |",
            f"  L1  Transformer            d1={self.d1}   params={pc['level1']:,}",
            f"  |",
            f"  Channel 12 (beta={self._current_betas['channel_12']:.2f})        params={pc['channel_12']:,}",
            f"  |",
            f"  L2  MLP Apex               d2={self.d2}    params={pc['level2']:,}",
            "=" * 60,
            f"  Total parameters: {pc['total']:,} ({pc['total_M']}M)",
            "=" * 60,
        ]
        return "\n".join(lines)


# =============================================================================
# Factory — build FIN from full config dict
# =============================================================================

def build_fin(cfg: dict) -> FIN:
    """
    Build a FIN instance from the full parsed YAML config.

    Args:
        cfg: full config dict (parsed from fin_cifar100.yaml)

    Returns:
        FIN instance ready for training

    Example:
        import yaml
        with open("configs/fin_cifar100.yaml") as f:
            cfg = yaml.safe_load(f)
        model = build_fin(cfg)
        print(model.architecture_summary())
    """
    return FIN(
        arch_cfg     = cfg["architecture"],
        bandwidth_cfg= cfg["bandwidth"],
        feedback_cfg = cfg["feedback"],
        loss_cfg     = cfg["loss"],
        data_cfg     = cfg["data"],
        annealing_cfg= cfg["annealing"],
        training_cfg = cfg["training"],
    )