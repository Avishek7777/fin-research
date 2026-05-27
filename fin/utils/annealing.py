"""
fin/utils/annealing.py

Beta Annealing Scheduler (Appendix — Training Strategy)
========================================================

Implements the beta_k annealing strategy noted during Phase 1 formalization:

    beta_k(t) = beta_k_max * exp(-lambda * t)    [exponential]
    beta_k(t) = beta_k_max * (1 - t/T)           [linear]
    beta_k(t) = beta_k_max * 0.5*(1+cos(pi*t/T)) [cosine]

Where t is the current epoch (after warmup) and T is total training epochs.

Motivation
----------
Starting with LOOSE constraints (large beta_k) and tightening over training
addresses a key instability in hierarchical learning:

Early training:
  - Higher levels (L1, L2) haven't learned meaningful representations yet
  - If bandwidth is tight from the start, L0 is forced to compress into
    representations that higher levels can't yet use
  - Result: all levels fail simultaneously — the hierarchy never bootstraps

With annealing:
  - Early: large beta_k -> channels are essentially open -> network learns
    basic task performance at all levels first
  - Later: beta_k tightens -> compression is enforced -> abstraction emerges
    from an already-functional hierarchy

This is analogous to curriculum learning: start easy, get harder over time.

Real-life analogy: teaching someone to write.
  - Week 1: write freely, no word limit (beta = large)
  - Week 4: summarize in 500 words (beta tightening)
  - Week 8: write a tweet — 280 chars (beta = final tight value)
  The constraint is imposed AFTER the skill is developed, not before.

Implementation
--------------
The scheduler wraps the BandwidthBottleneck channels and calls
channel.forward(..., beta_override=current_beta) each step.
The channels' self.beta values are never mutated — beta_override
is a per-forward-pass argument, keeping the original config values intact
for reference and logging.
"""

import math
from typing import Dict, List
from fin.channels.bottleneck import BandwidthBottleneck


# =============================================================================
# Beta Annealing Scheduler
# =============================================================================

class BetaAnnealingScheduler:
    """
    Anneals bandwidth budget beta_k across training epochs.

    Manages one or more BandwidthBottleneck channels simultaneously,
    each with their own beta_max (from config) and shared annealing schedule.

    Usage in training loop:
        scheduler = BetaAnnealingScheduler(channels, annealing_cfg, total_epochs)

        for epoch in range(total_epochs):
            current_betas = scheduler.step(epoch)
            # Pass current_betas to fin.forward() which routes them to channels

    Args:
        channels      (dict): {name: BandwidthBottleneck} e.g.
                              {"channel_01": ch01, "channel_12": ch12}
        annealing_cfg (dict): from config — strategy, warmup_epochs, decay_lambda
        total_epochs  (int) : total training epochs
    """

    def __init__(
        self,
        channels      : Dict[str, BandwidthBottleneck],
        annealing_cfg : dict,
        total_epochs  : int,
    ):
        self.channels      = channels
        self.strategy      = annealing_cfg.get("strategy", "exponential")
        self.warmup_epochs = annealing_cfg.get("warmup_epochs", 10)
        self.decay_lambda  = annealing_cfg.get("decay_lambda", 0.05)
        self.total_epochs  = total_epochs
        self.enabled       = annealing_cfg.get("enabled", True)

        # Store original beta_max values from each channel
        # These never change — we compute current beta relative to them
        self.beta_max: Dict[str, float] = {
            name: ch.beta for name, ch in channels.items()
        }

        # History for logging and paper plots
        self.history: Dict[str, List[float]] = {name: [] for name in channels}

        assert self.strategy in ("exponential", "linear", "cosine"), (
            f"Unknown annealing strategy: {self.strategy}. "
            f"Choose from: exponential, linear, cosine"
        )

    # -------------------------------------------------------------------------
    # Step: compute current betas for this epoch
    # -------------------------------------------------------------------------

    def step(self, epoch: int) -> Dict[str, float]:
        """
        Compute current beta_k values for the given epoch.

        During warmup: returns beta_max (no constraint tightening yet)
        After warmup : anneals according to strategy

        Args:
            epoch: current epoch (0-indexed)

        Returns:
            current_betas: {channel_name: current_beta_value}
        """
        current_betas = {}

        for name, channel in self.channels.items():
            beta_max = self.beta_max[name]

            if not self.enabled:
                # Annealing disabled — use fixed beta from config throughout
                current_beta = beta_max

            elif epoch < self.warmup_epochs:
                # Warmup phase: full bandwidth, no constraint
                current_beta = beta_max

            else:
                # Annealing phase: tighten constraint over time
                t = epoch - self.warmup_epochs
                T = max(self.total_epochs - self.warmup_epochs, 1)
                current_beta = self._anneal(beta_max, t, T)

            current_betas[name] = current_beta
            self.history[name].append(current_beta)

        return current_betas

    # -------------------------------------------------------------------------
    # Annealing strategies
    # -------------------------------------------------------------------------

    def _anneal(self, beta_max: float, t: int, T: int) -> float:
        """
        Compute annealed beta at step t of T total annealing steps.

        All strategies share the property:
            _anneal(beta_max, 0, T) = beta_max   (start at max)
            _anneal(beta_max, T, T) = beta_min    (end at minimum)

        We set beta_min = beta_max * 0.1 to avoid collapsing to zero
        (a channel with beta=0 would transmit nothing).
        """
        beta_min = beta_max * 0.1
        progress = t / T   # in [0, 1]

        if self.strategy == "exponential":
            # Fast initial drop, slow convergence to beta_min
            # beta(t) = beta_max * exp(-lambda * t)
            # Normalized so that at t=T we reach approximately beta_min
            scale = math.exp(-self.decay_lambda * t)
            # Clamp to beta_min so we don't go below it
            current = max(beta_max * scale, beta_min)

        elif self.strategy == "linear":
            # Uniform decay from beta_max to beta_min over T steps
            current = beta_max - (beta_max - beta_min) * progress

        elif self.strategy == "cosine":
            # Smooth cosine decay — slow start, fast middle, slow end
            # Matches cosine LR schedulers familiar to ML practitioners
            current = beta_min + 0.5 * (beta_max - beta_min) * (
                1 + math.cos(math.pi * progress)
            )

        return float(current) # type: ignore

    # -------------------------------------------------------------------------
    # Logging and diagnostics
    # -------------------------------------------------------------------------

    def current_betas(self) -> Dict[str, float]:
        """Return the most recent beta value for each channel."""
        return {
            name: hist[-1] if hist else self.beta_max[name]
            for name, hist in self.history.items()
        }

    def log_str(self, epoch: int) -> str:
        """Format current beta values for console output."""
        betas = self.current_betas()
        parts = [f"{name}={v:.3f}" for name, v in betas.items()]
        return f"Epoch {epoch:03d} | betas: {', '.join(parts)}"

    def get_history(self) -> Dict[str, List[float]]:
        """Return full beta history for plotting in the paper."""
        return dict(self.history)


# =============================================================================
# Factory
# =============================================================================

def build_annealing_scheduler(
    channels      : Dict[str, BandwidthBottleneck],
    annealing_cfg : dict,
    total_epochs  : int,
) -> BetaAnnealingScheduler:
    """
    Build a BetaAnnealingScheduler from config.

    Args:
        channels      : dict of {name: BandwidthBottleneck}
        annealing_cfg : from fin_cifar100.yaml annealing section
        total_epochs  : training.epochs from config

    Returns:
        BetaAnnealingScheduler instance
    """
    return BetaAnnealingScheduler(
        channels      = channels,
        annealing_cfg = annealing_cfg,
        total_epochs  = total_epochs,
    )