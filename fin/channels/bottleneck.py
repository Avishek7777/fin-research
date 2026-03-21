"""
fin/channels/bottleneck.py

Bandwidth-Limited Communication Channel (Definition 1.3)
=========================================================

Each channel sits at the boundary between two adjacent levels L_{k-1} and L_k.
It enforces the soft bandwidth constraint:

    I(Z_{k-1}; Z_k) <= beta_k

via a stochastic bottleneck and a hinge penalty on the KL divergence:

    L_BW = sum_k  gamma_k * max(0, KL(N(mu_k, sigma_k) || N(0,I)) - beta_k)

This is the critical mechanism that forces abstraction. Without it, FIN
collapses into a standard deep network — the hierarchy exists structurally
but no compression is enforced.

Key design decisions
--------------------
- We use a diagonal Gaussian N(mu, diag(sigma^2)) for tractability.
  The KL against N(0, I) has a closed form (see _kl_diagonal_gaussian).
- The hinge ensures NO penalty when the channel operates within budget beta_k.
  The network is free to use LESS bandwidth than beta_k.
- gamma_k controls constraint tightness:
    gamma_k -> 0   : constraint ignored, degrades to deterministic projection
    gamma_k -> inf : strictly enforced, may starve higher levels of information
- During beta annealing (Appendix), beta_k is passed in dynamically by the
  annealing scheduler rather than read from the fixed config value.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# =============================================================================
# Core Channel Module
# =============================================================================

class BandwidthBottleneck(nn.Module):
    """
    Stochastic bandwidth-limited channel between level k-1 and level k.

    Given an input representation z_{k-1} in R^{in_dim}, produces:
      - z_k in R^{out_dim}  : the upward message (compressed representation)
      - kl_loss             : the hinge penalty for this channel

    The channel learns a projection from in_dim -> out_dim while enforcing
    that the mutual information I(Z_{k-1}; Z_k) stays within beta_k nats.

    Args:
        in_dim  (int)   : dimensionality of z_{k-1}
        out_dim (int)   : dimensionality of z_k  (must be < in_dim for compression)
        beta    (float) : bandwidth budget in nats — soft upper bound on KL
        gamma   (float) : constraint tightness coefficient
        min_sigma(float): floor on sigma to prevent posterior collapse

    Shape:
        Input:  (B, in_dim)
        Output: z_k of shape (B, out_dim), kl_loss scalar tensor
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        beta: float = 8.0,
        gamma: float = 1.0,
        min_sigma: float = 1e-4,
    ):
        super().__init__()

        assert out_dim < in_dim, (
            f"BandwidthBottleneck requires out_dim ({out_dim}) < in_dim ({in_dim}) "
            f"to enforce compression. Got out_dim >= in_dim."
        )
        assert beta > 0, f"Bandwidth budget beta must be positive. Got beta={beta}."
        assert gamma >= 0, f"Tightness gamma must be non-negative. Got gamma={gamma}."

        self.in_dim    = in_dim
        self.out_dim   = out_dim
        self.beta      = beta       # can be overridden dynamically during annealing
        self.gamma     = gamma
        self.min_sigma = min_sigma

        # --- Encoder: z_{k-1} -> (mu_k, log_sigma_k) ---
        # We project to out_dim for both mu and log_sigma.
        # Using a shared trunk + two heads is more parameter-efficient
        # than two fully independent projections.
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, (in_dim + out_dim) // 2),
            nn.LayerNorm((in_dim + out_dim) // 2),
            nn.GELU(),
        )
        self.mu_head        = nn.Linear((in_dim + out_dim) // 2, out_dim)
        self.log_sigma_head = nn.Linear((in_dim + out_dim) // 2, out_dim)

        # Initialize log_sigma to produce sigma ~ 1.0 at start of training.
        # This means the channel starts "open" (high entropy), then learns
        # to compress. Works well with beta annealing starting loose.
        nn.init.zeros_(self.log_sigma_head.weight)
        nn.init.zeros_(self.log_sigma_head.bias)

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------

    def forward(
        self,
        z_prev: torch.Tensor,
        beta_override: float = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_prev        : input representation z_{k-1}, shape (B, in_dim)
            beta_override : if provided, overrides self.beta for this forward pass.
                            Used by the annealing scheduler to dynamically tighten
                            the constraint during training.

        Returns:
            z_k    : sampled representation, shape (B, out_dim)
            kl_loss: scalar tensor — hinge penalty for this channel
        """
        beta = beta_override if beta_override is not None else self.beta

        # 1. Encode: compute distribution parameters
        h          = self.trunk(z_prev)
        mu         = self.mu_head(h)
        log_sigma  = self.log_sigma_head(h)

        # Clamp log_sigma to prevent sigma from collapsing to 0 or exploding.
        # sigma in [min_sigma, 10.0] covers all reasonable operating ranges.
        log_sigma  = log_sigma.clamp(
            min=torch.log(torch.tensor(self.min_sigma)),
            max=torch.log(torch.tensor(10.0)),
        )
        sigma = log_sigma.exp()

        # 2. Sample: reparameterization trick
        #    z_k = mu + sigma * eps,  eps ~ N(0, I)
        #    This keeps the sample differentiable w.r.t. mu and sigma.
        z_k = self._reparameterize(mu, sigma)

        # 3. Compute KL divergence: KL(N(mu, sigma^2) || N(0, I))
        #    Closed-form for diagonal Gaussians (see derivation below).
        kl = self._kl_diagonal_gaussian(mu, sigma)   # shape: (B,)
        kl_mean = kl.mean()                           # scalar: mean over batch

        # 4. Hinge penalty: only penalize if KL exceeds budget beta
        #    max(0, KL - beta) — no gradient signal when within budget.
        kl_loss = self.gamma * F.relu(kl_mean - beta)

        return z_k, kl_loss

    # -------------------------------------------------------------------------
    # Inference (no sampling — use mean for deterministic forward pass)
    # -------------------------------------------------------------------------

    def forward_deterministic(self, z_prev: torch.Tensor) -> torch.Tensor:
        """
        Deterministic forward pass for inference — uses mu directly
        instead of sampling. More stable for evaluation and t-SNE visualization.

        Args:
            z_prev: input representation z_{k-1}, shape (B, in_dim)

        Returns:
            mu: mean of the bottleneck distribution, shape (B, out_dim)
        """
        h         = self.trunk(z_prev)
        mu        = self.mu_head(h)
        return mu

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _reparameterize(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + sigma * eps, eps ~ N(0, I).

        During eval (torch.no_grad), PyTorch won't track gradients anyway,
        but we still sample here. Use forward_deterministic() for eval
        if you want the mean instead.
        """
        eps = torch.randn_like(mu)
        return mu + sigma * eps

    @staticmethod
    def _kl_diagonal_gaussian(
        mu: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """
        Closed-form KL divergence from N(mu, diag(sigma^2)) to N(0, I).

        Derivation:
            KL(N(mu, sigma^2) || N(0,1))
            = 0.5 * sum_d [ sigma_d^2 + mu_d^2 - 1 - log(sigma_d^2) ]

        Summed over the latent dimension d, giving one scalar per batch item.

        Args:
            mu    : mean,  shape (B, D)
            sigma : std,   shape (B, D)  — must be positive

        Returns:
            kl    : per-sample KL, shape (B,)
        """
        # Per-dimension KL contribution
        kl_per_dim = 0.5 * (sigma.pow(2) + mu.pow(2) - 1 - 2 * sigma.log())
        # Sum over latent dimension -> one KL value per sample
        return kl_per_dim.sum(dim=-1)

    # -------------------------------------------------------------------------
    # Diagnostics (used in training loop for monitoring)
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def diagnostics(self, z_prev: torch.Tensor) -> dict:
        """
        Returns a dictionary of monitoring statistics for this channel.
        Call during training to log to TensorBoard.

        Returns dict with keys:
            kl_mean   : mean KL over batch (how much bandwidth is being used)
            kl_max    : max KL over batch
            sigma_mean: mean sigma (how stochastic the channel is)
            mu_norm   : mean L2 norm of mu (how much signal is encoded)
            budget_pct: KL / beta  (what fraction of budget is being used)
        """
        h         = self.trunk(z_prev)
        mu        = self.mu_head(h)
        log_sigma = self.log_sigma_head(h).clamp(
            min=torch.log(torch.tensor(self.min_sigma)),
            max=torch.log(torch.tensor(10.0)),
        )
        sigma = log_sigma.exp()
        kl    = self._kl_diagonal_gaussian(mu, sigma)

        return {
            "kl_mean"   : kl.mean().item(),
            "kl_max"    : kl.max().item(),
            "sigma_mean": sigma.mean().item(),
            "mu_norm"   : mu.norm(dim=-1).mean().item(),
            "budget_pct": (kl.mean() / self.beta).item(),
        }


# =============================================================================
# Channel Registry
# =============================================================================
# Convenience factory so fin.py can build channels from config dicts
# without importing this module directly everywhere.

def build_channel(
    in_dim: int,
    out_dim: int,
    channel_cfg: dict,
) -> BandwidthBottleneck:
    """
    Build a BandwidthBottleneck from a config dictionary.

    Expected config keys:
        beta  (float): bandwidth budget in nats
        gamma (float): constraint tightness

    Example (from fin_cifar100.yaml):
        channel_cfg = {"beta": 16.0, "gamma": 1.0}
    """
    return BandwidthBottleneck(
        in_dim  = in_dim,
        out_dim = out_dim,
        beta    = channel_cfg["beta"],
        gamma   = channel_cfg["gamma"],
    )