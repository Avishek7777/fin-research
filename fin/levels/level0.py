"""
fin/levels/level0.py

Level 0 — Raw Signal Processing (Definition 1.1)
=================================================

L0 = (Z_0, f_0, g_0, O_0)

    Z_0 : R^(B x 3 x 32 x 32)  — raw CIFAR-100 pixel space
    f_0 : pixels -> z_0 in R^d0  — CNN encoder (upward message)
    g_0 : z_0    -> pixels        — CNN decoder (reconstruction for O_0)
              also accepts top-down message from L1 via alpha gate

    O_0 : reconstruction loss — MSE between input pixels and g_0(z_0)
          Forces L0 to preserve raw signal fidelity.

Architecture
------------
Encoder: 3 convolutional blocks (Conv -> BN -> GELU -> Pool)
         progressively increasing channels: 3 -> 64 -> 128 -> 256
         followed by adaptive average pool + linear projection to d0=512

Decoder: mirrors the encoder — linear unproject + 3 transposed conv blocks
         reconstructs (3, 32, 32) from z_0

Top-down feedback (Definition 1.4):
         z_0_refined = z_0 + alpha * gate(top_down_message)
         where alpha is a learned scalar initialized at alpha_init (config).
         The gate is a linear projection from d1 -> d0.

Self-similarity note (Definition 1.5):
         L0 uses a CNN because the input is spatial (images).
         However, the encode -> compress -> communicate -> refine rule
         is identical at every level. The *template* is the same;
         only the instantiation changes to match the input modality.
         For a text FIN, L0 would be an embedding + positional encoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# =============================================================================
# Building Blocks
# =============================================================================

class ConvBlock(nn.Module):
    """
    Single convolutional block: Conv2d -> BatchNorm -> GELU.
    Used in the encoder. Optionally downsamples via stride=2.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TransposedConvBlock(nn.Module):
    """
    Single transposed convolutional block: ConvTranspose2d -> BatchNorm -> GELU.
    Used in the decoder. Optionally upsamples via stride=2.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        activation: bool = True,
    ):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        ]
        if activation:
            layers += [nn.BatchNorm2d(out_channels), nn.GELU()]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# =============================================================================
# Level 0 Encoder
# =============================================================================

class Level0Encoder(nn.Module):
    """
    f_0: pixels -> z_0

    Maps (B, 3, 32, 32) -> (B, d0=512).

    Architecture:
        Block 0: (3,  32, 32) -> (64,  16, 16)   stride=2
        Block 1: (64, 16, 16) -> (128,  8,  8)   stride=2
        Block 2: (128, 8,  8) -> (256,  4,  4)   stride=2
        AdaptiveAvgPool       -> (256,  1,  1)
        Flatten               -> (256,)
        Linear + LayerNorm    -> (512,)  = z_0

    Args:
        num_channels (list): progressive channel sizes e.g. [64, 128, 256]
        out_dim      (int) : d0, dimensionality of z_0
    """

    def __init__(self, num_channels: list = [64, 128, 256], out_dim: int = 512):
        super().__init__()

        assert len(num_channels) == 3, "Expected 3 channel stages for CIFAR-100 (32x32 input)"

        c0, c1, c2 = num_channels

        self.blocks = nn.Sequential(
            ConvBlock(3,  c0, stride=2),   # 32x32 -> 16x16
            ConvBlock(c0, c1, stride=2),   # 16x16 ->  8x8
            ConvBlock(c1, c2, stride=2),   #  8x8  ->  4x4
        )

        self.pool    = nn.AdaptiveAvgPool2d(1)   # (B, c2, 4, 4) -> (B, c2, 1, 1)
        self.flatten = nn.Flatten()               # -> (B, c2)

        self.project = nn.Sequential(
            nn.Linear(c2, out_dim),
            nn.LayerNorm(out_dim),
        )

        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: raw pixel input, shape (B, 3, 32, 32), values in [0, 1]
        Returns:
            z_0: shape (B, d0)
        """
        h = self.blocks(x)
        h = self.pool(h)
        h = self.flatten(h)
        z_0 = self.project(h)
        return z_0


# =============================================================================
# Level 0 Decoder
# =============================================================================

class Level0Decoder(nn.Module):
    """
    g_0: z_0 -> reconstructed pixels

    Maps (B, d0=512) -> (B, 3, 32, 32).

    Used for two purposes:
      1. Computing reconstruction loss O_0 (training signal for L0)
      2. Receiving top-down feedback: g_0 also serves as the
         "downward decoder" in Definition 1.4 when called from above.

    Architecture (mirror of encoder):
        Linear              : d0 -> 256
        Reshape             : (256,) -> (256, 4, 4)
        TransposedConvBlock :  4x4  ->  8x8
        TransposedConvBlock :  8x8  -> 16x16
        TransposedConvBlock : 16x16 -> 32x32
        Final Conv          : 32x32 -> (3, 32, 32) + Sigmoid

    Args:
        num_channels (list): must match encoder [64, 128, 256]
        in_dim       (int) : d0, dimensionality of z_0
    """

    def __init__(self, num_channels: list = [64, 128, 256], in_dim: int = 512):
        super().__init__()

        c0, c1, c2 = num_channels

        # Project back from d0 to spatial feature map
        self.unproject = nn.Sequential(
            nn.Linear(in_dim, c2 * 4 * 4),
            nn.GELU(),
        )

        # Upsample back to 32x32
        self.blocks = nn.Sequential(
            TransposedConvBlock(c2, c1),     # 4x4  ->  8x8
            TransposedConvBlock(c1, c0),     # 8x8  -> 16x16
            TransposedConvBlock(c0, c0),     # 16x16 -> 32x32
        )

        # Final pixel projection — no BN, sigmoid to keep values in [0,1]
        self.to_pixels = nn.Sequential(
            nn.Conv2d(c0, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        self.c2 = c2

    def forward(self, z_0: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_0: shape (B, d0)
        Returns:
            x_recon: reconstructed pixels, shape (B, 3, 32, 32), values in [0, 1]
        """
        h = self.unproject(z_0)
        h = h.view(h.size(0), self.c2, 4, 4)
        h = self.blocks(h)
        x_recon = self.to_pixels(h)
        return x_recon


# =============================================================================
# Level 0 — Full Module (Definition 1.1 + 1.4)
# =============================================================================

class Level0(nn.Module):
    """
    Full Level 0 module: encoder + decoder + top-down feedback gate.

    Implements Definition 1.1:
        L_0 = (Z_0, f_0, g_0, O_0)

    And Definition 1.4 (top-down feedback):
        z_0_refined = z_0 + alpha * gate(top_down_message)

    The top_down_message comes from Level 1 (shape: B x d1).
    The gate projects it from d1 -> d0 so dimensions match.

    Args:
        num_channels (list) : CNN channel progression [64, 128, 256]
        out_dim      (int)  : d0 — output dimensionality of z_0
        feedback_dim (int)  : d1 — dimensionality of top-down message from L1
        alpha_init   (float): initial value of the top-down gate scalar
        feedback_enabled (bool): if False, disables top-down connection (ablation)
    """

    def __init__(
        self,
        num_channels: list = [64, 128, 256],
        out_dim: int = 512,
        feedback_dim: int = 128,
        alpha_init: float = 0.1,
        feedback_enabled: bool = True,
    ):
        super().__init__()

        self.out_dim          = out_dim
        self.feedback_enabled = feedback_enabled

        # f_0: upward encoder
        self.encoder = Level0Encoder(num_channels=num_channels, out_dim=out_dim)

        # g_0: downward decoder (for reconstruction objective O_0)
        self.decoder = Level0Decoder(num_channels=num_channels, in_dim=out_dim)

        # Top-down feedback gate (Definition 1.4)
        # Projects top-down message from d1 -> d0
        if feedback_enabled:
            self.feedback_gate = nn.Sequential(
                nn.Linear(feedback_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
            )
            # alpha: learned scalar gate — starts small so top-down
            # influence grows gradually as L1 becomes meaningful
            self.alpha = nn.Parameter(torch.tensor(alpha_init))

    # -------------------------------------------------------------------------
    # Forward: upward pass (encoder only)
    # -------------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Bottom-up pass: pixels -> z_0.
        Called first in the FIN forward pass.

        Args:
            x: raw pixels, shape (B, 3, 32, 32)
        Returns:
            z_0: shape (B, d0)
        """
        return self.encoder(x)

    # -------------------------------------------------------------------------
    # Forward: top-down refinement (Definition 1.4)
    # -------------------------------------------------------------------------

    def refine(
        self,
        z_0: torch.Tensor,
        top_down_msg: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Refine z_0 using top-down message from L1.

        z_0_refined = z_0 + alpha * gate(top_down_msg)

        If feedback is disabled (ablation) or top_down_msg is None,
        returns z_0 unchanged.

        Args:
            z_0          : bottom-up representation, shape (B, d0)
            top_down_msg : message from L1, shape (B, d1)  [optional]
        Returns:
            z_0_refined  : shape (B, d0)
        """
        if not self.feedback_enabled or top_down_msg is None:
            return z_0

        feedback = self.feedback_gate(top_down_msg)
        # Clamp alpha to [0, 1] to prevent top-down from overwhelming bottom-up
        alpha    = self.alpha.clamp(0.0, 1.0)
        return z_0 + alpha * feedback

    # -------------------------------------------------------------------------
    # Forward: reconstruction (objective O_0)
    # -------------------------------------------------------------------------

    def decode(self, z_0: torch.Tensor) -> torch.Tensor:
        """
        Decode z_0 back to pixel space.
        Used to compute reconstruction loss O_0 = MSE(x, decode(z_0)).

        Args:
            z_0: shape (B, d0)
        Returns:
            x_recon: shape (B, 3, 32, 32)
        """
        return self.decoder(z_0)

    # -------------------------------------------------------------------------
    # Objective O_0: reconstruction loss
    # -------------------------------------------------------------------------

    @staticmethod
    def objective(
        x: torch.Tensor,
        x_recon: torch.Tensor,
    ) -> torch.Tensor:
        """
        O_0: pixel reconstruction loss.
        MSE between original and reconstructed image.

        Args:
            x      : original pixels, shape (B, 3, 32, 32)
            x_recon: reconstructed pixels, shape (B, 3, 32, 32)
        Returns:
            loss: scalar tensor
        """
        return F.mse_loss(x_recon, x)

    # -------------------------------------------------------------------------
    # Convenience: full forward (used during training)
    # -------------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        top_down_msg: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full L0 forward pass.

        Steps:
            1. Encode pixels -> z_0
            2. Refine z_0 with top-down message (if provided)
            3. Decode z_0_refined -> x_recon
            4. Compute reconstruction loss

        Args:
            x            : raw pixels, shape (B, 3, 32, 32)
            top_down_msg : from L1, shape (B, d1) — None on first pass

        Returns:
            z_0_refined : refined representation, shape (B, d0)
            x_recon     : reconstructed pixels, shape (B, 3, 32, 32)
            loss_0      : reconstruction loss scalar
        """
        z_0         = self.encode(x)
        z_0_refined = self.refine(z_0, top_down_msg)
        x_recon     = self.decode(z_0_refined)
        loss_0      = self.objective(x, x_recon)

        return z_0_refined, x_recon, loss_0

    # -------------------------------------------------------------------------
    # Parameter count utility
    # -------------------------------------------------------------------------

    def param_count(self) -> dict:
        """Returns parameter counts broken down by sub-module."""
        enc = sum(p.numel() for p in self.encoder.parameters())
        dec = sum(p.numel() for p in self.decoder.parameters())
        gate = (
            sum(p.numel() for p in self.feedback_gate.parameters()) + 1  # +1 for alpha
            if self.feedback_enabled else 0
        )
        return {
            "encoder" : enc,
            "decoder" : dec,
            "feedback": gate,
            "total"   : enc + dec + gate,
        }