"""
fin/levels/level2.py

Level 2 — Abstract Concept / Apex Level (Definition 1.1)
=========================================================

L2 = (Z_2, f_2, g_2, O_2)

    Z_2 : R^(B x d2=32)  — most compressed representation space
    f_2 : z_1 -> z_2      — MLP encoder (upward message)
    g_2 : z_2 -> z_1_space — linear projection (downward message to L1)
    O_2 : coarse classification loss (CrossEntropy, 20 superclasses)
          Forces L2 to encode category-level abstractions.

Why an MLP at Level 2?
-----------------------
By the time information reaches L2 it has already been:
  - Spatially compressed by L0's CNN
  - Relationally processed by L1's Transformer
  - Further compressed by the L1->L2 bandwidth channel (beta_2=4.0 nats)

What remains is a dense, highly compressed 128-dim vector. There are no
spatial or sequential relationships left to exploit — self-attention would
add parameters without adding value. A deep MLP with residual connections
is the right tool: purely functional transformation with maximum parameter
efficiency.

This also demonstrates FIN's flexibility: the self-similarity condition
(Definition 1.5) governs the *structural rule* (encode->compress->communicate
->refine), not the specific layer type. Each level chooses the best
architecture for its input modality.

Apex level properties:
  - NO incoming top-down feedback (nothing above L2)
  - ONLY sends downward to L1 via g_2
  - Coarsest objective: 20 superclasses vs L1's 100 fine classes
  - Smallest representation: d2=32 vs d1=128 vs d0=512
  - Most aggressive compression ratio: 128 -> 32 (4x)

CIFAR-100 superclass structure:
  The 100 fine classes map to 20 coarse superclasses
  (e.g. "beaver, dolphin, otter, seal, whale" -> "aquatic mammals").
  L2 learning superclasses while L1 learns fine classes is a natural
  test of whether the hierarchy genuinely produces different abstraction
  levels — exactly what Theorem 1.1 predicts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# =============================================================================
# Building Blocks
# =============================================================================

class ResidualMLP(nn.Module):
    """
    MLP block with residual connection.

    Structure: LayerNorm -> Linear -> GELU -> Dropout -> Linear -> residual

    Using residual connections even in a small MLP because:
    1. Stabilizes training when combined with aggressive compression
    2. Allows the network to learn identity + correction rather than
       full transformation — useful when z_1 already carries good signal
    3. Consistent with the residual refinement idea in Definition 1.4

    Args:
        dim     (int)  : input and output dimension (residual requires same dim)
        hidden  (int)  : hidden layer dimension
        dropout (float): dropout rate
    """

    def __init__(self, dim: int, hidden: int, dropout: float = 0.1):
        super().__init__()

        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


# =============================================================================
# Level 2 Encoder
# =============================================================================

class Level2Encoder(nn.Module):
    """
    f_2: z_1 -> z_2

    Maps (B, d1=128) -> (B, d2=32).

    Architecture:
        ResidualMLP x2  : (128) -> (128)  — refine z_1 in its own space
        Compression     : (128) -> (64)   — first compression step
        ResidualMLP x1  : (64)  -> (64)   — refine at intermediate scale
        Compression     : (64)  -> (32)   — final compression to d2
        LayerNorm       : stabilize output z_2

    Two-stage compression (128->64->32) rather than direct (128->32)
    because a single large compression step loses information abruptly.
    Gradual compression gives the network a chance to reorganize
    representations before the final bottleneck.

    Args:
        in_dim     (int)  : d1
        out_dim    (int)  : d2
        hidden_dim (int)  : intermediate MLP hidden dimension
        dropout    (float): dropout rate
    """

    def __init__(
        self,
        in_dim    : int   = 128,
        out_dim   : int   = 32,
        hidden_dim: int   = 64,
        dropout   : float = 0.1,
    ):
        super().__init__()

        assert out_dim < in_dim, (
            f"Level2Encoder requires out_dim ({out_dim}) < in_dim ({in_dim})"
        )

        mid_dim = (in_dim + out_dim) // 2   # e.g. (128+32)//2 = 80, rounded to 64

        # Stage 1: refine in z_1's space
        self.refine_stage = nn.Sequential(
            ResidualMLP(in_dim, hidden_dim, dropout),
            ResidualMLP(in_dim, hidden_dim, dropout),
        )

        # Stage 2: first compression (in_dim -> mid_dim)
        self.compress1 = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.GELU(),
        )

        # Stage 3: refine at intermediate scale
        self.refine_mid = ResidualMLP(mid_dim, hidden_dim // 2, dropout)

        # Stage 4: final compression (mid_dim -> out_dim)
        self.compress2 = nn.Sequential(
            nn.Linear(mid_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

        self.out_dim = out_dim

    def forward(self, z_1: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_1: shape (B, d1)
        Returns:
            z_2: shape (B, d2)
        """
        h = self.refine_stage(z_1)
        h = self.compress1(h)
        h = self.refine_mid(h)
        z_2 = self.compress2(h)
        return z_2


# =============================================================================
# Level 2 Classifier Head (Objective O_2)
# =============================================================================

class Level2ClassifierHead(nn.Module):
    """
    Coarse classification head for O_2.
    Maps z_2 -> logits over 20 CIFAR-100 superclasses.

    Note the contrast with L1's head:
      L1: z_1 (128-dim) -> 100 fine classes
      L2: z_2  (32-dim) -> 20 coarse classes

    The fact that a 32-dim vector is expected to distinguish 20 superclasses
    while a 128-dim vector distinguishes 100 fine classes is precisely the
    abstraction hierarchy Theorem 1.1 predicts — more compressed
    representations encode coarser, more general distinctions.

    Args:
        in_dim        (int): d2
        num_classes   (int): 20 for CIFAR-100 coarse labels
    """

    def __init__(self, in_dim: int = 32, num_classes: int = 20):
        super().__init__()
        self.head = nn.Linear(in_dim, num_classes)

    def forward(self, z_2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_2: shape (B, d2)
        Returns:
            logits: shape (B, num_classes)
        """
        return self.head(z_2)


# =============================================================================
# Level 2 — Full Module (Definition 1.1 + 1.4)
# =============================================================================

class Level2(nn.Module):
    """
    Full Level 2 module: MLP encoder + coarse classifier +
    downward projection to L1.

    Implements Definition 1.1:
        L_2 = (Z_2, f_2, g_2, O_2)

    And Definition 1.4 (downward only — apex level has no incoming top-down):
        Upward  : z_2 = f_2(z_1)           (output — apex, no higher level)
        Downward: top_down_to_L1 = g_2(z_2) (to L1's refine())

    Key difference from L0 and L1:
        L2 has NO refine() method — there is no Level 3 above it.
        It only sends top-down, never receives it.
        This is architecturally honest: the apex of the fractal
        hierarchy has no higher-level prior to incorporate.

    Args:
        in_dim            (int)  : d1
        out_dim           (int)  : d2
        hidden_dim        (int)  : MLP hidden dimension
        dropout           (float): dropout rate
        num_coarse_classes(int)  : 20 for CIFAR-100
    """

    def __init__(
        self,
        in_dim            : int   = 128,
        out_dim           : int   = 32,
        hidden_dim        : int   = 64,
        dropout           : float = 0.1,
        num_coarse_classes: int   = 20,
    ):
        super().__init__()

        self.out_dim = out_dim
        self.in_dim  = in_dim

        # f_2: upward encoder
        self.encoder = Level2Encoder(
            in_dim     = in_dim,
            out_dim    = out_dim,
            hidden_dim = hidden_dim,
            dropout    = dropout,
        )

        # O_2: coarse classification head
        self.classifier = Level2ClassifierHead(
            in_dim      = out_dim,
            num_classes = num_coarse_classes,
        )

        # g_2: downward projection — sends z_2 back to L1's space (d1)
        # L1 will use this as its top_down_msg in refine()
        # Note: projecting from small (d2=32) back to larger (d1=128)
        # The top-down message is an "expectation" or "prior" — it doesn't
        # need to carry full information, just enough to guide L1's refinement.
        self.downward_proj = nn.Sequential(
            nn.Linear(out_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
        )

    # -------------------------------------------------------------------------
    # Upward pass
    # -------------------------------------------------------------------------

    def encode(self, z_1: torch.Tensor) -> torch.Tensor:
        """
        Bottom-up pass: z_1 -> z_2.

        Args:
            z_1: shape (B, d1)
        Returns:
            z_2: shape (B, d2)
        """
        return self.encoder(z_1)

    # -------------------------------------------------------------------------
    # Downward message to L1 (Definition 1.4)
    # -------------------------------------------------------------------------

    def downward_message(self, z_2: torch.Tensor) -> torch.Tensor:
        """
        g_2: produce top-down message for L1.

        Projects z_2 back into L1's representation space (d1).
        L1 will use this in its refine() call.

        Real-life analogy: the front-page editor (L2) has decided the
        story is about "natural disasters" (coarse concept). They send
        a brief back to the journalist (L1): "focus on the flood angle,
        not the political response". The journalist refines their article
        accordingly — more specific, but guided by the global framing.

        Args:
            z_2: shape (B, d2)
        Returns:
            msg: shape (B, d1)
        """
        return self.downward_proj(z_2)

    # -------------------------------------------------------------------------
    # Objective O_2: coarse classification loss
    # -------------------------------------------------------------------------

    def classify(self, z_2: torch.Tensor) -> torch.Tensor:
        """
        Compute coarse-class logits from z_2.

        Args:
            z_2: shape (B, d2)
        Returns:
            logits: shape (B, num_coarse_classes)
        """
        return self.classifier(z_2)

    @staticmethod
    def objective(
        logits: torch.Tensor,
        coarse_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        O_2: coarse classification loss.
        CrossEntropy over 20 CIFAR-100 superclasses.

        Args:
            logits       : shape (B, 20)
            coarse_labels: shape (B,) — integer superclass indices
        Returns:
            loss: scalar tensor
        """
        return F.cross_entropy(logits, coarse_labels)

    # -------------------------------------------------------------------------
    # Convenience: full forward
    # -------------------------------------------------------------------------

    def forward(
        self,
        z_1: torch.Tensor,
        coarse_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full L2 forward pass.

        Note: no top_down_msg argument — L2 is the apex.
        No refine() step here.

        Steps:
            1. Encode z_1 -> z_2
            2. Classify z_2 -> coarse logits
            3. Compute coarse classification loss
            4. Compute downward message for L1

        Args:
            z_1          : from L1 via bandwidth channel, shape (B, d1)
            coarse_labels: CIFAR-100 coarse labels, shape (B,)

        Returns:
            z_2          : apex representation, shape (B, d2)
            coarse_logits: shape (B, 20)
            loss_2       : coarse classification loss scalar
            down_msg     : top-down message for L1, shape (B, d1)
        """
        z_2           = self.encode(z_1)
        coarse_logits = self.classify(z_2)
        loss_2        = self.objective(coarse_logits, coarse_labels)
        down_msg      = self.downward_message(z_2)

        return z_2, coarse_logits, loss_2, down_msg

    # -------------------------------------------------------------------------
    # Parameter count utility
    # -------------------------------------------------------------------------

    def param_count(self) -> dict:
        enc  = sum(p.numel() for p in self.encoder.parameters())
        clf  = sum(p.numel() for p in self.classifier.parameters())
        down = sum(p.numel() for p in self.downward_proj.parameters())
        return {
            "encoder"      : enc,
            "classifier"   : clf,
            "downward_proj": down,
            "total"        : enc + clf + down,
        }