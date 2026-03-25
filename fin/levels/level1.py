"""
fin/levels/level1.py

Level 1 — Local Pattern / Fine Concept (Definition 1.1)
========================================================

L1 = (Z_1, f_1, g_1, O_1)

    Z_1 : R^(B x d1=128)  — intermediate representation space
    f_1 : z_0 -> z_1       — Small Transformer encoder (upward message)
    g_1 : z_1 -> z_0_space — linear projection (downward message to L0)
    O_1 : fine-grained classification loss (CrossEntropy, 100 classes)
          Forces L1 to encode class-discriminative features.

Why a Transformer at Level 1?
------------------------------
L0 (CNN) captures spatial structure — where things are.
L1 needs to capture relational structure — what things mean relative
to each other. Transformers are well-suited for this because self-attention
explicitly models pairwise relationships.

However, we keep it SMALL (2 layers, 4 heads) because:
  1. z_0 is already compressed to d0=512 — not a long token sequence
  2. Kaggle T4 compute budget
  3. The bandwidth channel above L1 forces compression anyway —
     a large transformer would just waste parameters on information
     that gets discarded at the channel.

Input treatment:
  z_0 arrives as a single vector (B, d0). We treat it as a sequence of
  n_patches "tokens" by splitting the d0-dimensional vector into chunks.
  This gives the self-attention mechanism something to attend over, while
  keeping the architecture simple.

  d0=512, n_patches=8 -> token_dim=64 per patch token.

Top-down connections (Definition 1.4):
  L1 receives top-down from L2:   z_1_refined = z_1 + alpha_1 * gate_1(z_2)
  L1 sends top-down to L0:        top_down_msg_to_L0 = project_down(z_1)

Self-similarity (Definition 1.5):
  The structural rule encode -> compress -> communicate -> refine
  is identical to L0. Only the instantiation (Transformer vs CNN) differs
  to match the representation modality at this scale.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# =============================================================================
# Building Blocks
# =============================================================================

class TransformerBlock(nn.Module):
    """
    Single Transformer block: LayerNorm -> MultiHeadAttention -> residual
                              LayerNorm -> FFN -> residual

    Pre-norm formulation (norm before attention) — more stable than
    post-norm for small transformers with limited data.

    Args:
        dim      (int): token embedding dimension
        n_heads  (int): number of attention heads (dim must be divisible by n_heads)
        mlp_ratio(float): FFN hidden dim = dim * mlp_ratio
        dropout  (float): dropout on attention weights and FFN
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        assert dim % n_heads == 0, (
            f"dim ({dim}) must be divisible by n_heads ({n_heads})"
        )

        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(
            embed_dim   = dim,
            num_heads   = n_heads,
            dropout     = dropout,
            batch_first = True,   # (B, T, D) convention throughout
        )

        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: token sequence, shape (B, T, dim)
        Returns:
            x: updated token sequence, shape (B, T, dim)
        """
        # Self-attention with pre-norm
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # FFN with pre-norm
        x = x + self.ffn(self.norm2(x))
        return x


# =============================================================================
# Level 1 Encoder
# =============================================================================

class Level1Encoder(nn.Module):
    """
    f_1: z_0 -> z_1

    Maps (B, d0=512) -> (B, d1=128).

    Strategy: treat z_0 as a sequence of n_patches tokens,
    apply self-attention, then aggregate and project to d1.

    Steps:
        1. Split z_0 into n_patches tokens of size token_dim = d0 / n_patches
        2. Project each token to embed_dim (internal attention dimension)
        3. Add learnable [CLS] token (carries global summary)
        4. Apply num_layers Transformer blocks
        5. Extract [CLS] token -> project to d1

    Args:
        in_dim      (int): d0
        out_dim     (int): d1
        n_patches   (int): number of tokens to split z_0 into
        embed_dim   (int): internal attention dimension
        n_heads     (int): attention heads
        num_layers  (int): number of Transformer blocks
        dropout     (float): attention + FFN dropout
    """

    def __init__(
        self,
        in_dim    : int   = 512,
        out_dim   : int   = 128,
        n_patches : int   = 8,
        embed_dim : int   = 64,
        n_heads   : int   = 4,
        num_layers: int   = 2,
        dropout   : float = 0.1,
        mlp_mode  : bool  = False,
    ):
        super().__init__()

        self.mlp_mode  = mlp_mode
        self.out_dim   = out_dim

        if mlp_mode:
            # no_self_similarity ablation: replace Transformer with plain MLP
            # Same parameter budget as Transformer (roughly), no self-attention
            hidden = (in_dim + out_dim) * 2
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden // 2, out_dim),
                nn.LayerNorm(out_dim),
            )
        else:
            assert in_dim % n_patches == 0, (
                f"in_dim ({in_dim}) must be divisible by n_patches ({n_patches}). "
                f"Got {in_dim} % {n_patches} = {in_dim % n_patches}."
            )

            self.n_patches  = n_patches
            self.token_dim  = in_dim // n_patches
            self.embed_dim  = embed_dim

            self.token_proj = nn.Sequential(
                nn.Linear(self.token_dim, embed_dim),
                nn.LayerNorm(embed_dim),
            )

            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

            self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

            self.blocks = nn.ModuleList([
                TransformerBlock(dim=embed_dim, n_heads=n_heads, dropout=dropout)
                for _ in range(num_layers)
            ])

            self.norm = nn.LayerNorm(embed_dim)

            self.out_proj = nn.Sequential(
                nn.Linear(embed_dim, out_dim),
                nn.LayerNorm(out_dim),
            )

    def forward(self, z_0: torch.Tensor) -> torch.Tensor:
        if self.mlp_mode:
            return self.mlp(z_0)

        B = z_0.size(0)
        tokens = z_0.view(B, self.n_patches, self.token_dim)
        tokens = self.token_proj(tokens)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        cls_out = tokens[:, 0, :]
        z_1 = self.out_proj(cls_out)
        return z_1


# =============================================================================
# Level 1 Classifier Head (Objective O_1)
# =============================================================================

class Level1ClassifierHead(nn.Module):
    """
    Fine-grained classification head for O_1.
    Maps z_1 -> logits over 100 CIFAR-100 classes.

    Kept deliberately simple — a single linear layer.
    The representation z_1 should do the heavy lifting,
    not the classifier head.

    Args:
        in_dim      (int): d1
        num_classes (int): 100 for CIFAR-100 fine labels
    """

    def __init__(self, in_dim: int = 128, num_classes: int = 100):
        super().__init__()
        self.head = nn.Linear(in_dim, num_classes)

    def forward(self, z_1: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_1: shape (B, d1)
        Returns:
            logits: shape (B, num_classes)
        """
        return self.head(z_1)


# =============================================================================
# Level 1 — Full Module (Definition 1.1 + 1.4)
# =============================================================================

class Level1(nn.Module):
    """
    Full Level 1 module: Transformer encoder + classifier head +
    top-down feedback gate + downward projection to L0.

    Implements Definition 1.1:
        L_1 = (Z_1, f_1, g_1, O_1)

    And Definition 1.4 (bidirectional message passing):
        Upward  : z_1 = f_1(z_0)                               (to L2 via channel)
        Downward: top_down_to_L0 = g_1(z_1)                    (to L0)
        Refinement: z_1_refined = z_1 + alpha_1 * gate_1(z_2)  (from L2)

    Args:
        in_dim          (int)  : d0 — dimensionality of incoming z_0
        out_dim         (int)  : d1 — dimensionality of z_1
        n_patches       (int)  : token split count for self-attention
        embed_dim       (int)  : internal attention dimension
        n_heads         (int)  : attention heads
        num_layers      (int)  : transformer depth
        dropout         (float): dropout rate
        num_fine_classes(int)  : number of fine classes (100 for CIFAR-100)
        feedback_dim    (int)  : d2 — dim of top-down message from L2
        alpha_init      (float): initial top-down gate scalar
        feedback_enabled(bool) : if False, disables top-down (ablation)
    """

    def __init__(
        self,
        in_dim          : int   = 512,
        out_dim         : int   = 128,
        n_patches       : int   = 8,
        embed_dim       : int   = 64,
        n_heads         : int   = 4,
        num_layers      : int   = 2,
        dropout         : float = 0.1,
        num_fine_classes: int   = 100,
        feedback_dim    : int   = 32,
        alpha_init      : float = 0.1,
        feedback_enabled: bool  = True,
        mlp_mode        : bool = False,
    ):
        super().__init__()

        self.out_dim          = out_dim
        self.in_dim           = in_dim
        self.feedback_enabled = feedback_enabled

        # f_1: upward encoder
        self.encoder = Level1Encoder(
            in_dim    = in_dim,
            out_dim   = out_dim,
            n_patches = n_patches,
            embed_dim = embed_dim,
            n_heads   = n_heads,
            num_layers= num_layers,
            dropout   = dropout,
            mlp_mode  = mlp_mode,
        )

        # O_1: fine-grained classification head
        self.classifier = Level1ClassifierHead(
            in_dim      = out_dim,
            num_classes = num_fine_classes,
        )

        # g_1: downward projection — sends z_1 back to L0's space
        # L0 will use this as its top_down_msg in refine()
        self.downward_proj = nn.Sequential(
            nn.Linear(out_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
        )

        # Top-down feedback gate — receives z_2 from L2 (Definition 1.4)
        if feedback_enabled:
            self.feedback_gate = nn.Sequential(
                nn.Linear(feedback_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
            )
            self.alpha = nn.Parameter(torch.tensor(alpha_init))

    # -------------------------------------------------------------------------
    # Upward pass
    # -------------------------------------------------------------------------

    def encode(self, z_0: torch.Tensor) -> torch.Tensor:
        """
        Bottom-up pass: z_0 -> z_1.

        Args:
            z_0: shape (B, d0)
        Returns:
            z_1: shape (B, d1)
        """
        return self.encoder(z_0)

    # -------------------------------------------------------------------------
    # Top-down refinement (Definition 1.4)
    # -------------------------------------------------------------------------

    def refine(
        self,
        z_1: torch.Tensor,
        top_down_msg: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Refine z_1 using top-down message from L2.

        z_1_refined = z_1 + alpha_1 * gate_1(top_down_msg)

        Args:
            z_1          : bottom-up representation, shape (B, d1)
            top_down_msg : from L2, shape (B, d2)  [optional]
        Returns:
            z_1_refined  : shape (B, d1)
        """
        if not self.feedback_enabled or top_down_msg is None:
            return z_1

        feedback = self.feedback_gate(top_down_msg)
        alpha    = self.alpha.clamp(0.0, 1.0)
        return z_1 + alpha * feedback

    # -------------------------------------------------------------------------
    # Downward message to L0 (Definition 1.4)
    # -------------------------------------------------------------------------

    def downward_message(self, z_1: torch.Tensor) -> torch.Tensor:
        """
        g_1: produce top-down message for L0.

        Projects z_1 back into L0's representation space (d0)
        so L0 can use it in its refine() call.

        Args:
            z_1: shape (B, d1)
        Returns:
            msg: shape (B, d0)
        """
        return self.downward_proj(z_1)

    # -------------------------------------------------------------------------
    # Objective O_1: fine classification loss
    # -------------------------------------------------------------------------

    def classify(self, z_1: torch.Tensor) -> torch.Tensor:
        """
        Compute fine-class logits from z_1.

        Args:
            z_1: shape (B, d1)
        Returns:
            logits: shape (B, num_fine_classes)
        """
        return self.classifier(z_1)

    @staticmethod
    def objective(
        logits: torch.Tensor,
        fine_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        O_1: fine-grained classification loss.
        CrossEntropy over 100 CIFAR-100 classes.

        Args:
            logits     : shape (B, 100)
            fine_labels: shape (B,) — integer class indices
        Returns:
            loss: scalar tensor
        """
        return F.cross_entropy(logits, fine_labels)

    # -------------------------------------------------------------------------
    # Convenience: full forward
    # -------------------------------------------------------------------------

    def forward(
        self,
        z_0: torch.Tensor,
        fine_labels: torch.Tensor,
        top_down_msg: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full L1 forward pass.

        Steps:
            1. Encode z_0 -> z_1
            2. Refine z_1 with top-down from L2 (if provided)
            3. Classify z_1 -> fine logits
            4. Compute fine classification loss
            5. Compute downward message for L0

        Args:
            z_0          : from L0 via bandwidth channel, shape (B, d0)
            fine_labels  : CIFAR-100 fine labels, shape (B,)
            top_down_msg : from L2, shape (B, d2) [optional]

        Returns:
            z_1_refined  : refined representation, shape (B, d1)
            fine_logits  : shape (B, 100)
            loss_1       : fine classification loss scalar
            down_msg     : top-down message for L0, shape (B, d0)
        """
        z_1         = self.encode(z_0)
        z_1_refined = self.refine(z_1, top_down_msg)
        fine_logits = self.classify(z_1_refined)
        loss_1      = self.objective(fine_logits, fine_labels)
        down_msg    = self.downward_message(z_1_refined)

        return z_1_refined, fine_logits, loss_1, down_msg

    # -------------------------------------------------------------------------
    # Parameter count utility
    # -------------------------------------------------------------------------

    def param_count(self) -> dict:
        enc  = sum(p.numel() for p in self.encoder.parameters())
        clf  = sum(p.numel() for p in self.classifier.parameters())
        down = sum(p.numel() for p in self.downward_proj.parameters())
        gate = (
            sum(p.numel() for p in self.feedback_gate.parameters()) + 1
            if self.feedback_enabled else 0
        )
        return {
            "encoder"       : enc,
            "classifier"    : clf,
            "downward_proj" : down,
            "feedback"      : gate,
            "total"         : enc + clf + down + gate,
        }