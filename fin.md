# FIN Module Documentation

## Module Overview

The **Fractal Intelligence Network (FIN)** is a hierarchical, information-theoretically constrained neural network architecture that decomposes visual recognition into three abstraction levels with bandwidth-limited communication channels.

### Key Innovation

FIN enforces **information-theoretic constraints** on inter-level communication via stochastic bottleneck channels. Each channel limits mutual information between consecutive levels, **forcing genuine abstraction** rather than passive hierarchical composition. Without these constraints, FIN would collapse into a standard deep network—the hierarchy exists structurally but information would flow freely.

### Core Principle

The architecture implements **self-similar hierarchical processing**: each level follows the identical structural rule of encode → compress → communicate → refine, but instantiates it with different architectures matched to the representation modality at that scale.

### Primary Use Case

**CIFAR-100 Image Classification** — FIN learns a 3-level hierarchy:
- **L0 (CNN):** Raw pixel → spatial features (512D)
- **L1 (Transformer):** Spatial features → fine-grained concepts (128D) — learns 100 fine classes
- **L2 (MLP):** Fine concepts → abstract concepts (32D) — learns 20 coarse superclasses

The task tests whether progressive compression naturally produces increasingly abstract representations, as predicted by information theory.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FRACTAL INTELLIGENCE NETWORK                         │
└─────────────────────────────────────────────────────────────────────────────┘

                                INPUT IMAGE (B, 3, 32, 32)
                                         │
                                         ▼
                        ┌───────────────────────────────┐
                        │    LEVEL 0: CNN Encoder (f_0) │
                        │  Pixels → z_0 (B, 512D)       │
                        │  • Conv blocks: 3→64→128→256  │
                        │  • AdaptiveAvgPool + Linear    │
                        │  Objective O_0: pixel recon   │
                        └───────────────────────────────┘
                                    ▲    │
                                    │    │ upward
                                    │    ▼
                        ╔═══════════════════════════════╗
                        ║  CHANNEL 0→1 (Bottleneck)     ║
                        ║  • Stochastic: z_0 → μ_1, σ_1║
                        ║  • KL penalty: I(Z_0;Z_1)≤β_1 ║
                        ║  • Compression: 512 → 128D    ║
                        ╚═══════════════════════════════╝
                                    ▲    │
                                    │    │ upward
                                    │    ▼
                        ┌───────────────────────────────┐
                        │  LEVEL 1: Transformer (f_1)   │
                        │  z_0 → z_1 (B, 128D)          │
                        │  • Self-attention blocks       │
                        │  • CLS token aggregation       │
                        │  Objective O_1: fine cls (100)│
                        │  Sends top-down to L0 (g_1)   │
                        └───────────────────────────────┘
                                    ▲    │
                                    │    │ upward
                                    │    ▼
                        ╔═══════════════════════════════╗
                        ║  CHANNEL 1→2 (Bottleneck)     ║
                        ║  • Stochastic: z_1 → μ_2, σ_2║
                        ║  • KL penalty: I(Z_1;Z_2)≤β_2 ║
                        ║  • Compression: 128 → 32D     ║
                        ╚═══════════════════════════════╝
                                    ▲    │
                                    │    │ upward
                                    │    ▼
                        ┌───────────────────────────────┐
                        │   LEVEL 2: MLP Encoder (f_2)  │
                        │   z_1 → z_2 (B, 32D)          │
                        │   • ResidualMLP: 128→64→32    │
                        │   • 4x compression ratio       │
                        │   Objective O_2: coarse cls(20)
                        │   Sends top-down to L1 (g_2)  │
                        └───────────────────────────────┘
                                         │
                             ┌───────────┴───────────┐
                             ▼                       ▼
                        Fine logits              Coarse logits
                        (100 classes)            (20 classes)

BIDIRECTIONAL FLOW:
  ⬆ Bottom-up: raw signal processed through successive bottlenecks
  ⬇ Top-down: refined representations flow back down to guide lower levels

---

## Directory Structure

| File | Lines | Purpose |
|------|-------|---------|
| `fin/network/fin.py` | 514 | Main FIN orchestrator — coordinates 3 levels + 2 channels + loss assembly |
| `fin/levels/level0.py` | 418 | CNN-based raw signal processor — encodes pixels, decodes for reconstruction |
| `fin/levels/level1.py` | 500 | Transformer-based fine-concept learner — attends over compressed features |
| `fin/levels/level2.py` | 404 | MLP-based abstract-concept learner — apex level, no incoming top-down |
| `fin/channels/bottleneck.py` | 281 | Bandwidth-constrained communication — stochastic compression with KL penalty |
| `fin/losses/hierarchical.py` | 316 | Loss assembly and metrics — combines per-level objectives + bandwidth penalties |
| `fin/utils/annealing.py` | 232 | Beta annealing scheduler — tightens bandwidth budget during training |

**Total: ~2,665 lines of core architecture**

---

## Core Components

### 1. FIN Orchestrator (`fin.network.FIN`)

The top-level module that coordinates the entire hierarchy.

**Responsibilities:**
- Instantiate 3 levels (L0, L1, L2) and 2 channels with user config
- Orchestrate 3-phase forward pass: bottom-up encoding, top-down refinement, objective computation
- Optionally apply beta annealing to channel bandwidth budgets
- Aggregate per-level losses and KL penalties

**Key Methods:**
```python
forward(x, fine_labels, coarse_labels, current_betas=None)
  # x: (B, 3, 32, 32) raw pixels
  # fine_labels: (B,) CIFAR-100 fine class indices [0-99]
  # coarse_labels: (B,) CIFAR-100 coarse class indices [0-19]
  # current_betas: dict of beta overrides for annealing
  # Returns: loss_breakdown, predictions
```

**Execution phases:**
1. **Bottom-up**: x → z_0 (L0) → z_1 (channel 0→1) → classify (L1) → z_2 (channel 1→2) → classify (L2)
2. **Top-down**: z_2 message → refine L1 → L1 message → refine L0
3. **Objectives**: compute L_0 (reconstruction), L_1 (fine class), L_2 (coarse class), L_BW (bandwidth)

**Design Note:** The orchestrator is intentionally lightweight — it delegates computation to levels and channels, acting as a coordinator. This modular design enables easy ablations (disable a level, a channel, or an objective by config).

---

### 2. Level 0: CNN Raw Signal Processor

Encodes raw CIFAR-100 pixels into a spatial-feature representation space.

**Architecture:** `Level0 = (Z_0, f_0, g_0, O_0)`

```
Input (B, 3, 32, 32)
  │
  ├─ Encoder (f_0): progressively downsampling CNN
  │  ├─ ConvBlock: (3, 32, 32) → (64, 16, 16)  [stride=2]
  │  ├─ ConvBlock: (64, 16, 16) → (128, 8, 8)  [stride=2]
  │  ├─ ConvBlock: (128, 8, 8) → (256, 4, 4)   [stride=2]
  │  ├─ AdaptiveAvgPool: (256, 4, 4) → (256, 1, 1)
  │  ├─ Flatten + Linear projection
  │  └─ z_0 (B, 512)
  │
  ├─ Refine (top-down): z_0_refined = z_0 + α * gate(msg_from_L1)
  │  ├─ gate: linear projection from d1=128 → d0=512
  │  └─ α: learned scalar [0, 1], initially α_init=0.1
  │
  └─ Decoder (g_0): upsampling to reconstruct pixels
     ├─ Linear unproject: (512) → (256×4×4)
     ├─ TransposedConvBlock: (256, 4, 4) → (128, 8, 8)
     ├─ TransposedConvBlock: (128, 8, 8) → (64, 16, 16)
     ├─ TransposedConvBlock: (64, 16, 16) → (3, 32, 32)
     └─ Output (B, 3, 32, 32) with sigmoid → [0, 1]

Objective O_0: MSE(input, reconstructed)
  Ensures L0 preserves raw signal fidelity.
```

**Why CNN at L0?**
Input is spatial (images). CNNs exploit 2D locality and translation invariance. Spatial downsampling mirrors the hierarchical compression principle.

**Top-Down Feedback (Definition 1.4):**
L1's refined representation z_1_refined is projected back to L0's space and added to z_0 with a learned gate weight α:
```
z_0_refined = z_0 + α * gate(z_1_refined)
```
This allows L1 (which may have learned higher-level features) to guide L0's reconstruction — e.g., "this region is probably a dog's ear, reconstruct accordingly."

**Parameter Count:** ~150K (encoder + decoder + feedback gate)

---

### 3. Level 1: Transformer Fine-Concept Learner

Processes L0's spatially-encoded features to learn fine-grained (100-class) distinctions.

**Architecture:** `Level1 = (Z_1, f_1, g_1, O_1)`

```
Input z_0 (B, 512)
  │
  ├─ Encoder (f_1): Transformer
  │  ├─ Reshape z_0 into tokens: (B, 512) → (B, 8, 64)  [n_patches=8]
  │  ├─ Project tokens: (64) → (64) [embed_dim]
  │  ├─ Prepend learnable CLS token: (B, 8, 64) → (B, 9, 64)
  │  ├─ Add positional embeddings: sine-cosine or learnable
  │  ├─ TransformerBlock × 2 (4 heads, pre-norm)
  │  │  ├─ LayerNorm → MultiHeadAttention → residual
  │  │  └─ LayerNorm → FFN → residual
  │  ├─ Extract CLS output: (B, 9, 64) → (B, 64)
  │  ├─ Project CLS to z_1: (64) → (128)
  │  └─ z_1 (B, 128)
  │
  ├─ Refine (top-down): z_1_refined = z_1 + α_1 * gate_1(msg_from_L2)
  │  ├─ gate_1: linear projection from d2=32 → d1=128
  │  └─ α_1: learned scalar [0, 1]
  │
  ├─ Classify: z_1 → logits (B, 100)
  │  └─ Single linear layer, no bottleneck
  │
  └─ Downward projection: z_1_refined → message for L0
     └─ Linear + LayerNorm: (128) → (512)

Objective O_1: CrossEntropy(logits, fine_labels)
  Forces z_1 to be discriminative for 100 fine classes.
```

**Why Transformer at L1?**
- L0's z_0 already loses spatial structure (512D vector, not 2D feature map)
- Transformer excels at learning **relational structure** via self-attention
- Small Transformer (2 layers, 4 heads) is efficient for already-compressed input
- Token splitting (512→8 patches) gives attention something to attend over

**Token Splitting Strategy:**
```
z_0: (B, 512) → reshape to (B, 8, 64)
  • 8 patches (n_patches), each 64D (token_dim = 512/8)
  • Self-attention attends pairwise across patches
  • Learned CLS token aggregates global information
```

**Bidirectional Message Passing:**
- **Upward**: encode z_0 → z_1, send to L2 via channel
- **Downward**: receive z_2 from L2, refine z_1, send refined z_1 back to L0

**Parameter Count:** ~180K (encoder + classifier + feedback + downward projection)

---

### 4. Level 2: MLP Abstract-Concept Learner

The apex level that learns abstract (20-class coarse) categories.

**Architecture:** `Level2 = (Z_2, f_2, g_2, O_2)`

```
Input z_1 (B, 128)
  │
  ├─ Encoder (f_2): MLP with residual connections
  │  ├─ ResidualMLP(128): 128 → 128 + skip
  │  ├─ ResidualMLP(128): 128 → 128 + skip
  │  ├─ Compress: (128) → (64)
  │  ├─ ResidualMLP(64): 64 → 64 + skip
  │  ├─ Compress: (64) → (32)
  │  └─ z_2 (B, 32)
  │
  ├─ Classify: z_2 → logits (B, 20)
  │  └─ Single linear layer
  │
  └─ Downward projection: z_2 → message for L1
     └─ Linear + LayerNorm: (32) → (128)

Objective O_2: CrossEntropy(logits, coarse_labels)
  Forces z_2 to be discriminative for 20 coarse superclasses.

NOTE: No top-down refinement — L2 is the apex, nothing above it.
```

**Why MLP at L2?**
By L2, information has been:
- Spatially compressed (L0's CNN)
- Relationally processed (L1's Transformer)
- Further compressed (channel bottleneck)

Result: a 128D vector with **no remaining spatial or sequential structure**. A Transformer would add parameters without value. Deep MLP with residuals is parameter-efficient and sufficient.

**Compression Ratio:** 128D → 32D (4x compression)
This aggressive compression forces L2 to encode only the **most abstract, coarse distinctions**. Theorem 1.1 predicts that more compressed representations should learn more general concepts — CIFAR-100's 20 superclasses are indeed coarser than the 100 fine classes learned by L1.

**Design Philosophy:**
L2 demonstrates FIN's flexibility: each level chooses its architecture based on representation modality, not from a fixed blueprint. The self-similarity rule is structural (encode → compress → communicate → refine), not architectural.

**Parameter Count:** ~70K (encoder + classifier + downward projection)

---

### 5. BandwidthBottleneck Channel

The critical mechanism enforcing information-theoretic compression.

**Purpose:** Compress z_{k-1} to z_k while enforcing a soft upper bound on mutual information I(Z_{k-1}; Z_k) ≤ β_k nats.

**Mechanism:**
```
Input z_{k-1} (B, in_dim)
  │
  ├─ Trunk: shared feature extraction
  │  └─ Linear + LayerNorm + GELU: in_dim → mid_dim
  │
  ├─ Mu head: mid_dim → out_dim  [mean of Gaussian]
  │
  └─ Log-Sigma head: mid_dim → out_dim  [log variance of Gaussian]
     └─ Clamp to [log(min_sigma), max] for numerical stability

Stochastic sampling (during training):
  ε ~ N(0, I)
  z_k = μ + σ · ε  [reparameterization trick]

KL divergence penalty:
  KL(N(μ, σ²) || N(0, I)) = Σ_i (σ²_i + μ²_i - 1 - log(σ²_i))
  
Hinge loss (soft constraint):
  L_BW = γ · max(0, KL - β)
  
  • If KL ≤ β: no penalty (constraint satisfied)
  • If KL > β: linear penalty (proportional to violation)

Output: z_k (B, out_dim), kl_loss scalar
```

**Compression Ratio Example:**
- Channel 0→1: 512D → 128D (4x)
- Channel 1→2: 128D → 32D (4x)
- Total cascade: 512D → 32D (16x)

**Why Hinge, Not Hard Constraint?**
- Hard constraint (KL = β exactly) is discontinuous, hard to optimize
- Hinge is differentiable and allows graceful flexibility
- During training, network can use LESS bandwidth if it helps the objective
- No penalty if operating within budget

**Min-Sigma Floor:**
```python
sigma = exp(log_sigma)
sigma = torch.clamp(sigma, min=min_sigma, max=5.0)
```
Prevents posterior collapse (σ → 0) which would make the channel deterministic.

**Beta Annealing:**
Channels accept `beta_override` parameter in forward pass:
```python
kl_loss = self.compute_kl_loss(mu, log_sigma, beta=beta_override)
```
This allows the BetaAnnealingScheduler to tighten bandwidth dynamically without mutating channel state.

---

### 6. HierarchicalLoss

Combines per-level losses and bandwidth penalties into the total training objective.

**Formula:**
```
L_FIN = Σ_k λ_k · O_k(z_k) + Σ_k γ_k · max(0, KL_k - β_k)
        └─────────────────┬────────────────┘   └─────┬─────┘
              per-level objectives           bandwidth penalties
```

**Components:**
- **O_0**: MSE(input, reconstructed) — L0 preserves raw signal fidelity
- **O_1**: CrossEntropy(fine_logits, fine_labels) — L1 learns 100 fine classes
- **O_2**: CrossEntropy(coarse_logits, coarse_labels) — L2 learns 20 coarse classes
- **L_BW**: sum of KL penalties from channels, already computed during forward

**Weighting Strategy:**
Per-level importance can be tuned via λ_k:
```python
loss_0 = lambda_0 * O_0  # typically λ_0 = 0.1 to 1.0 (reconstruction)
loss_1 = lambda_1 * O_1  # typically λ_1 = 1.0 (primary task)
loss_2 = lambda_2 * O_2  # typically λ_2 = 0.1 to 1.0 (auxiliary)
```

**Monitoring Metrics:**
- **Fine accuracy**: top-1 accuracy on 100-class prediction
- **Coarse accuracy**: top-1 accuracy on 20-class prediction
- **Loss breakdown**: separate logging for each component

**Design Philosophy:**
Loss computation is intentionally decoupled from network modules. This enables:
- Clean ablations: set λ_k = 0 to disable a level
- Transparent loss composition visible in TensorBoard
- Easy debugging (isolate which component causes problems)

---

### 7. BetaAnnealingScheduler

Tightens bandwidth budgets during training to enforce compression gradually.

**Motivation:**
Starting with tight bandwidth constraints causes instability:
- Early in training, L1 and L2 haven't yet learned meaningful representations
- If β is tight, L0 is forced to compress into representations L1 can't decode
- All levels fail simultaneously — hierarchy never bootstraps

Solution: **curriculum learning for compression**
```
Early epochs:  large β → channels open → learn basic task performance
Middle epochs: β decreases → compression tightens gradually
Late epochs:   small β → final tight bandwidth → forced abstraction
```

**Annealing Schedules:**

1. **Exponential** (default):
   ```
   β_k(t) = β_k_max · exp(-λ · (t - warmup))
   ```
   Smooth exponential decay after warmup.

2. **Linear**:
   ```
   β_k(t) = β_k_max · (1 - (t - warmup) / (T - warmup))
   ```
   Linearly decreases from β_max to 0.

3. **Cosine**:
   ```
   β_k(t) = β_k_max · 0.5 · (1 + cos(π · (t - warmup) / (T - warmup)))
   ```
   Smooth cosine-annealed decrease.

**Usage in Training Loop:**
```python
scheduler = BetaAnnealingScheduler(channels, annealing_cfg, total_epochs)

for epoch in range(total_epochs):
    current_betas = scheduler.step(epoch)
    # current_betas is a dict: {"channel_01": beta_1, "channel_12": beta_2}
    
    for batch in dataloader:
        loss, breakdown = fin.forward(..., current_betas=current_betas)
        loss.backward()
        optimizer.step()
```

**Configuration Parameters:**
```python
annealing_cfg = {
    "enabled": True,
    "strategy": "exponential",  # or "linear", "cosine"
    "warmup_epochs": 10,        # how many epochs at full β_max
    "decay_lambda": 0.05,       # exponential decay constant
}
```

---

## Execution Flow

### Phase 1: Bottom-Up Encoding
```
Input x (B, 3, 32, 32)
  ↓
L0.encode(x) → z_0 (B, 512)
  ↓
Channel 0→1: z_0 → z_1 via stochastic bottleneck (B, 128)
  ↓
L1.encode(z_1) → z_1_refined (B, 128)  [not yet refined]
  ↓
Channel 1→2: z_1 → z_2 via stochastic bottleneck (B, 32)
  ↓
L2.encode(z_2) → z_2 (B, 32)
```

**KL penalties accumulated:**
- kl_loss_01 from Channel 0→1
- kl_loss_12 from Channel 1→2

### Phase 2: Top-Down Refinement
```
From apex L2:
  z_2 → g_2 (downward projection) → message_to_L1 (B, 128)
    ↓
L1.refine(z_1, message_to_L1) → z_1_refined
  z_1_refined → g_1 (downward projection) → message_to_L0 (B, 512)
    ↓
L0.refine(z_0, message_to_L0) → z_0_refined
```

**Refinement formula (all levels):**
```
z_k_refined = z_k + α_k * gate_k(message_from_above)
  where α_k ∈ [0, 1] is a learned scalar gate
```

### Phase 3: Objective Computation
```
L0 objective:
  x_recon = L0.decode(z_0_refined)
  O_0 = MSE(x, x_recon)

L1 objective:
  fine_logits = L1.classify(z_1_refined)
  O_1 = CrossEntropy(fine_logits, fine_labels)

L2 objective:
  coarse_logits = L2.classify(z_2)
  O_2 = CrossEntropy(coarse_logits, coarse_labels)

Total loss:
  L_FIN = λ_0·O_0 + λ_1·O_1 + λ_2·O_2 + γ·(kl_loss_01 + kl_loss_12)
```

---

## Design Decisions

### 1. Why Three Levels?

**Answer:** Three is sufficient to demonstrate hierarchical abstraction:
- L0: captures low-level spatial structure (pixels → features)
- L1: captures mid-level relational structure (fine concepts)
- L2: captures high-level abstract structure (coarse concepts)

Going deeper (4+ levels) would add computational cost for CIFAR-100's limited complexity. Three is a sweet spot for the dataset.

### 2. Why Bandwidth Constraints?

**Answer:** Without bandwidth limits, FIN collapses into a standard deep network:
- Information flows freely through levels
- Hierarchy is structural but not functional
- No emergent abstraction — just progressive elaboration

The bottleneck channels **force** information pruning, making each level encode genuinely different aspects of the input.

### 3. Why Annealing?

**Answer:** Tight bandwidth from the start causes training collapse:
- Early epochs: L1, L2 haven't yet learned useful representations
- If β is tight, L0 compresses into representations L1 can't use
- Solution: loosen β early, tighten later (curriculum learning)

### 4. Why These Specific Architectures?

| Level | Architecture | Reason |
|-------|--------------|--------|
| L0 | CNN | Input is spatial (images); CNNs exploit 2D locality |
| L1 | Transformer | z_0 is dense vector; Transformer learns relational structure via attention |
| L2 | MLP | z_1 is already compressed; no remaining spatial/sequential structure; pure functional transformation |

Each level uses the best architecture for its *input modality*, not from a fixed template.

### 5. Why Residual Feedback?

```
z_k_refined = z_k + α_k * gate_k(message_from_above)
```

**Answer:**
- **Non-destructive**: always preserves z_k if gate is weak
- **Trainable gate α**: network controls how much to trust the top-down prior
- **Consistent with compression principle**: message_from_above is more compressed (less information), so it should refine, not replace
- **Stable optimization**: easier to learn small corrections than large replacements

### 6. Why No Top-Down Feedback to L2?

**Answer:** L2 is the apex — nothing above it to send guidance. Architecturally honest: don't add mechanisms that can't be used.

---

## Integration with Training Loop

### Minimal Example

```python
import torch
import torch.optim as optim
from fin.network.fin import FIN
from fin.losses.hierarchical import HierarchicalLoss
from fin.utils.annealing import BetaAnnealingScheduler

# Create FIN
config = {
    "level0": {...},
    "level1": {...},
    "level2": {...},
    "channel": {"beta_01": 8.0, "beta_12": 4.0, ...},
    "loss": {"lambda_0": 0.5, "lambda_1": 1.0, "lambda_2": 0.5},
    "annealing": {"enabled": True, "strategy": "exponential", ...}
}
fin = FIN(config)
loss_fn = HierarchicalLoss(config)
scheduler = BetaAnnealingScheduler(fin.channels, config["annealing"], total_epochs=200)

optimizer = optim.Adam(fin.parameters(), lr=1e-3)

# Training loop
for epoch in range(200):
    for batch in train_loader:
        x, fine_labels, coarse_labels = batch
        
        # Annealing
        current_betas = scheduler.step(epoch)
        
        # Forward pass
        loss, breakdown = fin(x, fine_labels, coarse_labels, current_betas)
        
        # Backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Logging
        for key, val in breakdown.to_dict().items():
            tb_logger.log_scalar(f"train/{key}", val, global_step)
```

**Key Integration Points:**
1. **Annealing**: call `scheduler.step(epoch)` before forward pass to get current β values
2. **Loss**: `loss_fn()` combines all objectives and penalties
3. **Logging**: `breakdown.to_dict()` provides all metrics for TensorBoard

---

## Extension Points

### A. New Datasets

To adapt FIN to a new dataset (e.g., ImageNet, MNIST, medical imaging):

1. **Modify L0 architecture** for input characteristics:
   - ImageNet (224×224): add more conv layers
   - MNIST (28×28): reduce conv layers
   - Medical 3D volumes: use 3D convolutions

2. **Adjust dimensionalities**:
   - d0 (L0 output): depends on input size and desired compression
   - d1, d2: follow from channel bandwidth budgets

3. **Update L1/L2 objectives**:
   - If dataset has 10 classes instead of 100: update classifier heads
   - If no coarse labels: set lambda_2 = 0 (disable L2)
   - If multi-label: replace CrossEntropy with BCE

4. **Retune β budgets**:
   - Larger dataset → may tolerate tighter bandwidth
   - Different modality → may need different compression ratios

### B. New Level Architectures

To replace a level (e.g., use a Vision Transformer instead of CNN at L0):

1. Ensure the new architecture preserves the interface:
   ```python
   class Level0(nn.Module):
       def encode(self, x) -> z_0: ...         # required
       def refine(self, z_0, msg) -> z_0_ref: ... # required
       def decode(self, z_0) -> x_recon: ...   # required
       def forward(self, x, msg=None): ...     # required
   ```

2. Update L0's dimensionality to match downstream L1 input

3. Retune feedback gate projections

### C. Alternative Loss Schemes

To modify the loss (e.g., add contrastive learning, KL divergence to class prior):

1. Extend `HierarchicalLoss` or create a new loss class
2. Compute additional loss terms using FIN's per-level representations
3. Combine with existing per-level objectives

Example:
```python
class ContrastiveFINLoss(HierarchicalLoss):
    def forward(self, z_0, z_1, z_2, ..., current_betas=None):
        base_loss, breakdown = super().forward(...)
        
        # Add contrastive loss between representations
        contrastive = self.compute_contrastive(z_1, z_2)
        total = base_loss + 0.1 * contrastive
        
        breakdown.contrastive = contrastive
        return total, breakdown
```

---

## Key Formulas

### KL Divergence for Diagonal Gaussian

```
z = μ + σ · ε  where ε ~ N(0, I)
KL(N(μ, diag(σ²)) || N(0, I)) = Σ_i (σ²_i + μ²_i - 1 - log(σ²_i))
                                = 0.5 · Σ_i (σ²_i + μ²_i - 1 - 2·log(σ_i))
```

Numerically stable implementation:
```python
kl = 0.5 * torch.sum(
    torch.exp(2 * log_sigma) + mu**2 - 1 - 2 * log_sigma,
    dim=1
)
```

### Soft Bandwidth Constraint

```
L_BW = γ · max(0, KL - β)
  • KL ≤ β: no penalty (constraint satisfied)
  • KL > β: penalty grows linearly with violation
  • γ: tightness of constraint (larger γ → stricter enforcement)
```

### Hierarchical Loss

```
L_FIN = Σ_k λ_k · O_k + Σ_j γ_j · max(0, KL_j - β_j)

L0 component:
  O_0 = MSE(x, g_0(z_0_refined))

L1 component:
  O_1 = CE(f_1(z_1_refined), fine_labels)

L2 component:
  O_2 = CE(f_2(z_2), coarse_labels)
```

### Refinement Gate

```
z_k_refined = z_k + α_k · gate_k(message_from_above)

α_k ∈ [0, 1]  (clamped)
gate_k: linear projection (usually d_upper → d_k)
message_from_above: output from g_{k+1} downward projection
```

### Beta Annealing (Exponential)

```
β_k(t) = β_k_max · exp(-λ · max(0, t - warmup))

• For t < warmup: β_k(t) = β_k_max  (no constraint)
• For t ≥ warmup: exponential decay with rate λ
• λ ∈ [0.01, 0.1] typical range
```

---

## Quick Reference Table

### Dimension Flow

| Component | Input Shape | Output Shape | Parameters |
|-----------|-------------|--------------|------------|
| L0 Encoder | (B, 3, 32, 32) | (B, 512) | ~70K |
| Channel 0→1 | (B, 512) | (B, 128) | ~38K |
| L1 Encoder | (B, 128) | (B, 128) | ~60K |
| L1 Classifier | (B, 128) | (B, 100) | ~13K |
| Channel 1→2 | (B, 128) | (B, 32) | ~10K |
| L2 Encoder | (B, 32) | (B, 32) | ~25K |
| L2 Classifier | (B, 32) | (B, 20) | ~0.6K |
| **Total** | - | - | **~217K** |

### Loss Components

| Term | Equation | Weight | Typical Range |
|------|----------|--------|----------------|
| L_0 | MSE(x, decode(z_0)) | λ_0 | [0.1, 1.0] |
| L_1 | CE(classify(z_1), fine_labels) | λ_1 | [0.5, 2.0] |
| L_2 | CE(classify(z_2), coarse_labels) | λ_2 | [0.1, 1.0] |
| L_BW | Σ γ_k·max(0, KL_k - β_k) | 1.0 | - |

### Configuration Defaults

| Parameter | Default | Role |
|-----------|---------|------|
| d0 (L0 output) | 512 | first bottleneck input |
| d1 (L1 output) | 128 | second bottleneck input; fine classifier input |
| d2 (L2 output) | 32 | coarse classifier input |
| β_01 (channel 0→1) | 8.0 | mutual information budget in nats |
| β_12 (channel 1→2) | 4.0 | mutual information budget in nats |
| γ (tightness) | 1.0 | constraint enforcement strength |
| α_init (feedback) | 0.1 | initial top-down gate weight |
| min_sigma | 1e-4 | floor on channel variance |
| warmup_epochs | 10 | epochs before beta annealing starts |
| decay_lambda | 0.05 | exponential decay constant |

---

## Frequently Asked Questions

### Q: Why do we need bandwidth constraints if we're learning anyway?

**A:** Without constraints, FIN is just a deep network. The network would learn end-to-end but wouldn't necessarily learn hierarchical abstractions. The bottleneck **forces** the network to discard information at each level, leading to qualitatively different feature sets at each level. Information theory predicts this should produce coarser concepts at higher levels — that's what we measure.

### Q: Can I train FIN without beta annealing?

**A:** Yes, but not recommended. Annealing helps the hierarchy bootstrap by starting loose and tightening over time. Without it, you can often achieve reasonable test accuracy, but the hierarchy may not genuinely produce different abstraction levels — ablations would show that L1 and L2 are redundant. With annealing, you should see clean separation: L1 specializes in fine classification, L2 in coarse classification.

### Q: What if I want to disable a level?

**A:** Set its loss weight to 0 in the config:
```python
config["loss"]["lambda_2"] = 0.0  # disable L2 objectives
```
This removes its contribution to the total loss. The level still exists in the forward pass but isn't optimized directly — it's only refined via top-down feedback from above (or is the apex). This is a useful ablation.

### Q: How sensitive is FIN to the β values?

**A:** Very sensitive. Too tight (small β) → channels starve, training fails. Too loose (large β) → constraints ignored, no compression enforced. Good starting points:
- β_01 = 8.0 nats (mild compression, 512→128)
- β_12 = 4.0 nats (tighter compression, 128→32)

Start here and tune based on your observed KL vs. accuracy trade-off.

### Q: Can I use FIN for tasks other than CIFAR-100?

**A:** Yes. The architecture is general-purpose, but you'll need to:
1. Adjust input/output dimensions for your dataset
2. Modify the per-level objectives (e.g., if no multi-level labels available)
3. Retune β values and λ weights for your task
4. Potentially replace L0's CNN with a different encoder if input modality differs

---

## Summary

The Fractal Intelligence Network is a theoretically-motivated hierarchical architecture that combines:
- **Three abstraction levels** with different architectures matched to representation modality
- **Two bandwidth-limited channels** enforcing information-theoretic compression
- **Bidirectional message passing** enabling top-down refinement of bottom-up features
- **Hierarchical loss assembly** balancing per-level objectives against bandwidth penalties
- **Beta annealing** for curriculum learning of compression

Key insight: FIN demonstrates that information-theoretic constraints force genuine hierarchical abstraction, producing qualitatively different feature sets at each level as predicted by theory. This is validated on CIFAR-100 where L1 learns fine-grained concepts and L2 learns abstract superclasses, despite both operating on the same input.

