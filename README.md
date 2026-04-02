# Fractal Intelligence Networks (FIN)

> **"Intelligence emerges through recursive, self-similar structures across multiple scales,
> where each level abstracts and compresses the one below it."**

Official implementation for the paper:
**"FIN: Learning Abstractions Through Bandwidth-Limited Fractal Hierarchies"**

---

## Overview

Current deep networks are single-scale — even 100-layer ResNets and Transformers process
everything in one representation space. FIN introduces a domain-agnostic architectural
principle where:

- A **hierarchy of levels** each operate in their own representation space
- **Bandwidth-limited channels** between levels force genuine abstraction
- **Per-level learning objectives** specialize each level's role
- **Bidirectional message passing** allows top-down constraints to refine lower levels
- A **self-similarity condition** ensures the same structural rule repeats at every scale

This is inspired by the large-scale structure of the universe and human cognition —
both of which exhibit self-similar hierarchies with compressed inter-scale communication.

---

## Repository Structure

```
fin-research/
├── README.md
├── requirements.txt
├── configs/
│   └── fin_cifar100.yaml       # All hyperparameters (maps to paper definitions)
├── fin/
│   ├── levels/
│   │   ├── level0.py           # L0: CNN encoder/decoder  (Def 1.1)
│   │   ├── level1.py           # L1: Small Transformer    (Def 1.1)
│   │   └── level2.py           # L2: MLP + top-down gate  (Def 1.1, 1.4)
│   ├── channels/
│   │   └── bottleneck.py       # Stochastic bandwidth channel (Def 1.3)
│   ├── network/
│   │   └── fin.py              # Full FIN assembly         (Def 1.2)
│   ├── losses/
│   │   └── hierarchical.py     # Hierarchical loss L_FIN   (Def 1.6)
│   └── utils/
│       └── annealing.py        # β_k annealing scheduler   (Appendix)
├── train.py                    # Main training script
├── evaluate.py                 # Evaluation + t-SNE visualization
└── experiments/
    └── ablations.py            # Ablation study runner
```

---

## Core Concepts (Paper → Code Mapping)

| Paper Definition | Code Location | Description |
|-----------------|---------------|-------------|
| Def 1.1 — Level $\mathcal{L}_k$ | `fin/levels/level{0,1,2}.py` | Encoder, decoder, objective per level |
| Def 1.2 — FIN_K | `fin/network/fin.py` | Ordered stack of levels |
| Def 1.3 — Bandwidth constraint | `fin/channels/bottleneck.py` | Hinge loss on KL divergence |
| Def 1.4 — Message passing | `fin/levels/level{1,2}.py` | Top-down $\alpha_k$ gating |
| Def 1.5 — Self-similarity | Shared `LevelBlock` template | Same structural rule at every scale |
| Def 1.6 — Hierarchical loss | `fin/losses/hierarchical.py` | $\mathcal{L}_{FIN}$ with per-level weights |
| Theorem 1.1 | Verified empirically | `evaluate.py` (CKA + t-SNE) |

---

## Quickstart

### 1. Setup Environment

**Clone and navigate to repository:**
```bash
cd fractal-intelligence-network
```

**Create a virtual environment (recommended):**
```bash
# Using Python venv
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Or using conda
conda create -n fin python=3.10
conda activate fin
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

> **System Requirements:**
> - Python 3.10+
> - CUDA 11.8+ (for GPU training; CPU-only installation available with torch CPU variant)
> - 8GB+ RAM (16GB+ recommended for training)
> - Compatible with Kaggle Notebooks, Google Colab, and HPC clusters

### 2. Prepare Data

Download CIFAR-100 (automatic on first run, ~170MB):
```bash
# Data will be automatically downloaded to ./data/ on first training run
# Or pre-download:
python -c "from torchvision import datasets; datasets.CIFAR100(root='./data', download=True)"
```

### 3. Training

**Basic training on CIFAR-100:**
```bash
python train.py --config configs/fin_cifar100.yaml
```

**Custom hyperparameters (override config file):**
```bash
python train.py --config configs/fin_cifar100.yaml \
  --learning_rate 0.001 \
  --batch_size 64 \
  --epochs 200
```

**Resume training from checkpoint:**
```bash
python train.py --config configs/fin_cifar100.yaml --resume checkpoints/latest.pt
```

Checkpoints are saved to `./checkpoints/` during training.

### 4. Evaluation & Visualization

**Evaluate and visualize learned representations:**
```bash
python evaluate.py --config configs/fin_cifar100.yaml \
  --checkpoint checkpoints/best.pt
```

This generates:
- `eval_outputs/cka_matrix.png` — CKA similarity heatmap across levels
- `eval_outputs/tsne_*.png` — t-SNE visualization of learned representations
- `eval_outputs/metrics.json` — Quantitative results (accuracy, loss, CKA, etc.)

**Generate compression visualization:**
```bash
python visualize_compression.py --checkpoint checkpoints/best.pt
```

**Generate t-SNE plots (standalone):**
```bash
python visualize_tsne.py --checkpoint checkpoints/best.pt --output eval_outputs/
```

### 5. Run Ablation Studies

**Full ablation suite (evaluates all component contributions):**
```bash
python eval_ablations.py --config configs/fin_cifar100.yaml
```

This tests:
- Full FIN model
- FIN without bandwidth constraints
- FIN without per-level objectives
- FIN without top-down feedback
- Flat ResNet baseline

Results are saved to `eval_outputs/ablations/` with comparison plots and statistics.

### 6. Tensorboard Monitoring

Monitor training in real-time:
```bash
tensorboard --logdir runs/
# Then navigate to http://localhost:6006
```

---

## Project Structure Details

```
fractal-intelligence-network/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── train.py                   # Main training script
├── evaluate.py                # Evaluation + visualization
├── eval_ablations.py          # Ablation studies
├── visualize_tsne.py          # t-SNE visualization
├── visualize_cka.py           # CKA analysis
├── visualize_compression.py   # Compression analysis
│
├── configs/
│   └── fin_cifar100.yaml      # Training hyperparameters
│
├── fin/                       # Core FIN package
│   ├── levels/
│   │   ├── level0.py          # L0: CNN encoder/decoder
│   │   ├── level1.py          # L1: Transformer
│   │   └── level2.py          # L2: MLP + gating
│   ├── channels/
│   │   └── bottleneck.py      # Bandwidth-limited channel
│   ├── network/
│   │   └── fin.py             # FIN assembly
│   ├── losses/
│   │   └── hierarchical.py    # Hierarchical loss
│   └── utils/
│       └── annealing.py       # Schedulers
│
├── data/                      # CIFAR-100 dataset (auto-downloaded)
├── checkpoints/               # Saved model weights
├── eval_outputs/              # Results, plots, metrics
├── experiments/               # Additional experiment scripts
├── figures/                   # Generated visualizations
└── notes/                     # Research notes (not tracked)
```

---

## Configuration

Edit `configs/fin_cifar100.yaml` to customize:

```yaml
# Network
num_levels: 3
hidden_dims: [512, 256, 128]  # Per-level dimensions
level_channels: [64, 32, 16]  # Bandwidth constraints

# Training
epochs: 200
batch_size: 128
learning_rate: 0.001

# Objectives
use_bandwidth_loss: true
use_per_level_loss: true
use_top_down_feedback: true
```

See `configs/fin_cifar100.yaml` for all parameters and their descriptions.

---

## Troubleshooting

**CUDA out of memory:**
```bash
python train.py --config configs/fin_cifar100.yaml --batch_size 32  # Reduce batch size
```

**Data download fails:**
```bash
# Pre-download manually
python -c "from torchvision import datasets; datasets.CIFAR100(root='./data', download=True, train=True)"
python -c "from torchvision import datasets; datasets.CIFAR100(root='./data', download=True, train=False)"
```

**Checkpoint not found:**
```bash
# List available checkpoints
ls -lh checkpoints/
# Use relative path, e.g., checkpoints/best.pt
```

---

## Performance & Reproducibility

- **Reproducibility:** Set `--seed` in config for deterministic runs
- **Performance:** GPU training (CUDA) ~10-15 minutes/epoch; CPU ~60+ minutes/epoch
- **Optimization:** Mixed precision available with `--amp` flag

---

### Main Results — CIFAR-100

| Model | Fine Acc (100 cls) | Coarse Acc (20 cls) | Params |
|-------|-------------------|---------------------|--------|
| Flat ResNet-18 baseline | — | — | — |
| FIN (full) | — | — | — |
| FIN — no bandwidth | — | — | — |
| FIN — no per-level obj | — | — | — |
| FIN — no top-down feedback | — | — | — |

*Results to be filled after experiments.*

---

## Citation

```bibtex
@article{fin2025,
  title   = {FIN: Learning Abstractions Through Bandwidth-Limited Fractal Hierarchies},
  author  = {Avishek Shrabon Kerketa},
  journal = {N/A},
  year    = {2026}
}
```

---

## License

MIT License. See `LICENSE` for details.