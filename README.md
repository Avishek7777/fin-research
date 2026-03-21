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

### Install
```bash
git clone https://github.com/YOUR_USERNAME/fin-research.git
cd fin-research
pip install -r requirements.txt
```

### Train (CIFAR-100)
```bash
python train.py --config configs/fin_cifar100.yaml
```

### Evaluate + Visualize Representations
```bash
python evaluate.py --config configs/fin_cifar100.yaml --checkpoint checkpoints/best.pt
```

### Run Ablations
```bash
python experiments/ablations.py --config configs/fin_cifar100.yaml
```

---

## Experiments

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
  author  = {YOUR NAME},
  journal = {arXiv preprint},
  year    = {2025}
}
```

---

## License

MIT License. See `LICENSE` for details.