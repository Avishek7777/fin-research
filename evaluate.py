"""
evaluate.py

FIN Evaluation, Representation Analysis & Visualization
========================================================

Usage:
    # Full evaluation on test set
    python evaluate.py --config configs/fin_cifar100.yaml \
                       --checkpoint checkpoints/best.pt

    # With t-SNE visualization
    python evaluate.py --config configs/fin_cifar100.yaml \
                       --checkpoint checkpoints/best.pt \
                       --tsne

    # Compare FIN vs ablation checkpoint
    python evaluate.py --config configs/fin_cifar100.yaml \
                       --checkpoint checkpoints/best.pt \
                       --compare checkpoints/no_bandwidth/best.pt \
                       --tsne

What this script produces:
    1. Accuracy table (fine + coarse) for paper Section 5.1
    2. t-SNE plots of z0, z1, z2 — visual evidence for Theorem 1.1
    3. CKA similarity matrix between levels — quantitative evidence
       that each level learns a genuinely different representation
    4. Bandwidth usage report — how much of beta_k each channel used
    5. Alpha gate values — how much top-down influence each level learned

Theorem 1.1 Empirical Verification
------------------------------------
Theorem 1.1 states:
    H(Z_k) < H(Z_{k-1})          [entropy decreasing up hierarchy]
    I(Z_k; Y) >= I(Z_{k-1}; Y) - eps  [task info preserved]

We verify this empirically via:
  - t-SNE: more clustered at higher levels = lower effective entropy
  - CKA: low similarity between levels = genuinely different representations
  - Fine + coarse accuracy at L1 and L2 = task information preserved
  - Representation variance: var(z2) < var(z1) < var(z0)
"""

import os
import sys
import yaml
import argparse
import numpy as np
from typing import Optional, List, Tuple, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for Kaggle/server
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from fin.network.fin import build_fin, FIN
from train import CIFAR100_COARSE_LABELS, COARSE_LABEL_TENSOR, get_coarse_labels


# =============================================================================
# CIFAR-100 Class Names (for plot labels)
# =============================================================================

CIFAR100_SUPERCLASSES = [
    "aquatic mammals", "fish", "flowers", "food containers",
    "fruit & vegetables", "household electrical devices", "household furniture",
    "insects", "large carnivores", "large man-made outdoor things",
    "large natural outdoor scenes", "large omnivores & herbivores",
    "medium-sized mammals", "non-insect invertebrates", "people",
    "reptiles", "small mammals", "trees", "vehicles 1", "vehicles 2",
]


# =============================================================================
# Load Model
# =============================================================================

def load_model(cfg: dict, checkpoint_path: str, device: torch.device) -> FIN:
    """Load FIN from checkpoint."""
    model = build_fin(cfg).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[Load] Loaded checkpoint: {checkpoint_path}")
    if "metrics" in ckpt:
        print(f"[Load] Checkpoint metrics: {ckpt['metrics']}")
    return model


# =============================================================================
# Extract Representations
# =============================================================================

@torch.no_grad()
def extract_representations(
    model      : FIN,
    loader     : DataLoader,
    device     : torch.device,
    max_samples: int = 5000,
) -> Dict:
    """
    Extract z0, z1, z2 representations and labels from the dataset.
    Uses deterministic forward (channel means, no sampling) for
    stable, reproducible embeddings suitable for analysis.

    Args:
        model      : FIN in eval mode
        loader     : DataLoader (shuffled=False for consistency)
        device     : compute device
        max_samples: cap at this many samples (t-SNE is O(n^2))

    Returns:
        dict with keys: z0, z1, z2, fine_labels, coarse_labels
        all as numpy arrays
    """
    z0_list, z1_list, z2_list = [], [], []
    fine_list, coarse_list    = [], []
    total = 0

    for x, fine_labels in tqdm(loader, desc="Extracting representations"):
        if total >= max_samples:
            break

        x             = x.to(device)
        fine_labels   = fine_labels.to(device)
        coarse_labels = get_coarse_labels(fine_labels).to(device)

        z0, z1, z2 = model.encode_deterministic(x)

        z0_list.append(z0.cpu().numpy())
        z1_list.append(z1.cpu().numpy())
        z2_list.append(z2.cpu().numpy())
        fine_list.append(fine_labels.cpu().numpy())
        coarse_list.append(coarse_labels.cpu().numpy())

        total += x.size(0)

    return {
        "z0"           : np.concatenate(z0_list,    axis=0)[:max_samples],
        "z1"           : np.concatenate(z1_list,    axis=0)[:max_samples],
        "z2"           : np.concatenate(z2_list,    axis=0)[:max_samples],
        "fine_labels"  : np.concatenate(fine_list,  axis=0)[:max_samples],
        "coarse_labels": np.concatenate(coarse_list, axis=0)[:max_samples],
    }


# =============================================================================
# Accuracy Evaluation
# =============================================================================

@torch.no_grad()
def evaluate_accuracy(
    model : FIN,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Compute fine and coarse top-1 accuracy on the full test set.

    Returns:
        dict: fine_acc, coarse_acc, joint_acc (all in %)
    """
    model.eval()
    fine_correct   = 0
    coarse_correct = 0
    total          = 0

    for x, fine_labels in tqdm(loader, desc="Evaluating accuracy"):
        x             = x.to(device)
        fine_labels   = fine_labels.to(device)
        coarse_labels = get_coarse_labels(fine_labels).to(device)

        # Use stochastic forward for accuracy (matches training conditions)
        out = model(x, fine_labels, coarse_labels)

        fine_correct   += (out.fine_logits.argmax(1) == fine_labels).sum().item()
        coarse_correct += (out.coarse_logits.argmax(1) == coarse_labels).sum().item()
        total          += x.size(0)

    fine_acc   = 100.0 * fine_correct   / total
    coarse_acc = 100.0 * coarse_correct / total
    joint      = fine_acc + coarse_acc

    return {
        "fine_acc"  : fine_acc,
        "coarse_acc": coarse_acc,
        "joint_acc" : joint,
    }


# =============================================================================
# t-SNE Visualization
# =============================================================================

def compute_tsne(
    representations: Dict,
    perplexity     : int = 30,
    n_iter         : int = 1000,
    random_state   : int = 42,
) -> Dict:
    """
    Compute t-SNE embeddings for z0, z1, z2.

    Each representation is standardized before t-SNE to ensure
    fair comparison across levels (different scales otherwise).

    Returns:
        dict: tsne_z0, tsne_z1, tsne_z2 — each (N, 2) numpy arrays
    """
    results = {}
    scaler  = StandardScaler()

    for key in ["z0", "z1", "z2"]:
        z = representations[key]
        print(f"  t-SNE on {key} (shape={z.shape})...")

        z_scaled = scaler.fit_transform(z)
        tsne     = TSNE(
            n_components = 2,
            perplexity   = perplexity,
            n_iter       = n_iter,
            random_state = random_state,
            learning_rate= "auto",
            init         = "pca",
        )
        results[f"tsne_{key}"] = tsne.fit_transform(z_scaled)

    return results


def plot_tsne(
    representations: Dict,
    tsne_results   : Dict,
    save_path      : str,
    label_type     : str = "coarse",   # "coarse" or "fine"
):
    """
    Plot t-SNE embeddings for all three levels side by side.
    This is Figure X in the paper — direct visual evidence for Theorem 1.1.

    Color = class label. Tighter clusters at higher levels = lower entropy
    = more abstract, more organized representation space.

    Args:
        label_type: "coarse" (20 colors) or "fine" (100 colors, noisier)
    """
    if label_type == "coarse":
        labels     = representations["coarse_labels"]
        n_classes  = 20
        class_names= CIFAR100_SUPERCLASSES
        title_suffix = "(20 superclasses)"
    else:
        labels     = representations["fine_labels"]
        n_classes  = 100
        class_names= [str(i) for i in range(100)]
        title_suffix = "(100 fine classes)"

    # Color palette — use tab20 for coarse (20 colors), viridis for fine
    if n_classes <= 20:
        cmap   = plt.cm.get_cmap("tab20", n_classes)
        colors = [cmap(i) for i in range(n_classes)]
    else:
        cmap   = plt.cm.get_cmap("nipy_spectral", n_classes)
        colors = [cmap(i) for i in range(n_classes)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"FIN Representation t-SNE — {title_suffix}\n"
        f"Theorem 1.1: each level should show tighter clustering than the one below",
        fontsize=13, y=1.02
    )

    level_info = [
        ("tsne_z0", "L0 — Raw Signal\n(CNN, d=512)",      "z0"),
        ("tsne_z1", "L1 — Local Patterns\n(Transformer, d=128)", "z1"),
        ("tsne_z2", "L2 — Abstract Concepts\n(MLP Apex, d=32)",  "z2"),
    ]

    for ax, (tsne_key, title, z_key) in zip(axes, level_info):
        tsne_emb = tsne_results[tsne_key]

        # Plot each class
        for cls_idx in range(n_classes):
            mask = labels == cls_idx
            if mask.sum() == 0:
                continue
            ax.scatter(
                tsne_emb[mask, 0],
                tsne_emb[mask, 1],
                c=[colors[cls_idx]],
                s=4,
                alpha=0.6,
                label=class_names[cls_idx] if n_classes <= 20 else None,
            )

        # Compute and display cluster tightness (mean intra-class variance)
        intra_vars = []
        for cls_idx in range(n_classes):
            mask = labels == cls_idx
            if mask.sum() < 2:
                continue
            pts = tsne_emb[mask]
            intra_vars.append(pts.var(axis=0).mean())
        tightness = np.mean(intra_vars)

        ax.set_title(f"{title}\nCluster tightness: {tightness:.1f} (lower = tighter)",
                     fontsize=10)
        ax.set_xlabel("t-SNE dim 1")
        ax.set_ylabel("t-SNE dim 2")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[["top","right"]].set_visible(False)

    # Legend (only for coarse — 20 classes is readable)
    if n_classes <= 20:
        handles = [
            plt.scatter([], [], c=[colors[i]], s=30, label=class_names[i])
            for i in range(n_classes)
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=5,
            fontsize=8,
            bbox_to_anchor=(0.5, -0.15),
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] t-SNE saved: {save_path}")


# =============================================================================
# CKA (Centered Kernel Alignment)
# =============================================================================

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute Linear CKA similarity between two representation matrices.

    CKA(X, Y) in [0, 1]:
        0.0 = completely different representations
        1.0 = identical representations (up to linear transformation)

    For FIN, we expect:
        CKA(z0, z1) < 1.0  — L1 is genuinely different from L0
        CKA(z1, z2) < 1.0  — L2 is genuinely different from L1
        CKA(z0, z2) << 1.0 — L2 is very different from L0

    Low CKA across levels = self-similarity condition holds
    (same structural rule, different learned content).

    Reference: Kornblith et al. (2019) "Similarity of Neural Network
    Representations Revisited"
    """
    def center(K):
        n    = K.shape[0]
        unit = np.ones([n, n])
        I    = np.eye(n)
        H    = I - unit / n
        return H @ K @ H

    K_X = X @ X.T
    K_Y = Y @ Y.T

    K_Xc = center(K_X)
    K_Yc = center(K_Y)

    hsic_xy = np.sum(K_Xc * K_Yc)
    hsic_xx = np.sum(K_Xc * K_Xc)
    hsic_yy = np.sum(K_Yc * K_Yc)

    if hsic_xx == 0 or hsic_yy == 0:
        return 0.0

    return float(hsic_xy / np.sqrt(hsic_xx * hsic_yy))


def compute_cka_matrix(representations: Dict, subsample: int = 500) -> np.ndarray:
    """
    Compute 3x3 CKA similarity matrix between z0, z1, z2.

    Subsamples to `subsample` examples for computational tractability
    (CKA requires O(n^2) kernel computation).

    Returns:
        (3, 3) numpy array of CKA values
    """
    n  = min(subsample, len(representations["z0"]))
    idx = np.random.choice(len(representations["z0"]), n, replace=False)

    levels = ["z0", "z1", "z2"]
    matrix = np.zeros((3, 3))

    scaler = StandardScaler()

    zs = {
        k: scaler.fit_transform(representations[k][idx])
        for k in levels
    }

    for i, ki in enumerate(levels):
        for j, kj in enumerate(levels):
            matrix[i, j] = linear_cka(zs[ki], zs[kj])

    return matrix


def plot_cka_matrix(cka_matrix: np.ndarray, save_path: str):
    """
    Plot CKA similarity matrix as a heatmap.
    Low off-diagonal values confirm each level learned genuinely
    different representations — quantitative evidence for Theorem 1.1.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    level_labels = ["L0\n(d=512)", "L1\n(d=128)", "L2\n(d=32)"]

    sns.heatmap(
        cka_matrix,
        annot     = True,
        fmt       = ".3f",
        cmap      = "Blues",
        vmin      = 0.0,
        vmax      = 1.0,
        xticklabels = level_labels,
        yticklabels = level_labels,
        ax        = ax,
        linewidths= 0.5,
    )

    ax.set_title(
        "CKA Representation Similarity Matrix\n"
        "Low off-diagonal = genuinely different representations per level\n"
        "(Empirical evidence for Theorem 1.1)",
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] CKA matrix saved: {save_path}")


# =============================================================================
# Representation Statistics (Theorem 1.1 evidence)
# =============================================================================

def compute_representation_stats(representations: Dict) -> Dict:
    """
    Compute statistics that directly test Theorem 1.1's predictions:
        H(Z_k) decreasing  -> proxy: representation variance decreasing
        I(Z_k; Y) preserved -> proxy: per-level classification accuracy

    Returns dict of stats for each level.
    """
    stats = {}
    for key in ["z0", "z1", "z2"]:
        z = representations[key]
        stats[key] = {
            "dim"       : z.shape[1],
            "var_mean"  : float(z.var(axis=0).mean()),    # mean feature variance
            "var_total" : float(z.var()),                  # total variance
            "norm_mean" : float(np.linalg.norm(z, axis=1).mean()),
            "norm_std"  : float(np.linalg.norm(z, axis=1).std()),
        }
    return stats


def print_representation_report(
    stats       : Dict,
    accuracies  : Dict,
    cka_matrix  : np.ndarray,
    model       : FIN,
):
    """
    Print a full evaluation report suitable for the paper appendix.
    """
    print("\n" + "=" * 65)
    print("FIN EVALUATION REPORT")
    print("=" * 65)

    print("\n── Accuracy (Section 5.1) ──────────────────────────────────")
    print(f"  Fine   accuracy (L1, 100 classes) : {accuracies['fine_acc']:.2f}%")
    print(f"  Coarse accuracy (L2,  20 classes) : {accuracies['coarse_acc']:.2f}%")
    print(f"  Joint  accuracy (fine + coarse)   : {accuracies['joint_acc']:.2f}")

    print("\n── Representation Statistics (Theorem 1.1 evidence) ────────")
    print(f"  {'Level':<8} {'Dim':<8} {'Var(mean)':<14} {'||z|| mean':<14}")
    print(f"  {'-'*50}")
    for key, label in [("z0","L0"), ("z1","L1"), ("z2","L2")]:
        s = stats[key]
        print(f"  {label:<8} {s['dim']:<8} {s['var_mean']:<14.4f} {s['norm_mean']:<14.4f}")
    print()

    # Verify Theorem 1.1 prediction: variance should decrease up hierarchy
    var_z0 = stats["z0"]["var_mean"]
    var_z1 = stats["z1"]["var_mean"]
    var_z2 = stats["z2"]["var_mean"]

    if var_z0 > var_z1 > var_z2:
        print("  [PASS] Var(z0) > Var(z1) > Var(z2) — consistent with Theorem 1.1")
    else:
        print("  [NOTE] Variance order not strictly decreasing — check bandwidth params")
        print(f"         var_z0={var_z0:.4f}, var_z1={var_z1:.4f}, var_z2={var_z2:.4f}")

    print("\n── CKA Similarity Matrix ───────────────────────────────────")
    print("  (0.0=completely different, 1.0=identical)")
    labels = ["L0", "L1", "L2"]
    header = f"  {'':>4}" + "".join(f"{l:>8}" for l in labels)
    print(header)
    for i, row_label in enumerate(labels):
        row = f"  {row_label:>4}" + "".join(f"{cka_matrix[i,j]:>8.3f}" for j in range(3))
        print(row)

    off_diag = [cka_matrix[i,j] for i in range(3) for j in range(3) if i != j]
    print(f"\n  Mean off-diagonal CKA: {np.mean(off_diag):.3f}")
    if np.mean(off_diag) < 0.7:
        print("  [PASS] Low inter-level similarity — genuinely different representations")
    else:
        print("  [NOTE] High inter-level similarity — levels may not be specializing")

    print("\n── Bandwidth Usage ─────────────────────────────────────────")
    print(f"  Channel 01: beta_max={model._current_betas['channel_01']:.3f}")
    print(f"  Channel 12: beta_max={model._current_betas['channel_12']:.3f}")

    print("\n── Alpha Gate Values (top-down influence) ──────────────────")
    if hasattr(model.level0, "alpha"):
        a0 = model.level0.alpha.item()
        print(f"  alpha_0 (L1->L0 influence): {a0:.4f}")
    if hasattr(model.level1, "alpha"):
        a1 = model.level1.alpha.item()
        print(f"  alpha_1 (L2->L1 influence): {a1:.4f}")

    print("\n── Parameter Count ─────────────────────────────────────────")
    pc = model.param_count()
    for k, v in pc.items():
        if k != "total_M":
            print(f"  {k:<16}: {v:>10,}")
    print(f"  {'Total (M)':<16}: {pc['total_M']:>10.2f}M")
    print("=" * 65)


# =============================================================================
# Results Table (for paper Section 5.1)
# =============================================================================

def print_results_table(results: List[Dict]):
    """
    Print a formatted results table for the paper.
    Call this with results from multiple model variants
    (FIN full, ablations, baselines).

    Args:
        results: list of dicts with keys:
                 name, fine_acc, coarse_acc, joint_acc, params_M
    """
    print("\n── Results Table (Section 5.1) ─────────────────────────────")
    print(f"  {'Model':<35} {'Fine%':>7} {'Coarse%':>9} {'Joint':>8} {'Params':>8}")
    print(f"  {'-'*72}")
    for r in results:
        print(
            f"  {r['name']:<35} "
            f"{r['fine_acc']:>7.2f} "
            f"{r['coarse_acc']:>9.2f} "
            f"{r['joint_acc']:>8.2f} "
            f"{r.get('params_M', '—'):>8}"
        )
    print()


# =============================================================================
# Main Evaluation Function
# =============================================================================

def evaluate(
    cfg           : dict,
    checkpoint_path: str,
    run_tsne      : bool = False,
    compare_path  : Optional[str] = None,
    output_dir    : str = "eval_outputs",
    max_samples   : int = 5000,
):
    """
    Full evaluation pipeline.

    Args:
        cfg            : config dict
        checkpoint_path: path to FIN checkpoint
        run_tsne       : whether to compute and save t-SNE plots
        compare_path   : optional second checkpoint to compare against
        output_dir     : directory to save plots and reports
        max_samples    : max samples for t-SNE / CKA
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    val_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5071, 0.4867, 0.4408],
                    std =[0.2675, 0.2565, 0.2761]),
    ])
    val_dataset = torchvision.datasets.CIFAR100(
        root=cfg["data"]["root"], train=False,
        download=True, transform=val_transform,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = cfg["data"]["batch_size"] * 2,
        shuffle     = False,
        num_workers = cfg["data"]["num_workers"],
    )

    # ── Load model ───────────────────────────────────────────────────────────
    model = load_model(cfg, checkpoint_path, device)

    # ── Accuracy ─────────────────────────────────────────────────────────────
    print("\n[Eval] Computing accuracy...")
    accuracies = evaluate_accuracy(model, val_loader, device)

    # ── Representations ──────────────────────────────────────────────────────
    print("\n[Eval] Extracting representations...")
    representations = extract_representations(model, val_loader, device, max_samples)

    # ── CKA ──────────────────────────────────────────────────────────────────
    print("\n[Eval] Computing CKA similarity matrix...")
    cka_matrix = compute_cka_matrix(representations)
    plot_cka_matrix(cka_matrix, os.path.join(output_dir, "cka_matrix.png"))

    # ── Representation statistics ─────────────────────────────────────────────
    stats = compute_representation_stats(representations)

    # ── Full report ──────────────────────────────────────────────────────────
    print_representation_report(stats, accuracies, cka_matrix, model)

    # ── t-SNE ────────────────────────────────────────────────────────────────
    if run_tsne:
        print("\n[Eval] Computing t-SNE (this takes a few minutes)...")
        tsne_results = compute_tsne(representations)

        plot_tsne(
            representations, tsne_results,
            save_path  = os.path.join(output_dir, "tsne_coarse.png"),
            label_type = "coarse",
        )
        plot_tsne(
            representations, tsne_results,
            save_path  = os.path.join(output_dir, "tsne_fine.png"),
            label_type = "fine",
        )

    # ── Comparison model (ablation) ───────────────────────────────────────────
    if compare_path:
        print(f"\n[Eval] Evaluating comparison model: {compare_path}")
        model_cmp  = load_model(cfg, compare_path, device)
        acc_cmp    = evaluate_accuracy(model_cmp, val_loader, device)
        rep_cmp    = extract_representations(model_cmp, val_loader, device, max_samples)
        cka_cmp    = compute_cka_matrix(rep_cmp)
        stats_cmp  = compute_representation_stats(rep_cmp)

        print_representation_report(stats_cmp, acc_cmp, cka_cmp, model_cmp)

        # Print side-by-side results table
        pc_main = model.param_count()
        pc_cmp  = model_cmp.param_count()
        print_results_table([
            {
                "name"      : f"FIN (full) — {checkpoint_path}",
                "fine_acc"  : accuracies["fine_acc"],
                "coarse_acc": accuracies["coarse_acc"],
                "joint_acc" : accuracies["joint_acc"],
                "params_M"  : f"{pc_main['total_M']}M",
            },
            {
                "name"      : f"Comparison — {compare_path}",
                "fine_acc"  : acc_cmp["fine_acc"],
                "coarse_acc": acc_cmp["coarse_acc"],
                "joint_acc" : acc_cmp["joint_acc"],
                "params_M"  : f"{pc_cmp['total_M']}M",
            },
        ])

        if run_tsne:
            rep_cmp_tsne = extract_representations(
                model_cmp, val_loader, device, max_samples
            )
            tsne_cmp = compute_tsne(rep_cmp_tsne)
            plot_tsne(
                rep_cmp_tsne, tsne_cmp,
                save_path  = os.path.join(output_dir, "tsne_comparison_coarse.png"),
                label_type = "coarse",
            )

    print(f"\n[Done] All outputs saved to: {output_dir}/")


# =============================================================================
# Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate Fractal Intelligence Network")

    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (best.pt)"
    )
    parser.add_argument(
        "--tsne", action="store_true",
        help="Compute and save t-SNE visualizations (takes ~5 min)"
    )
    parser.add_argument(
        "--compare", type=str, default=None,
        help="Optional: path to a second checkpoint to compare against"
    )
    parser.add_argument(
        "--output_dir", type=str, default="eval_outputs",
        help="Directory to save plots and reports"
    )
    parser.add_argument(
        "--max_samples", type=int, default=5000,
        help="Max samples for t-SNE and CKA (default: 5000)"
    )

    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    evaluate(
        cfg             = cfg,
        checkpoint_path = args.checkpoint,
        run_tsne        = args.tsne,
        compare_path    = args.compare,
        output_dir      = args.output_dir,
        max_samples     = args.max_samples,
    )


if __name__ == "__main__":
    main()