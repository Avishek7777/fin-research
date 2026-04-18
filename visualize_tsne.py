"""
visualize_tsne.py
=================
Generates a t-SNE comparison figure between full FIN and the no-bandwidth
ablation variant, at each representation level (z0, z1, z2).

For the paper, z2 is the primary plot (top of figure). z0 and z1 are saved
as supplementary — caption can note "similar trends observed at z0 and z1."

Output
------
    figures/tsne_z2_comparison.png   ← main paper figure
    figures/tsne_all_levels.png      ← supplementary / appendix

Usage
-----
    python visualize_tsne.py \
        --config      configs/fin_cifar100.yaml \
        --ckpt_full   checkpoints/fin_cifar100_cifar10_cifar10_seed1/best_seed1.pt \
        --ckpt_nobw   checkpoints/no_bandwidth/cifar_10/seed_1/fin_no_bandwidth_s1_cifar10_cifar10_seed1/best_seed1.pt \
        --dataset     cifar10 \
        --seed        1 \
        --n_samples   2000 \
        --output_dir  figures
"""

import argparse
import os
import random

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from sklearn.manifold import TSNE
from torchvision import datasets, transforms

from fin.network.fin import build_fin


# =============================================================================
# Helpers
# =============================================================================

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

CIFAR100_FINE_CLASSES = None  # populated lazily if needed

# Palette — 10 visually distinct colours for CIFAR-10 classes
PALETTE_10 = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_model(cfg: dict, ckpt_path: str, device: torch.device) -> torch.nn.Module:
    model = build_fin(cfg)
    state = torch.load(ckpt_path, map_location=device)
    # Handle checkpoint dict format from save_checkpoint()
    if "model_state" in state:
        state = state["model_state"]
    elif "model_state_dict" in state:
        state = state["model_state_dict"]
    elif "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def get_dataloader(dataset_name: str, n_samples: int, seed: int):
    """Return a DataLoader with a fixed random subset of the test split."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    root = "./data"
    if dataset_name == "cifar10":
        ds = datasets.CIFAR10(root=root, train=False, download=True,
                              transform=transform)
        class_names = CIFAR10_CLASSES
    elif dataset_name == "cifar100":
        ds = datasets.CIFAR100(root=root, train=False, download=True,
                               transform=transform)
        class_names = [str(i) for i in range(100)]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Fixed random subset
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(ds), size=min(n_samples, len(ds)), replace=False)
    subset = torch.utils.data.Subset(ds, indices)
    loader = torch.utils.data.DataLoader(subset, batch_size=256, shuffle=False,
                                         num_workers=2, pin_memory=True)
    return loader, class_names


@torch.no_grad()
def extract_embeddings(model, loader, device):
    """
    Run encode_deterministic over the loader.
    Returns z0, z1, z2 as numpy arrays and labels as numpy array.
    """
    z0s, z1s, z2s, lbls = [], [], [], []

    for images, labels in loader:
        images = images.to(device)
        z0, z1, z2 = model.encode_deterministic(images)
        z0s.append(z0.cpu().numpy())
        z1s.append(z1.cpu().numpy())
        z2s.append(z2.cpu().numpy())
        lbls.append(labels.numpy())

    return (
        np.concatenate(z0s),
        np.concatenate(z1s),
        np.concatenate(z2s),
        np.concatenate(lbls),
    )


def run_tsne(embeddings: np.ndarray, seed: int, perplexity: int = 40) -> np.ndarray:
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=1000,
        random_state=seed,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(embeddings)


# =============================================================================
# Plotting
# =============================================================================

def plot_tsne_panel(
    ax,
    coords: np.ndarray,
    labels: np.ndarray,
    class_names: list,
    title: str,
    palette: list,
    show_legend: bool = False,
):
    """Plot a single t-SNE panel onto ax."""
    for cls_idx, (name, color) in enumerate(zip(class_names, palette)):
        mask = labels == cls_idx
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=color, label=name,
            s=6, alpha=0.65, linewidths=0,
            rasterized=True,
        )

    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)

    if show_legend:
        legend = ax.legend(
            loc="lower right",
            markerscale=2.5,
            fontsize=7,
            framealpha=0.85,
            handletextpad=0.3,
            borderpad=0.5,
            labelspacing=0.3,
        )


def make_z2_figure(
    full_z2_tsne, nobw_z2_tsne, labels, class_names, palette, output_path
):
    """
    Main paper figure — z2 side-by-side comparison.
    Left: Full FIN (with bandwidth constraint)
    Right: No bandwidth constraint
    Legend: horizontal, centered at bottom, above subtitle
    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.8))
    fig.subplots_adjust(wspace=0.05, bottom=0.22)

    plot_tsne_panel(
        axes[0], full_z2_tsne, labels, class_names,
        title="HSBN (with bandwidth constraint)",
        palette=palette,
        show_legend=False,
    )
    plot_tsne_panel(
        axes[1], nobw_z2_tsne, labels, class_names,
        title="HSBN (no bandwidth constraint)",
        palette=palette,
        show_legend=False,
    )

    # Horizontal legend centered below both panels
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                   markersize=7, label=name)
        for name, color in zip(class_names, palette)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(class_names),
        fontsize=8,
        framealpha=0.0,
        handletextpad=0.2,
        columnspacing=0.8,
        bbox_to_anchor=(0.5, 0.09),
    )

    # Subtitle below legend
    fig.text(0.5, 0.01, "Level Z2 representations (t-SNE, seed 1)",
             ha="center", fontsize=8, color="#777777")

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def make_all_levels_figure(
    full_embeds, nobw_embeds, labels, class_names, palette,
    seed, output_path
):
    """
    Supplementary figure — all three levels side by side.
    Rows: z0, z1, z2
    Cols: Full FIN | No bandwidth
    """
    level_names = ["Z0 (CNN, d=512)", "Z1 (Transformer, d=128)", "Z2 (MLP apex, d=32)"]
    full_arrays  = [full_embeds[0],  full_embeds[1],  full_embeds[2]]
    nobw_arrays  = [nobw_embeds[0],  nobw_embeds[1],  nobw_embeds[2]]

    fig, axes = plt.subplots(3, 2, figsize=(9, 12))
    fig.subplots_adjust(hspace=0.18, wspace=0.05)

    for row, (lname, farr, narr) in enumerate(zip(level_names, full_arrays, nobw_arrays)):
        full_tsne = run_tsne(farr, seed=seed)
        nobw_tsne = run_tsne(narr, seed=seed)

        show_leg = (row == 0)
        plot_tsne_panel(axes[row, 0], full_tsne, labels, class_names,
                        title=f"Full HSBN — {lname}", palette=palette,
                        show_legend=show_leg)
        plot_tsne_panel(axes[row, 1], nobw_tsne, labels, class_names,
                        title=f"No bandwidth — {lname}", palette=palette,
                        show_legend=False)

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="t-SNE: FIN vs no-bandwidth")
    parser.add_argument("--config",      required=True,  help="Path to fin YAML config")
    parser.add_argument("--ckpt_full",   required=True,  help="Checkpoint: full FIN")
    parser.add_argument("--ckpt_nobw",   required=True,  help="Checkpoint: no-bandwidth ablation")
    parser.add_argument("--dataset",     default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--seed",        type=int, default=1)
    parser.add_argument("--n_samples",   type=int, default=2000,
                        help="Number of test samples to embed (default 2000)")
    parser.add_argument("--perplexity",  type=int, default=40)
    parser.add_argument("--output_dir",  default="figures")
    parser.add_argument("--all_levels",  action="store_true",
                        help="Also generate the 3-level supplementary figure")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Load both models
    print("Loading full FIN checkpoint...")
    model_full = load_model(cfg, args.ckpt_full, device)

    print("Loading no-bandwidth checkpoint...")
    model_nobw = load_model(cfg, args.ckpt_nobw, device)

    # Data
    print(f"Loading {args.dataset} test set ({args.n_samples} samples)...")
    loader, class_names = get_dataloader(args.dataset, args.n_samples, args.seed)
    palette = PALETTE_10[:len(class_names)]

    # Extract embeddings
    print("Extracting embeddings — full FIN...")
    full_z0, full_z1, full_z2, labels = extract_embeddings(model_full, loader, device)

    print("Extracting embeddings — no-bandwidth...")
    nobw_z0, nobw_z1, nobw_z2, _      = extract_embeddings(model_nobw, loader, device)

    # t-SNE for z2 (main figure)
    print("Running t-SNE on z2 (full FIN)...")
    full_z2_tsne = run_tsne(full_z2, seed=args.seed, perplexity=args.perplexity)

    print("Running t-SNE on z2 (no bandwidth)...")
    nobw_z2_tsne = run_tsne(nobw_z2, seed=args.seed, perplexity=args.perplexity)

    # Save main figure
    z2_path = os.path.join(args.output_dir, "tsne_z2_comparison.png")
    make_z2_figure(full_z2_tsne, nobw_z2_tsne, labels, class_names, palette, z2_path)

    # Optional: all levels supplementary figure
    if args.all_levels:
        print("Generating all-levels supplementary figure (runs t-SNE 6 times)...")
        all_path = os.path.join(args.output_dir, "tsne_all_levels.png")
        make_all_levels_figure(
            full_embeds=(full_z0, full_z1, full_z2),
            nobw_embeds=(nobw_z0, nobw_z1, nobw_z2),
            labels=labels,
            class_names=class_names,
            palette=palette,
            seed=args.seed,
            output_path=all_path,
        )

    print("Done.")


if __name__ == "__main__":
    main()