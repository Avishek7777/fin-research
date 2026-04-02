"""
visualize_cka.py
================
Generates CKA heatmap comparison figure from evaluation_results.json.
Reads precomputed CKA values directly — no checkpoint loading required.

Shows three panels side by side:
    MobileNetV2-Fine | MobileNetV2-Aux | FIN

The all-1.0 matrices for the baselines contrast sharply with FIN's
structured off-diagonal values, visually proving the core thesis.

Output
------
    figures/cka_heatmaps.png

Usage
-----
    python visualize_cka.py \
        --results evaluation_results.json \
        --output_dir figures
"""

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable


# =============================================================================
# Helpers
# =============================================================================

LEVEL_LABELS = ["Z0\n(CNN)", "Z1\n(Trans.)", "Z2\n(MLP)"]

MODEL_DISPLAY_NAMES = {
    "mobilenet_fine": "MobileNetV2\n(Fine-tuned)",
    "mobilenet_aux":  "MobileNetV2\n(+ Aux Head)",
    "fin_cifar100_cifar10": "FIN\n(Ours)",
}

# Order for left-to-right panel layout
PANEL_ORDER = ["mobilenet_fine", "mobilenet_aux", "fin_cifar100_cifar10"]


def extract_cka_matrix(model_data: dict) -> np.ndarray:
    """
    Build a 3x3 symmetric CKA matrix from summary_statistics.
    Uses mean CKA values across seeds.

    summary_statistics.cka keys: '01', '12', '02'
    Matrix layout:
        [[1.0,   cka_01, cka_02],
         [cka_01, 1.0,  cka_12],
         [cka_02, cka_12, 1.0 ]]
    """
    cka = model_data["summary_statistics"]["cka"]
    c01 = cka["01"]["mean"]
    c12 = cka["12"]["mean"]
    c02 = cka["02"]["mean"]

    mat = np.array([
        [1.0,  c01,  c02],
        [c01,  1.0,  c12],
        [c02,  c12,  1.0],
    ])
    return mat


# =============================================================================
# Plotting
# =============================================================================

def plot_cka_panel(ax, matrix: np.ndarray, title: str, is_fin: bool):
    """
    Plot a single CKA heatmap panel.

    Args:
        ax      : matplotlib axis
        matrix  : 3x3 CKA matrix
        title   : panel title
        is_fin  : if True, use a more expressive colormap range to highlight
                  the structured values; baselines are all 1.0 so full range
                  is less informative
    """
    # Use shared vmin/vmax so all panels are on the same color scale
    vmin, vmax = 0.0, 1.0

    im = ax.imshow(matrix, cmap="RdYlGn", vmin=vmin, vmax=vmax, aspect="equal")

    # Gridlines
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Axis labels
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(LEVEL_LABELS, fontsize=8)
    ax.set_yticklabels(LEVEL_LABELS, fontsize=8)
    ax.tick_params(axis="both", which="major", length=0)

    # Annotate each cell with its value
    for i in range(3):
        for j in range(3):
            val = matrix[i, j]
            # Use dark text on light cells, light on dark
            text_color = "black" if val > 0.4 else "white"
            ax.text(j, i, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color=text_color)

    # Title
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)

    return im


def make_cka_figure(results: dict, output_path: str):
    """
    Three-panel CKA heatmap figure.
    Shared colorbar on the right.
    """
    per_model = results["per_model"]

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    fig.subplots_adjust(wspace=0.35)

    last_im = None
    for ax, model_key in zip(axes, PANEL_ORDER):
        model_data = per_model[model_key]
        matrix     = extract_cka_matrix(model_data)
        title      = MODEL_DISPLAY_NAMES[model_key]
        is_fin     = model_key.startswith("fin")

        last_im = plot_cka_panel(ax, matrix, title, is_fin)

    # Shared colorbar — attach to rightmost axis
    divider = make_axes_locatable(axes[-1])
    cax = divider.append_axes("right", size="6%", pad=0.08)
    cbar = fig.colorbar(last_im, cax=cax)
    cbar.set_label("CKA Similarity", fontsize=8, labelpad=6)
    cbar.ax.tick_params(labelsize=7)

    # Figure caption line
    fig.text(
        0.5, -0.04,
        "CKA similarity between level representations (mean over 3 seeds). "
        "Baselines show identical representations across all levels (CKA=1.0). "
        "FIN develops genuinely differentiated hierarchical structure.",
        ha="center", fontsize=7.5, color="#555555",
        wrap=True
    )

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="CKA heatmap figure from evaluation_results.json")
    parser.add_argument("--results",     required=True, help="Path to evaluation_results.json")
    parser.add_argument("--output_dir",  default="figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.results) as f:
        results = json.load(f)

    output_path = os.path.join(args.output_dir, "cka_heatmaps.png")
    make_cka_figure(results, output_path)
    print("Done.")


if __name__ == "__main__":
    main()