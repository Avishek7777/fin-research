"""
visualize_compression.py
========================
Generates representation compression figure from evaluation_results.json.
Reads precomputed representation stats directly — no checkpoint loading needed.

Two side-by-side subplots:
    Left  : Intrinsic Dimensionality across levels (z0 → z1 → z2)
    Right : Participation Ratio across levels (z0 → z1 → z2)

One line per model. FIN shows decreasing values (compression / specialization).
Baselines show flat lines (identical representations at all levels).

Output
------
    figures/representation_compression.png

Usage
-----
    python visualize_compression.py \
        --results evaluation_results.json \
        --output_dir figures
"""

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# =============================================================================
# Config
# =============================================================================

LEVEL_KEYS   = ["z0", "z1", "z2"]
LEVEL_LABELS = ["Z0\n(CNN)", "Z1\n(Transformer)", "Z2\n(MLP)"]

MODEL_DISPLAY_NAMES = {
    "mobilenet_fine":        "MobileNetV2 (Fine-tuned)",
    "mobilenet_aux":         "MobileNetV2 (+ Aux Head)",
    "fin_cifar100_cifar10":  "HSBN (Ours)",
}

# Line styles — FIN stands out, baselines are distinct but secondary
MODEL_STYLES = {
    "mobilenet_fine":       {"color": "#4363d8", "linestyle": "--", "marker": "o", "linewidth": 1.5, "markersize": 6, "zorder": 2},
    "mobilenet_aux":        {"color": "#f58231", "linestyle": ":",  "marker": "s", "linewidth": 1.5, "markersize": 6, "zorder": 2},
    "fin_cifar100_cifar10": {"color": "#e6194b", "linestyle": "-",  "marker": "D", "linewidth": 2.2, "markersize": 7, "zorder": 3},
}

PANEL_ORDER = ["mobilenet_fine", "mobilenet_aux", "fin_cifar100_cifar10"]


# =============================================================================
# Data extraction
# =============================================================================

def extract_metric(model_data: dict, metric: str) -> tuple:
    """
    Extract mean ± std for a metric across z0, z1, z2.

    Args:
        model_data : per-model dict from evaluation_results.json
        metric     : 'intrinsic_dim' or 'participation_ratio'

    Returns:
        means : list of 3 floats
        stds  : list of 3 floats (from per-seed variance)
    """
    # Compute mean and std across seeds from per_seed entries
    per_seed = model_data["per_seed"]
    values = {level: [] for level in LEVEL_KEYS}

    for seed_entry in per_seed:
        rep_stats = seed_entry["representation_stats"]
        for level in LEVEL_KEYS:
            values[level].append(rep_stats[level][metric])

    means = [np.mean(values[level]) for level in LEVEL_KEYS]
    stds  = [np.std(values[level])  for level in LEVEL_KEYS]
    return means, stds


# =============================================================================
# Plotting
# =============================================================================

def plot_metric_panel(ax, results: dict, metric: str, ylabel: str, title: str):
    """Plot one metric panel with one line per model."""
    x = np.arange(len(LEVEL_KEYS))

    for model_key in PANEL_ORDER:
        model_data = results["per_model"][model_key]
        means, stds = extract_metric(model_data, metric)
        style = MODEL_STYLES[model_key]
        label = MODEL_DISPLAY_NAMES[model_key]

        ax.plot(x, means, label=label, **style)
        ax.fill_between(
            x,
            np.array(means) - np.array(stds),
            np.array(means) + np.array(stds),
            alpha=0.12,
            color=style["color"],
            zorder=1,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(LEVEL_LABELS, fontsize=8.5)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.tick_params(axis="y", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.8)
    ax.set_xlim(-0.25, 2.25)


def make_compression_figure(results: dict, output_path: str):
    """Two-panel compression figure."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    fig.subplots_adjust(wspace=0.35)

    plot_metric_panel(
        axes[0], results,
        metric="intrinsic_dim",
        ylabel="Intrinsic Dimensionality",
        title="Intrinsic Dimensionality vs Level",
    )

    plot_metric_panel(
        axes[1], results,
        metric="participation_ratio",
        ylabel="Participation Ratio",
        title="Participation Ratio vs Level",
    )

    # Shared legend below both panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=3,
        fontsize=8.5,
        framealpha=0.0,
        handletextpad=0.4,
        columnspacing=1.2,
        bbox_to_anchor=(0.5, -0.08),
    )

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Representation compression figure from evaluation_results.json"
    )
    parser.add_argument("--results",    required=True, help="Path to evaluation_results.json")
    parser.add_argument("--output_dir", default="figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.results) as f:
        results = json.load(f)

    output_path = os.path.join(args.output_dir, "representation_compression.png")
    make_compression_figure(results, output_path)
    print("Done.")


if __name__ == "__main__":
    main()