"""
eval_ablations.py

Evaluate Already-Trained Ablation Checkpoints (Section 5.2)
=============================================================

Use this when ablation training has already finished (via ablations.py)
but evaluation failed or was skipped. Loads checkpoints from disk,
runs evaluation, and produces the full ablation results table + plots.

Checkpoint layout expected (matches what ablations.py + train_single save):
    checkpoints/{ablation}/{dataset}/seed_{seed}/
        fin_{ablation}_s{seed}_{dataset}_{dataset}_seed{seed}/
            best_seed{seed}.pt

Full FIN baseline checkpoints (from train.py):
    checkpoints/{exp_name}_{dataset}_seed{seed}/best_seed{seed}.pt
  or
    checkpoints/{exp_name}_{dataset}_{dataset}_seed{seed}/best_seed{seed}.pt

Usage:
    # Evaluate all cifar100 ablation checkpoints (3 seeds)
    python eval_ablations.py --config configs/fin_cifar100.yaml \\
                             --dataset cifar100 --seeds "1,2,3"

    # Evaluate cifar10 ablation checkpoints
    python eval_ablations.py --config configs/fin_cifar100.yaml \\
                             --dataset cifar10 --seeds "1,2,3"

    # Evaluate a single ablation only
    python eval_ablations.py --config configs/fin_cifar100.yaml \\
                             --dataset cifar100 --seeds "1,2,3" \\
                             --ablation no_bandwidth

    # Override checkpoint root if your paths differ
    python eval_ablations.py --config configs/fin_cifar100.yaml \\
                             --dataset cifar100 --seeds "1,2,3" \\
                             --checkpoint_root /path/to/checkpoints
"""

import os
import sys
import yaml
import argparse
import json
import copy
import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import parse_seeds


# =============================================================================
# Ablation Definitions (mirrors ablations.py)
# =============================================================================

ABLATIONS = [
    {"name": "no_bandwidth",           "component": "bandwidth_constraint"},
    {"name": "no_per_level_objectives","component": "per_level_objectives"},
    {"name": "no_top_down_feedback",   "component": "top_down_feedback"},
    {"name": "no_hierarchy_flat",      "component": "hierarchical_structure"},
    {"name": "fixed_bandwidth",        "component": "beta_annealing"},
    {"name": "no_self_similarity",     "component": "self_similarity"},
]

CIFAR10_CLASSES    = 10
CIFAR100_CLASSES   = 100
COARSE_CLASSES_C100 = 20


# =============================================================================
# Checkpoint Path Resolution
# =============================================================================

def find_ablation_checkpoint(
    checkpoint_root: str,
    ablation: str,
    dataset: str,
    seed: int,
) -> Optional[str]:
    """
    Find the best checkpoint for a given ablation/dataset/seed.

    Tries all known naming patterns that ablations.py + train_single produce.
    Returns the first path that exists, or None.
    """
    # exp_name matches what ablations.py sets:
    #   exp_cfg["experiment"]["name"] = f"fin_{ablation}_s{seed}_{dataset}"
    exp_name = f"fin_{ablation}_s{seed}_{dataset}"

    # base_ckpt_dir matches what ablations.py sets:
    #   exp_cfg["experiment"]["checkpoint_dir"] = f"checkpoints/{ablation}/{dataset}/seed_{seed}"
    base_dir = os.path.join(checkpoint_root, ablation, dataset, f"seed_{seed}")

    candidates = [
        # train_single appends "{exp_name}_{dataset}_seed{seed}" as a subdirectory
        os.path.join(base_dir, f"{exp_name}_{dataset}_seed{seed}", f"best_seed{seed}.pt"),
        # Fallback: flat inside base_dir
        os.path.join(base_dir, f"best_seed{seed}.pt"),
        os.path.join(base_dir, "best.pt"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    return None


def find_baseline_checkpoint(
    checkpoint_root: str,
    exp_name: str,
    dataset: str,
    seed: int,
) -> Optional[str]:
    """
    Find the full FIN baseline checkpoint saved by train.py.

    train.py's main() appends _{dataset} to exp_name, then train_single
    appends _{dataset}_seed{seed} as a subdirectory, producing the
    triple-repeat pattern: {exp_name}_{dataset}_{dataset}_seed{seed}.
    """
    candidates = [
        # Triple repeat (train.py main() + train_single both append dataset)
        os.path.join(checkpoint_root, f"{exp_name}_{dataset}_{dataset}_seed{seed}", f"best_seed{seed}.pt"),
        # Double repeat fallback
        os.path.join(checkpoint_root, f"{exp_name}_{dataset}_seed{seed}", f"best_seed{seed}.pt"),
        # No repeat fallback
        os.path.join(checkpoint_root, f"{exp_name}_seed{seed}", f"best_seed{seed}.pt"),
        os.path.join(checkpoint_root, f"best_seed{seed}.pt"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    return None


# =============================================================================
# Metrics from Checkpoint
# =============================================================================

def load_metrics_from_checkpoint(ckpt_path: str) -> Dict[str, float]:
    """
    Load the best validation metrics that train_single saved into the checkpoint.
    Returns fine_acc, coarse_acc, joint_acc.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    metrics = ckpt.get("metrics", {})
    fine_acc   = float(metrics.get("fine_acc",   0.0))
    coarse_acc = float(metrics.get("coarse_acc", 0.0))
    # train_single saves joint as "joint" in metrics dict
    joint_acc  = float(metrics.get("joint",      fine_acc + coarse_acc))
    return {
        "fine_acc":   fine_acc,
        "coarse_acc": coarse_acc,
        "joint_acc":  joint_acc,
    }


# =============================================================================
# Aggregation & Statistics
# =============================================================================

def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute mean ± std across seeds for a list of result dicts."""
    fine_accs   = np.array([r["fine_acc"]   for r in results])
    coarse_accs = np.array([r["coarse_acc"] for r in results])
    joint_accs  = np.array([r["joint_acc"]  for r in results])

    return {
        "fine_acc_mean":   float(np.mean(fine_accs)),
        "fine_acc_std":    float(np.std(fine_accs)),
        "coarse_acc_mean": float(np.mean(coarse_accs)),
        "coarse_acc_std":  float(np.std(coarse_accs)),
        "joint_acc_mean":  float(np.mean(joint_accs)),
        "joint_acc_std":   float(np.std(joint_accs)),
        "n_seeds":         len(results),
        "individual_results": {
            "fine_acc":   fine_accs.tolist(),
            "coarse_acc": coarse_accs.tolist(),
            "joint_acc":  joint_accs.tolist(),
        },
    }


def compute_degradation(ab_stats: Dict, baseline: Dict) -> Dict[str, float]:
    return {
        "fine_degradation_pct":   float(baseline["fine_acc_mean"]   - ab_stats["fine_acc_mean"]),
        "coarse_degradation_pct": float(baseline["coarse_acc_mean"] - ab_stats["coarse_acc_mean"]),
        "joint_degradation_pct":  float(baseline["joint_acc_mean"]  - ab_stats["joint_acc_mean"]),
    }


def significance_test(ab_vals: np.ndarray, base_vals: np.ndarray) -> Tuple[float, float]:
    if len(ab_vals) != len(base_vals):
        t, p = scipy_stats.ttest_ind(ab_vals, base_vals)
    else:
        t, p = scipy_stats.ttest_rel(ab_vals, base_vals)
    return float(t), float(p)


# =============================================================================
# Reporting
# =============================================================================

def fmt(mean: float, std: float) -> str:
    return f"{mean:.2f} ± {std:.2f}"


def print_results_table(
    baseline: Dict,
    aggregated: Dict[str, Dict],
    ablations: List[Dict],
    dataset: str,
):
    print("\n" + "=" * 120)
    print(f"ABLATION STUDY RESULTS — {dataset.upper()}")
    print("=" * 120)
    print(f"{'Model':<32} {'Fine Acc':<20} {'Coarse Acc':<20} {'Joint Acc':<20} {'ΔJoint':<12}")
    print("-" * 120)

    print(
        f"{'FIN (full)':<32} "
        f"{fmt(baseline['fine_acc_mean'],   baseline['fine_acc_std']):<20} "
        f"{fmt(baseline['coarse_acc_mean'], baseline['coarse_acc_std']):<20} "
        f"{fmt(baseline['joint_acc_mean'],  baseline['joint_acc_std']):<20} "
        f"{'—':<12}"
    )
    print("-" * 120)

    for info in ablations:
        name = info["name"]
        if name not in aggregated:
            print(f"  {'— ' + name:<30} {'(no checkpoints found)'}")
            continue

        ab_stats = aggregated[name]
        deg      = compute_degradation(ab_stats, baseline)
        sig      = ""

        if ab_stats["n_seeds"] >= 2:
            _, p_val = significance_test(
                np.array(ab_stats["individual_results"]["joint_acc"]),
                np.array(baseline["individual_results"]["joint_acc"]),
            )
            if p_val < 0.01:   sig = "**"
            elif p_val < 0.05: sig = "*"

        print(
            f"{'— ' + name:<32} "
            f"{fmt(ab_stats['fine_acc_mean'],   ab_stats['fine_acc_std']):<20} "
            f"{fmt(ab_stats['coarse_acc_mean'], ab_stats['coarse_acc_std']):<20} "
            f"{fmt(ab_stats['joint_acc_mean'],  ab_stats['joint_acc_std']):<20} "
            f"{deg['joint_degradation_pct']:+.2f}{sig:<8}"
        )

    print("-" * 120)
    print("* p < 0.05, ** p < 0.01 (paired t-test vs full FIN)")


def print_importance_ranking(
    baseline: Dict,
    aggregated: Dict[str, Dict],
    ablations: List[Dict],
):
    print("\n" + "=" * 70)
    print("COMPONENT IMPORTANCE RANKING (by Joint Accuracy Degradation)")
    print("=" * 70)

    rankings = []
    for info in ablations:
        name = info["name"]
        if name not in aggregated:
            continue

        ab_stats = aggregated[name]
        deg      = compute_degradation(ab_stats, baseline)

        if ab_stats["n_seeds"] >= 2:
            _, p_val = significance_test(
                np.array(ab_stats["individual_results"]["joint_acc"]),
                np.array(baseline["individual_results"]["joint_acc"]),
            )
        else:
            p_val = 1.0

        rankings.append({
            "name":        name,
            "component":   info["component"],
            "degradation": deg["joint_degradation_pct"],
            "p_value":     float(p_val),
            "significant": bool(p_val < 0.05),
        })

    rankings.sort(key=lambda x: x["degradation"], reverse=True)

    print(f"{'Rank':<6} {'Component':<28} {'Ablation':<25} {'ΔJoint':>8}   {'p-value':>8}")
    print("-" * 80)
    for i, r in enumerate(rankings, 1):
        sig = "✓" if r["significant"] else "✗"
        print(
            f"{i:<6} {r['component']:<28} {r['name']:<25} "
            f"{r['degradation']:>+8.2f}%  {r['p_value']:>8.4f} {sig}"
        )
    print("-" * 80)
    print("✓ = statistically significant (p < 0.05)")


# =============================================================================
# Visualization
# =============================================================================

def create_plots(
    baseline: Dict,
    aggregated: Dict[str, Dict],
    ablations: List[Dict],
    output_dir: str,
    dataset: str,
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Warning] matplotlib not available, skipping plots")
        return

    present = [(info["name"], info["component"]) for info in ablations if info["name"] in aggregated]
    if not present:
        return

    present_sorted = sorted(present, key=lambda x: aggregated[x[0]]["joint_acc_mean"])
    names      = [x[0] for x in present_sorted]
    joint_means = [aggregated[n]["joint_acc_mean"] for n in names]
    joint_stds  = [aggregated[n]["joint_acc_std"]  for n in names]
    fine_means  = [aggregated[n]["fine_acc_mean"]  for n in names]
    fine_stds   = [aggregated[n]["fine_acc_std"]   for n in names]
    coarse_means = [aggregated[n]["coarse_acc_mean"] for n in names]
    coarse_stds  = [aggregated[n]["coarse_acc_std"]  for n in names]

    baseline_joint = baseline["joint_acc_mean"]
    x = np.arange(len(names))

    # ── Joint accuracy bar chart ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(x, joint_means, yerr=joint_stds, capsize=4, color="steelblue", alpha=0.8)

    for bar, mean in zip(bars, joint_means):
        deg = baseline_joint - mean
        bar.set_color("crimson" if deg > 5 else "darkorange" if deg > 2 else "forestgreen")

    ax.axhline(y=baseline_joint, color="black", linestyle="--", linewidth=2,
               label=f"Full FIN ({baseline_joint:.2f})")
    ax.set_xlabel("Ablation Variant", fontsize=12)
    ax.set_ylabel("Joint Accuracy", fontsize=12)
    ax.set_title(f"Component Importance: Joint Accuracy ({dataset.upper()})", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in names], ha="center", fontsize=9)
    ax.legend(loc="lower right")
    ax.set_ylim(0, max(joint_means + [baseline_joint]) * 1.15)
    ax.grid(axis="y", alpha=0.3)

    for bar, mean, std in zip(bars, joint_means, joint_stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.5,
                f"{mean:.1f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"ablation_joint_{dataset}.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── Multi-metric ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (title, means, stds, color) in zip(axes, [
        ("Fine Accuracy",   fine_means,   fine_stds,   "steelblue"),
        ("Coarse Accuracy", coarse_means, coarse_stds, "forestgreen"),
        ("Joint Accuracy",  joint_means,  joint_stds,  "darkorange"),
    ]):
        ax.bar(x, means, yerr=stds, capsize=4, color=color, alpha=0.8)
        ax.set_title(title, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace("_", "\n") for n in names], ha="center", fontsize=8)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"FIN Ablation Study — All Metrics ({dataset.upper()})", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"ablation_all_metrics_{dataset}.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── Degradation heatmap ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    deg_matrix = np.array([
        [compute_degradation(aggregated[n], baseline)[k]
         for k in ("fine_degradation_pct", "coarse_degradation_pct", "joint_degradation_pct")]
        for n in names
    ])
    im = ax.imshow(deg_matrix, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=15)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Fine Acc", "Coarse Acc", "Joint Acc"])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace("_", " ") for n in names])
    for i in range(len(names)):
        for j in range(3):
            ax.text(j, i, f"{deg_matrix[i, j]:.1f}%", ha="center", va="center", fontsize=10)
    ax.set_title(f"Accuracy Degradation by Component ({dataset.upper()})", fontsize=14)
    plt.colorbar(im, ax=ax, label="Degradation (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"ablation_degradation_heatmap_{dataset}.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[Plots] Saved to: {output_dir}")


# =============================================================================
# Save JSON
# =============================================================================

def save_json(
    baseline: Dict,
    aggregated: Dict[str, Dict],
    ablations: List[Dict],
    output_dir: str,
    dataset: str,
    seeds: List[int],
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "dataset": dataset,
        "seeds": seeds,
        "baseline": {
            "fine_acc_mean":   baseline["fine_acc_mean"],
            "fine_acc_std":    baseline["fine_acc_std"],
            "coarse_acc_mean": baseline["coarse_acc_mean"],
            "coarse_acc_std":  baseline["coarse_acc_std"],
            "joint_acc_mean":  baseline["joint_acc_mean"],
            "joint_acc_std":   baseline["joint_acc_std"],
            "n_seeds":         baseline["n_seeds"],
            "individual_results": baseline["individual_results"],
        },
        "ablations": {},
    }

    for info in ablations:
        name = info["name"]
        if name not in aggregated:
            continue

        ab_stats = aggregated[name]
        deg      = compute_degradation(ab_stats, baseline)

        if ab_stats["n_seeds"] >= 2:
            _, p_val = significance_test(
                np.array(ab_stats["individual_results"]["joint_acc"]),
                np.array(baseline["individual_results"]["joint_acc"]),
            )
        else:
            p_val = 1.0

        out["ablations"][name] = {
            "component":     info["component"],
            "stats":         ab_stats,
            "degradation":   deg,
            "p_value":       float(p_val),
            "significant":   bool(p_val < 0.05),
        }

    path = os.path.join(output_dir, f"ablation_eval_{dataset}_{timestamp}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[JSON] Saved: {path}")
    return path


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate already-trained FIN ablation checkpoints"
    )
    parser.add_argument("--config",   type=str, required=True,
                        help="Path to YAML config (e.g. configs/fin_cifar100.yaml)")
    parser.add_argument("--dataset",  type=str, default="cifar100",
                        choices=["cifar10", "cifar100"],
                        help="Dataset to evaluate (default: cifar100)")
    parser.add_argument("--seeds",    type=str, default="1,2,3",
                        help="Comma-separated seeds (default: '1,2,3')")
    parser.add_argument("--ablation", type=str, default=None,
                        help="Evaluate only this ablation (default: all)")
    parser.add_argument("--checkpoint_root", type=str, default="checkpoints",
                        help="Root directory for ablation checkpoints (default: checkpoints)")
    parser.add_argument("--output_dir", type=str, default="eval_outputs/ablations",
                        help="Where to save results (default: eval_outputs/ablations)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seeds   = parse_seeds(args.seeds)
    dataset = args.dataset
    os.makedirs(args.output_dir, exist_ok=True)
    dataset_output = os.path.join(args.output_dir, dataset)
    os.makedirs(dataset_output, exist_ok=True)

    ablations_to_run = (
        [a for a in ABLATIONS if a["name"] == args.ablation]
        if args.ablation else ABLATIONS
    )
    if not ablations_to_run:
        print(f"[Error] Unknown ablation: {args.ablation}")
        print(f"Available: {[a['name'] for a in ABLATIONS]}")
        sys.exit(1)

    exp_name      = cfg["experiment"]["name"]
    baseline_ckpt = cfg["experiment"]["checkpoint_dir"]

    print("\n" + "=" * 70)
    print(f"FIN ABLATION EVALUATION — {dataset.upper()}")
    print(f"Seeds: {seeds}  |  Ablations: {len(ablations_to_run)}")
    print("=" * 70)

    # ── Load baseline metrics ─────────────────────────────────────────────────
    print("\n[Baseline] Loading full FIN checkpoints...")
    baseline_results = []
    for seed in seeds:
        path = find_baseline_checkpoint(baseline_ckpt, exp_name, dataset, seed)
        if path:
            metrics = load_metrics_from_checkpoint(path)
            print(f"  Seed {seed}: {path}")
            print(f"           fine={metrics['fine_acc']:.2f}%  "
                  f"coarse={metrics['coarse_acc']:.2f}%  "
                  f"joint={metrics['joint_acc']:.2f}")
            baseline_results.append(metrics)
        else:
            print(f"  [ERROR] Baseline checkpoint not found for seed {seed}")
            print(f"  Searched in: {baseline_ckpt}")
            print(f"  Tried: {exp_name}_{dataset}_{dataset}_seed{seed}/best_seed{seed}.pt  (and variants)")

    if not baseline_results:
        print("\n[Error] No baseline checkpoints found. Run train.py first.")
        sys.exit(1)

    baseline = aggregate_results(baseline_results)
    print(f"\n[Baseline] joint={baseline['joint_acc_mean']:.2f} ± {baseline['joint_acc_std']:.2f} "
          f"over {baseline['n_seeds']} seed(s)")

    # ── Load ablation metrics ─────────────────────────────────────────────────
    aggregated = {}

    for info in ablations_to_run:
        name = info["name"]
        print(f"\n[Ablation] {name}")
        ab_results = []

        for seed in seeds:
            path = find_ablation_checkpoint(args.checkpoint_root, name, dataset, seed)
            if path:
                metrics = load_metrics_from_checkpoint(path)
                print(f"  Seed {seed}: {path}")
                print(f"           fine={metrics['fine_acc']:.2f}%  "
                      f"coarse={metrics['coarse_acc']:.2f}%  "
                      f"joint={metrics['joint_acc']:.2f}")
                ab_results.append(metrics)
            else:
                print(f"  [Missing] Seed {seed} — checkpoint not found under "
                      f"{args.checkpoint_root}/{name}/{dataset}/seed_{seed}/")

        if ab_results:
            aggregated[name] = aggregate_results(ab_results)
            print(f"  → joint={aggregated[name]['joint_acc_mean']:.2f} ± "
                  f"{aggregated[name]['joint_acc_std']:.2f} ({len(ab_results)} seed(s))")
        else:
            print(f"  [Skip] No checkpoints found for {name} — skipping")

    if not aggregated:
        print("\n[Error] No ablation checkpoints found. Check --checkpoint_root.")
        sys.exit(1)

    # ── Report ────────────────────────────────────────────────────────────────
    print_results_table(baseline, aggregated, ablations_to_run, dataset)
    print_importance_ranking(baseline, aggregated, ablations_to_run)

    # ── Plots ─────────────────────────────────────────────────────────────────
    create_plots(baseline, aggregated, ablations_to_run, dataset_output, dataset)

    # ── JSON ──────────────────────────────────────────────────────────────────
    save_json(baseline, aggregated, ablations_to_run, dataset_output, dataset, seeds)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()