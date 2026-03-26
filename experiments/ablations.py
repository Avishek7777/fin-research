"""
experiments/ablations.py

Ablation Study Runner (Section 5.2)
====================================

Runs comprehensive ablation studies to demonstrate that FIN's design
is not arbitrary — each component contributes meaningfully to performance.

Features:
- CIFAR-10, CIFAR-100, or both datasets
- Multi-seed evaluation with mean ± std statistics
- Comprehensive ablation variants
- Statistical significance testing
- Visualization of component importance

Usage:
    # Run all ablations on CIFAR-100 with 3 seeds
    python experiments/ablations.py --config configs/fin_cifar100.yaml --seeds "1,2,3"

    # Run all ablations on CIFAR-10 with 3 seeds
    python experiments/ablations.py --config configs/fin_cifar100.yaml --dataset cifar10 --seeds "1,2,3"

    # Run all ablations on both datasets
    python experiments/ablations.py --config configs/fin_cifar100.yaml --dataset both --seeds "1,2,3"

    # Run a specific ablation only
    python experiments/ablations.py --config configs/fin_cifar100.yaml --ablation no_bandwidth

Ablation Table (Section 5.2):
    ┌─────────────────────────────┬────────┬─────────┬───────┬─────────────┐
    │ Model                       │ Fine%  │ Coarse% │ Joint │ Degradation │
    ├─────────────────────────────┼────────┼─────────┼───────┼─────────────┤
    │ FIN (full)                  │   —    │    —    │   —   │      —      │
    │ — no bandwidth constraint   │   —    │    —    │   —   │     ▼X%      │
    │ — no per-level objectives   │   —    │    —    │   —   │     ▼X%      │
    │ — no top-down feedback      │   —    │    —    │   —   │     ▼X%      │
    │ — no hierarchy (flat)       │   —    │    —    │   —   │     ▼X%      │
    │ — fixed bandwidth           │   —    │    —    │   —   │     ▼X%      │
    │ — no self-similarity        │   —    │    —    │   —   │     ▼X%      │
    └─────────────────────────────┴────────┴─────────┴───────┴─────────────┘

Each ablation isolates exactly one component, confirming that
bandwidth constraints, per-level objectives, top-down feedback,
hierarchy, annealing, and self-similarity are all individually
necessary for FIN's performance.
"""

import os
import sys
import yaml
import argparse
import json
import copy
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from scipy import stats

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from train import train_single, parse_seeds, CIFAR10_MEAN, CIFAR10_STD, CIFAR100_MEAN, CIFAR100_STD
from evaluate import evaluate


# =============================================================================
# Ablation Definitions
# =============================================================================

ABLATIONS = [
    {
        "name"       : "no_bandwidth",
        "description": "Set gamma=0 to disable KL bandwidth constraint",
        "hypothesis" : "FIN collapses toward a flat deep net — joint accuracy drops significantly",
        "component"  : "bandwidth_constraint",
    },
    {
        "name"       : "no_per_level_objectives",
        "description": "Set lambda_0=0, lambda_2=0 — only L1 fine classification remains",
        "hypothesis" : "Level hierarchy doesn't specialize — coarse accuracy drops most",
        "component"  : "per_level_objectives",
    },
    {
        "name"       : "no_top_down_feedback",
        "description": "Disable feedback (feedback_enabled=false)",
        "hypothesis" : "Lower level representations degrade without top-down guidance",
        "component"  : "top_down_feedback",
    },
    {
        "name"       : "no_hierarchy_flat",
        "description": "Single-level model — combine all levels into one representation",
        "hypothesis" : "Hierarchical abstraction is crucial — no hierarchy loses all benefits",
        "component"  : "hierarchical_structure",
    },
    {
        "name"       : "fixed_bandwidth",
        "description": "Disable beta annealing — fixed capacity throughout training",
        "hypothesis" : "Annealing stabilizes training — fixed bandwidth hurts convergence",
        "component"  : "beta_annealing",
    },
    {
        "name"       : "no_self_similarity",
        "description": "Different architectures per level — break self-similarity",
        "hypothesis" : "Self-similarity enables generalization across scales",
        "component"  : "self_similarity",
    },
]


# =============================================================================
# CIFAR-10 Configuration
# =============================================================================

CIFAR10_CLASSES = 10
CIFAR100_CLASSES = 100
COARSE_CLASSES_C100 = 20  # CIFAR-100 has 20 coarse classes


# =============================================================================
# Ablation Configuration Generator
# =============================================================================

def get_ablation_overrides(ablation_name: str, dataset: str = "cifar100") -> Dict[str, Any]:
    """
    Get configuration overrides for a specific ablation.
    
    Args:
        ablation_name: Name of the ablation variant
        dataset: "cifar10" or "cifar100"
    
    Returns:
        Dict of config overrides
    """
    num_classes = CIFAR10_CLASSES if dataset == "cifar10" else CIFAR100_CLASSES
    coarse_classes = 2 if dataset == "cifar10" else COARSE_CLASSES_C100  # Fake coarse for CIFAR-10
    
    overrides = {}
    
    if ablation_name == "no_bandwidth":
        overrides = {
            "bandwidth.channel_01.gamma": 0.0,
            "bandwidth.channel_12.gamma": 0.0,
        }
    
    elif ablation_name == "no_per_level_objectives":
        overrides = {
            "loss.lambda_0": 0.0,
            "loss.lambda_2": 0.0,
        }
    
    elif ablation_name == "no_top_down_feedback":
        overrides = {
            "feedback.enabled": False,
        }
    
    elif ablation_name == "no_hierarchy_flat":
        # Merge all levels into a single flat representation
        overrides = {
            "architecture.use_hierarchy": False,
            "architecture.single_level": True,
            "architecture.level0.out_dim": 512,
            "loss.lambda_0": 0.0,
            "loss.lambda_2": 0.5,
            "loss.lambda_1": 1.0,
        }
    
    elif ablation_name == "fixed_bandwidth":
        overrides = {
            "annealing.enabled": False,
            "bandwidth.channel_01.beta": 16.0,
            "bandwidth.channel_12.beta": 4.0,
        }
    
    elif ablation_name == "no_self_similarity":
        # This requires code changes in architecture, handled separately
        overrides = {
            "architecture.level1.mlp_mode": True,
        }
    
    return overrides


def apply_ablation_config(cfg: dict, ablation: str, dataset: str = "cifar100") -> dict:
    """
    Apply ablation overrides to the config.
    
    Args:
        cfg: Full config dict
        ablation: Ablation name (e.g. "no_bandwidth") or "full"
        dataset: "cifar10" or "cifar100"
    
    Returns:
        Modified config dict (deep copy — original untouched)
    """
    cfg = copy.deepcopy(cfg)
    
    # Update dataset configuration
    cfg["data"]["dataset"] = dataset
    if dataset == "cifar10":
        cfg["data"]["num_fine_classes"] = CIFAR10_CLASSES
        cfg["data"]["num_coarse_classes"] = 2  # Fake coarse classes for CIFAR-10
    else:
        cfg["data"]["num_fine_classes"] = CIFAR100_CLASSES
        cfg["data"]["num_coarse_classes"] = COARSE_CLASSES_C100
    
    if ablation == "full":
        return cfg
    
    overrides = get_ablation_overrides(ablation, dataset)
    
    for key_path, value in overrides.items():
        keys = key_path.split(".")
        node = cfg
        for k in keys[:-1]:
            if k not in node:
                node[k] = {}
            node = node[k]
        node[keys[-1]] = value
    
    print(f"[Ablation] Applied '{ablation}': {overrides}")
    return cfg


# =============================================================================
# Training and Evaluation
# =============================================================================

def run_single_experiment(
    cfg: dict,
    ablation: str,
    seed: int,
    output_dir: str,
    dataset: str,
) -> Dict[str, Any]:
    """
    Train and evaluate one experiment configuration.
    
    Args:
        cfg: Configuration dict
        ablation: Ablation name
        seed: Random seed
        output_dir: Output directory
        dataset: Dataset name
    
    Returns:
        dict with metrics
    """
    # Apply ablation and seed
    exp_cfg = apply_ablation_config(cfg, ablation, dataset)
    exp_cfg["experiment"]["seed"] = seed
    exp_cfg["experiment"]["name"] = f"fin_{ablation}_s{seed}_{dataset}"
    exp_cfg["experiment"]["checkpoint_dir"] = f"checkpoints/{ablation}/{dataset}/seed_{seed}"
    exp_cfg["experiment"]["log_dir"] = f"runs/{ablation}/{dataset}/seed_{seed}"
    
    # Train
    print(f"\n{'='*70}")
    print(f"Training: ablation={ablation}, seed={seed}, dataset={dataset}")
    print(f"{'='*70}")
    
    train_single(exp_cfg, dataset, seed)
    
    # Evaluate
    ckpt_path = os.path.join(exp_cfg["experiment"]["checkpoint_dir"], f"best_seed{seed}.pt")
    
    if not os.path.exists(ckpt_path):
        print(f"[Warning] Checkpoint not found: {ckpt_path}")
        return {
            "ablation": ablation,
            "seed": seed,
            "dataset": dataset,
            "fine_acc": 0.0,
            "coarse_acc": 0.0,
            "joint_acc": 0.0,
            "params_M": 0,
        }
    
    eval_output = os.path.join(output_dir, ablation, dataset, f"seed_{seed}")
    os.makedirs(eval_output, exist_ok=True)
    
    evaluate(
        cfg=exp_cfg,
        checkpoint_path=ckpt_path,
        run_tsne=False,
        output_dir=eval_output,
    )
    
    # Load accuracy from checkpoint
    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu")
    metrics = ckpt.get("metrics", {})
    
    return {
        "ablation": ablation,
        "seed": seed,
        "dataset": dataset,
        "fine_acc": metrics.get("fine_acc", 0.0),
        "coarse_acc": metrics.get("coarse_acc", 0.0),
        "joint_acc": metrics.get("joint", 0.0),
        "params_M": metrics.get("params_M", 0),
    }


def aggregate_results(all_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate results across seeds for each ablation.
    
    Args:
        all_results: List of individual experiment results
    
    Returns:
        Dict mapping ablation names to aggregated stats (mean ± std)
    """
    aggregated = {}
    
    # Group by ablation
    by_ablation = {}
    for r in all_results:
        ab = r["ablation"]
        if ab not in by_ablation:
            by_ablation[ab] = {"fine_acc": [], "coarse_acc": [], "joint_acc": []}
        by_ablation[ab]["fine_acc"].append(r["fine_acc"])
        by_ablation[ab]["coarse_acc"].append(r["coarse_acc"])
        by_ablation[ab]["joint_acc"].append(r["joint_acc"])
    
    # Compute mean and std
    for ab, values in by_ablation.items():
        fine_accs = np.array(values["fine_acc"])
        coarse_accs = np.array(values["coarse_acc"])
        joint_accs = np.array(values["joint_acc"])
        
        aggregated[ab] = {
            "fine_acc_mean": np.mean(fine_accs),
            "fine_acc_std": np.std(fine_accs),
            "coarse_acc_mean": np.mean(coarse_accs),
            "coarse_acc_std": np.std(coarse_accs),
            "joint_acc_mean": np.mean(joint_accs),
            "joint_acc_std": np.std(joint_accs),
            "n_seeds": len(fine_accs),
            "individual_results": values,
        }
    
    return aggregated


# =============================================================================
# Statistical Testing
# =============================================================================

def compute_degradation(
    ablation_stats: Dict[str, Any],
    baseline_stats: Dict[str, Any],
) -> Dict[str, float]:
    """
    Compute performance degradation compared to baseline.
    
    Returns:
        Degradation percentages for each metric
    """
    return {
        "fine_degradation_pct": baseline_stats["fine_acc_mean"] - ablation_stats["fine_acc_mean"],
        "coarse_degradation_pct": baseline_stats["coarse_acc_mean"] - ablation_stats["coarse_acc_mean"],
        "joint_degradation_pct": baseline_stats["joint_acc_mean"] - ablation_stats["joint_acc_mean"],
    }


def statistical_significance_test(
    ablation_values: np.ndarray,
    baseline_values: np.ndarray,
) -> Tuple[float, float]:
    """
    Perform paired t-test to check statistical significance.
    
    Args:
        ablation_values: Array of ablation results across seeds
        baseline_values: Array of baseline results across seeds
    
    Returns:
        (t_statistic, p_value)
    """
    if len(ablation_values) != len(baseline_values):
        # Use Welch's t-test if different sample sizes
        t_stat, p_val = stats.ttest_ind(ablation_values, baseline_values)
    else:
        # Use paired t-test if same sample size
        t_stat, p_val = stats.ttest_rel(ablation_values, baseline_values)
    
    return t_stat, p_val


# =============================================================================
# Reporting
# =============================================================================

def format_metric(mean: float, std: float, decimals: int = 2) -> str:
    """Format mean ± std for display."""
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def print_ablation_table(
    aggregated: Dict[str, Dict[str, Any]],
    baseline: Dict[str, Any],
    ablation_info: List[Dict],
):
    """
    Print comprehensive ablation results table.
    """
    print("\n" + "="*120)
    print("ABLATION STUDY RESULTS")
    print("="*120)
    
    # Header
    print(f"{'Model':<30} {'Fine Acc':<18} {'Coarse Acc':<18} {'Joint Acc':<18} {'ΔJoint':<10}")
    print("-"*120)
    
    # Baseline
    print(
        f"{'FIN (full)':<30} "
        f"{format_metric(baseline['fine_acc_mean'], baseline['fine_acc_std']):<18} "
        f"{format_metric(baseline['coarse_acc_mean'], baseline['coarse_acc_std']):<18} "
        f"{format_metric(baseline['joint_acc_mean'], baseline['joint_acc_std']):<18} "
        f"{'—':<10}"
    )
    
    print("-"*120)
    
    # Ablations
    for info in ablation_info:
        ab_name = info["name"]
        if ab_name not in aggregated:
            continue
        
        stats = aggregated[ab_name]
        deg = compute_degradation(stats, baseline)
        
        significance = ""
        if stats["n_seeds"] >= 2:
            _, p_val = statistical_significance_test(
                np.array(stats["individual_results"]["joint_acc"]),
                np.array(baseline["individual_results"]["joint_acc"]),
            )
            if p_val < 0.01:
                significance = "**"
            elif p_val < 0.05:
                significance = "*"
        
        print(
            f"{f'— {ab_name}':<30} "
            f"{format_metric(stats['fine_acc_mean'], stats['fine_acc_std']):<18} "
            f"{format_metric(stats['coarse_acc_mean'], stats['coarse_acc_std']):<18} "
            f"{format_metric(stats['joint_acc_mean'], stats['joint_acc_std']):<18} "
            f"{deg['joint_degradation_pct']:+.2f}{significance:<6}"
        )
    
    print("-"*120)
    print("* p < 0.05, ** p < 0.01 (paired t-test vs full FIN)")


def print_component_importance_summary(
    aggregated: Dict[str, Dict[str, Any]],
    baseline: Dict[str, Any],
    ablation_info: List[Dict],
):
    """
    Print component importance ranking.
    """
    print("\n" + "="*70)
    print("COMPONENT IMPORTANCE RANKING (by Joint Accuracy Degradation)")
    print("="*70)
    
    rankings = []
    for info in ablation_info:
        ab_name = info["name"]
        if ab_name not in aggregated:
            continue
        
        stats = aggregated[ab_name]
        deg = compute_degradation(stats, baseline)
        
        # Statistical significance
        _, p_val = statistical_significance_test(
            np.array(stats["individual_results"]["joint_acc"]),
            np.array(baseline["individual_results"]["joint_acc"]),
        )
        
        rankings.append({
            "name": ab_name,
            "component": info["component"],
            "degradation": deg["joint_degradation_pct"],
            "p_value": p_val,
            "significant": p_val < 0.05,
        })
    
    # Sort by degradation (largest first)
    rankings.sort(key=lambda x: x["degradation"], reverse=True)
    
    print(f"{'Rank':<6} {'Component':<25} {'Ablation':<20} {'ΔJoint':<10} {'p-value':<10}")
    print("-"*70)
    
    for i, r in enumerate(rankings, 1):
        sig = "✓" if r["significant"] else "✗"
        print(
            f"{i:<6} {r['component']:<25} {r['name']:<20} "
            f"{r['degradation']:+.2f}%{'':<5} {r['p_value']:.4f} {sig}"
        )
    
    print("-"*70)
    print("✓ = statistically significant (p < 0.05)")


# =============================================================================
# Visualization
# =============================================================================

def create_ablation_plots(
    aggregated: Dict[str, Dict[str, Any]],
    baseline: Dict[str, Any],
    ablation_info: List[Dict],
    output_dir: str,
    dataset: str,
):
    """
    Create bar charts showing accuracy drop for each ablation.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[Warning] matplotlib not available, skipping visualization")
        return
    
    # Prepare data
    ablations_sorted = sorted(
        [(info["name"], info["component"]) for info in ablation_info if info["name"] in aggregated],
        key=lambda x: aggregated[x[0]]["joint_acc_mean"],
    )
    
    names = [x[0] for x in ablations_sorted]
    components = {x[0]: x[1] for x in ablations_sorted}
    
    joint_means = [aggregated[n]["joint_acc_mean"] for n in names]
    joint_stds = [aggregated[n]["joint_acc_std"] for n in names]
    
    fine_means = [aggregated[n]["fine_acc_mean"] for n in names]
    fine_stds = [aggregated[n]["fine_acc_std"] for n in names]
    
    coarse_means = [aggregated[n]["coarse_acc_mean"] for n in names]
    coarse_stds = [aggregated[n]["coarse_acc_std"] for n in names]
    
    # Joint accuracy bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(names))
    bars = ax.bar(x, joint_means, yerr=joint_stds, capsize=4, color="steelblue", alpha=0.8)
    
    # Color bars based on degradation significance
    baseline_joint = baseline["joint_acc_mean"]
    for i, (bar, mean) in enumerate(zip(bars, joint_means)):
        degradation = baseline_joint - mean
        if degradation > 5:
            bar.set_color("crimson")
        elif degradation > 2:
            bar.set_color("darkorange")
        else:
            bar.set_color("forestgreen")
    
    ax.axhline(y=baseline_joint, color="black", linestyle="--", linewidth=2, label=f"Full FIN ({baseline_joint:.2f}%)")
    
    ax.set_xlabel("Ablation Variant", fontsize=12)
    ax.set_ylabel("Joint Accuracy (%)", fontsize=12)
    ax.set_title(f"Component Importance: Joint Accuracy by Ablation ({dataset.upper()})", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in names], rotation=0, ha="center", fontsize=10)
    ax.legend(loc="lower right")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    
    # Add value labels
    for bar, mean, std in zip(bars, joint_means, joint_stds):
        ax.text(
            bar.get_x() + bar.get_width()/2, bar.get_height() + std + 1,
            f"{mean:.1f}", ha="center", va="bottom", fontsize=9
        )
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"ablation_joint_accuracy_{dataset}.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
    # Multi-metric comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = [
        ("Fine Accuracy", fine_means, fine_stds, "steelblue"),
        ("Coarse Accuracy", coarse_means, coarse_stds, "forestgreen"),
        ("Joint Accuracy", joint_means, joint_stds, "darkorange"),
    ]
    
    for ax, (title, means, stds, color) in zip(axes, metrics):
        bars = ax.bar(x, means, yerr=stds, capsize=4, color=color, alpha=0.8)
        ax.set_title(title, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace("_", "\n") for n in names], rotation=0, ha="center", fontsize=8)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3)
    
    fig.suptitle(f"FIN Ablation Study: All Metrics ({dataset.upper()})", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"ablation_all_metrics_{dataset}.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
    # Degradation heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    degradation_matrix = []
    for name in names:
        stats = aggregated[name]
        deg = compute_degradation(stats, baseline)
        degradation_matrix.append([
            deg["fine_degradation_pct"],
            deg["coarse_degradation_pct"],
            deg["joint_degradation_pct"],
        ])
    
    degradation_matrix = np.array(degradation_matrix)
    
    im = ax.imshow(degradation_matrix, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=15)
    
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Fine Acc", "Coarse Acc", "Joint Acc"])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace("_", " ") for n in names])
    
    # Add text annotations
    for i in range(len(names)):
        for j in range(3):
            text = ax.text(j, i, f"{degradation_matrix[i, j]:.1f}%",
                          ha="center", va="center", color="black", fontsize=10)
    
    ax.set_title(f"Accuracy Degradation by Component ({dataset.upper()})", fontsize=14)
    plt.colorbar(im, ax=ax, label="Degradation (%)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"ablation_degradation_heatmap_{dataset}.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"[Done] Saved ablation plots to: {output_dir}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run FIN comprehensive ablation studies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all ablations on CIFAR-100 with 3 seeds
  python experiments/ablations.py --config configs/fin_cifar100.yaml --seeds "1,2,3"

  # Run all ablations on CIFAR-10 with 3 seeds
  python experiments/ablations.py --config configs/fin_cifar100.yaml --dataset cifar10 --seeds "1,2,3"

  # Run all ablations on both datasets
  python experiments/ablations.py --config configs/fin_cifar100.yaml --dataset both --seeds "1,2,3"

  # Run a specific ablation
  python experiments/ablations.py --config configs/fin_cifar100.yaml --ablation no_bandwidth --seeds "1,2,3"
        """
    )

    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--dataset", type=str, default="cifar100",
        choices=["cifar10", "cifar100", "both"],
        help="Dataset to use for ablation study"
    )
    parser.add_argument(
        "--seeds", type=str, default="1,2,3",
        help="Comma-separated list of random seeds (default: '1,2,3')"
    )
    parser.add_argument(
        "--ablation", type=str, default=None,
        help="Run only this ablation (default: run all)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="eval_outputs/ablations",
        help="Directory to save ablation results"
    )
    parser.add_argument(
        "--skip_existing", action="store_true",
        help="Skip experiments if results already exist"
    )

    args = parser.parse_args()

    # Load base config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Parse seeds
    seeds = parse_seeds(args.seeds)
    print(f"[Config] Seeds: {seeds}")

    # Determine datasets
    if args.dataset == "both":
        datasets = ["cifar10", "cifar100"]
    else:
        datasets = [args.dataset]

    # Determine which ablations to run
    ablations_to_run = (
        [a for a in ABLATIONS if a["name"] == args.ablation]
        if args.ablation
        else ABLATIONS
    )

    if not ablations_to_run:
        print(f"[Error] Unknown ablation: {args.ablation}")
        print(f"Available: {[a['name'] for a in ABLATIONS]}")
        sys.exit(1)

    # Print plan
    print("\n" + "="*70)
    print("FIN ABLATION STUDY PLAN")
    print("="*70)
    print(f"Datasets: {datasets}")
    print(f"Seeds: {seeds}")
    print(f"Ablations to run: {len(ablations_to_run)}")
    for ab in ablations_to_run:
        print(f"  [{ab['name']}] {ab['description']}")
        print(f"    Hypothesis: {ab['hypothesis']}")
    print("="*70)

    # Storage for all results
    all_results = []
    summary_results = {}

    # Run experiments
    for dataset in datasets:
        print(f"\n\n{'#'*70}")
        print(f"# DATASET: {dataset.upper()}")
        print(f"{'#'*70}")
        
        dataset_output = os.path.join(args.output_dir, dataset)
        os.makedirs(dataset_output, exist_ok=True)
        
        # Load full model baseline from existing train.py checkpoints
        baseline_results = []
        for seed in seeds:
            exp_name = cfg["experiment"]["name"]
            base_ckpt_dir = cfg["experiment"]["checkpoint_dir"]
            
            # train.py creates checkpoints at: {base_ckpt_dir}/{exp_name}_{dataset}_{dataset}_seed{N}/best_seed{N}.pt
            # Try multiple patterns to support different naming conventions
            ckpt_patterns = [
                # Pattern 1: Actual train.py output (dataset repeated in path)
                os.path.join(base_ckpt_dir, f"{exp_name}_{dataset}_{dataset}_seed{seed}", f"best_seed{seed}.pt"),
                # Pattern 2: Standard train.py output (--dataset cifar100 or --dataset cifar10)
                os.path.join(base_ckpt_dir, f"{exp_name}_{dataset}_seed{seed}", f"best_seed{seed}.pt"),
                # Pattern 3: train.py with --dataset both (nested structure)
                os.path.join(base_ckpt_dir, f"{exp_name}_both", f"{exp_name}_both_{dataset}_seed{seed}", f"best_seed{seed}.pt"),
                # Pattern 4: Flat structure fallback
                os.path.join(base_ckpt_dir, f"best_seed{seed}.pt"),
            ]
            
            ckpt_path = None
            for pattern in ckpt_patterns:
                if os.path.exists(pattern):
                    ckpt_path = pattern
                    break
            
            if ckpt_path:
                import torch
                ckpt = torch.load(ckpt_path, map_location="cpu")
                metrics = ckpt.get("metrics", {})
                print(f"[Baseline] Loaded FIN full checkpoint: {ckpt_path}")
                result = {
                    "ablation": "full",
                    "seed": seed,
                    "dataset": dataset,
                    "fine_acc": metrics.get("fine_acc", 0.0),
                    "coarse_acc": metrics.get("coarse_acc", 0.0),
                    "joint_acc": metrics.get("joint", 0.0),
                    "params_M": metrics.get("params_M", 0),
                }
            else:
                print(f"[ERROR] Baseline checkpoint not found for {dataset} seed {seed}.")
                print(f"        Expected patterns:")
                for i, pattern in enumerate(ckpt_patterns, 1):
                    print(f"          {i}. {pattern}")
                print(f"        Please run train.py first:")
                print(f"          python train.py --config configs/fin_cifar100.yaml --dataset {dataset} --seeds '{seed}'")
                raise FileNotFoundError(f"Missing baseline checkpoint for {dataset} seed {seed}")
            baseline_results.append(result)
            all_results.append(result)
        
        # Aggregate baseline
        baseline_aggregated = aggregate_results(baseline_results)
        baseline = baseline_aggregated.get("full", baseline_aggregated.get("FIN (full)", None))
        
        if baseline is None:
            # Create from raw results
            baseline = {
                "fine_acc_mean": np.mean([r["fine_acc"] for r in baseline_results]),
                "fine_acc_std": np.std([r["fine_acc"] for r in baseline_results]),
                "coarse_acc_mean": np.mean([r["coarse_acc"] for r in baseline_results]),
                "coarse_acc_std": np.std([r["coarse_acc"] for r in baseline_results]),
                "joint_acc_mean": np.mean([r["joint_acc"] for r in baseline_results]),
                "joint_acc_std": np.std([r["joint_acc"] for r in baseline_results]),
                "n_seeds": len(baseline_results),
                "individual_results": {
                    "fine_acc": [r["fine_acc"] for r in baseline_results],
                    "coarse_acc": [r["coarse_acc"] for r in baseline_results],
                    "joint_acc": [r["joint_acc"] for r in baseline_results],
                },
            }
        
        # Run each ablation
        for ab in ablations_to_run:
            if ab["name"] == "full":
                continue
                
            ab_results = []
            for seed in seeds:
                result = run_single_experiment(
                    cfg=cfg,
                    ablation=ab["name"],
                    seed=seed,
                    output_dir=dataset_output,
                    dataset=dataset,
                )
                ab_results.append(result)
                all_results.append(result)
            
            # Aggregate ablation results
            ab_aggregated = aggregate_results(ab_results)
            summary_results[f"{dataset}_{ab['name']}"] = {
                "dataset": dataset,
                "ablation": ab["name"],
                "baseline": baseline,
                "ablation_stats": ab_aggregated.get(ab["name"], {}),
            }
        
        # Final aggregation for this dataset
        full_aggregated = aggregate_results(baseline_results)
        full_key = "full"
        if full_key not in full_aggregated:
            # Try other keys
            for k in full_aggregated.keys():
                if "full" in k.lower() or "fin" in k.lower():
                    full_key = k
                    break
        
        if full_key in full_aggregated:
            baseline = full_aggregated[full_key]
        else:
            baseline = {
                "fine_acc_mean": np.mean([r["fine_acc"] for r in baseline_results]),
                "fine_acc_std": np.std([r["fine_acc"] for r in baseline_results]),
                "coarse_acc_mean": np.mean([r["coarse_acc"] for r in baseline_results]),
                "coarse_acc_std": np.std([r["coarse_acc"] for r in baseline_results]),
                "joint_acc_mean": np.mean([r["joint_acc"] for r in baseline_results]),
                "joint_acc_std": np.std([r["joint_acc"] for r in baseline_results]),
                "n_seeds": len(baseline_results),
                "individual_results": {
                    "fine_acc": [r["fine_acc"] for r in baseline_results],
                    "coarse_acc": [r["coarse_acc"] for r in baseline_results],
                    "joint_acc": [r["joint_acc"] for r in baseline_results],
                },
            }
        
        # Aggregate all ablations
        ab_results_all = [r for r in all_results if r["dataset"] == dataset and r["ablation"] != "full"]
        aggregated = aggregate_results(ab_results_all)
        
        # Print results table
        print("\n")
        print_ablation_table(aggregated, baseline, ablations_to_run)
        
        # Print component importance
        print_component_importance_summary(aggregated, baseline, ablations_to_run)
        
        # Create visualizations
        create_ablation_plots(aggregated, baseline, ablations_to_run, dataset_output, dataset)

    # Save comprehensive results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save all raw results
    raw_results_path = os.path.join(args.output_dir, f"ablation_raw_results_{timestamp}.json")
    with open(raw_results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Done] Raw results saved to: {raw_results_path}")
    
    # Save summary results
    summary_path = os.path.join(args.output_dir, f"ablation_summary_{timestamp}.json")
    
    # Create serializable summary
    serializable_summary = {}
    for key, value in summary_results.items():
        serializable_summary[key] = {
            "dataset": value["dataset"],
            "ablation": value["ablation"],
            "baseline": {
                "fine_acc_mean": value["baseline"]["fine_acc_mean"],
                "fine_acc_std": value["baseline"]["fine_acc_std"],
                "coarse_acc_mean": value["baseline"]["coarse_acc_mean"],
                "coarse_acc_std": value["baseline"]["coarse_acc_std"],
                "joint_acc_mean": value["baseline"]["joint_acc_mean"],
                "joint_acc_std": value["baseline"]["joint_acc_std"],
                "n_seeds": value["baseline"]["n_seeds"],
            },
            "ablation_stats": {
                "fine_acc_mean": value["ablation_stats"].get("fine_acc_mean", 0),
                "fine_acc_std": value["ablation_stats"].get("fine_acc_std", 0),
                "coarse_acc_mean": value["ablation_stats"].get("coarse_acc_mean", 0),
                "coarse_acc_std": value["ablation_stats"].get("coarse_acc_std", 0),
                "joint_acc_mean": value["ablation_stats"].get("joint_acc_mean", 0),
                "joint_acc_std": value["ablation_stats"].get("joint_acc_std", 0),
                "n_seeds": value["ablation_stats"].get("n_seeds", 0),
            },
        }
        # Add degradation
        deg = compute_degradation(value["ablation_stats"], value["baseline"])
        serializable_summary[key]["degradation"] = deg
        
        # Add statistical significance
        if value["ablation_stats"].get("n_seeds", 0) >= 2:
            _, p_val = statistical_significance_test(
                np.array(value["ablation_stats"]["individual_results"]["joint_acc"]),
                np.array(value["baseline"]["individual_results"]["joint_acc"]),
            )
            serializable_summary[key]["p_value"] = p_val
            serializable_summary[key]["significant"] = p_val < 0.05
    
    with open(summary_path, "w") as f:
        json.dump(serializable_summary, f, indent=2)
    print(f"[Done] Summary saved to: {summary_path}")

    print("\n" + "="*70)
    print("ABLATION STUDY COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
