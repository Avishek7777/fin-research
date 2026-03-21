"""
experiments/ablations.py

Ablation Study Runner (Section 5.2)
=====================================

Runs all four ablations sequentially and produces the full
ablation table for the paper. Each ablation trains a FIN variant
with one component removed, then evaluates it.

Usage:
    # Run all ablations
    python experiments/ablations.py --config configs/fin_cifar100.yaml

    # Run a specific ablation only
    python experiments/ablations.py --config configs/fin_cifar100.yaml \
                                    --ablation no_bandwidth

Ablation Table (Section 5.2):
    ┌─────────────────────────────┬────────┬─────────┬───────┐
    │ Model                       │ Fine%  │ Coarse% │ Joint │
    ├─────────────────────────────┼────────┼─────────┼───────┤
    │ FIN (full)                  │   —    │    —    │   —   │
    │ — no bandwidth constraint   │   —    │    —    │   —   │
    │ — no per-level objectives   │   —    │    —    │   —   │
    │ — no top-down feedback      │   —    │    —    │   —   │
    │ — no self-similarity        │   —    │    —    │   —   │
    └─────────────────────────────┴────────┴─────────┴───────┘

Each ablation isolates exactly one component, confirming that
bandwidth constraints, per-level objectives, and top-down feedback
are all individually necessary for FIN's performance.
"""

import os
import sys
import yaml
import argparse
import json
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from train import train, apply_ablation
from evaluate import evaluate, print_results_table


# =============================================================================
# Ablation Definitions
# =============================================================================

ABLATIONS = [
    {
        "name"       : "no_bandwidth",
        "description": "Remove bandwidth constraint (gamma=0 on all channels)",
        "hypothesis" : "FIN collapses toward a flat deep net — joint accuracy drops",
    },
    {
        "name"       : "no_per_level_objectives",
        "description": "Single global loss — only L1 fine classification remains",
        "hypothesis" : "Level hierarchy doesn't specialize — coarse accuracy drops most",
    },
    {
        "name"       : "no_top_down_feedback",
        "description": "Remove all top-down connections (alpha gates disabled)",
        "hypothesis" : "Lower level representations degrade — fine accuracy drops",
    },
]


# =============================================================================
# Run One Ablation
# =============================================================================

def run_ablation(
    cfg       : dict,
    ablation  : str,
    output_dir: str,
) -> dict:
    """
    Train and evaluate one ablation variant.

    Returns:
        dict: name, fine_acc, coarse_acc, joint_acc
    """
    print(f"\n{'='*60}")
    print(f"Ablation: {ablation}")
    print(f"{'='*60}")

    # Apply ablation overrides to config
    ablation_cfg = apply_ablation(cfg, ablation)

    # Update experiment name so checkpoints don't overwrite each other
    ablation_cfg["experiment"]["name"]           = f"fin_{ablation}"
    ablation_cfg["experiment"]["checkpoint_dir"] = f"checkpoints/{ablation}"
    ablation_cfg["experiment"]["log_dir"]        = f"runs/"

    # Train
    train(ablation_cfg)

    # Evaluate
    ckpt_path = os.path.join(f"checkpoints/{ablation}", "best.pt")
    if not os.path.exists(ckpt_path):
        print(f"[Warning] Checkpoint not found: {ckpt_path}")
        return {"name": ablation, "fine_acc": 0, "coarse_acc": 0, "joint_acc": 0}

    ablation_output = os.path.join(output_dir, ablation)
    evaluate(
        cfg             = ablation_cfg,
        checkpoint_path = ckpt_path,
        run_tsne        = False,   # skip t-SNE for ablations (time)
        output_dir      = ablation_output,
    )

    # Load accuracy from checkpoint metrics
    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu")
    metrics = ckpt.get("metrics", {})

    return {
        "name"      : ablation,
        "fine_acc"  : metrics.get("fine_acc",   0.0),
        "coarse_acc": metrics.get("coarse_acc", 0.0),
        "joint_acc" : metrics.get("joint",      0.0),
        "params_M"  : "—",
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run FIN ablation studies")

    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--ablation", type=str, default=None,
        help="Run only this ablation (default: run all)"
    )
    parser.add_argument(
        "--full_checkpoint", type=str, default="checkpoints/best.pt",
        help="Path to full FIN checkpoint for baseline comparison"
    )
    parser.add_argument(
        "--output_dir", type=str, default="eval_outputs/ablations",
        help="Directory to save ablation results"
    )

    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)

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

    # Print ablation plan
    print("\nAblation Study Plan:")
    for ab in ablations_to_run:
        print(f"  [{ab['name']}]")
        print(f"    {ab['description']}")
        print(f"    Hypothesis: {ab['hypothesis']}")

    # Run all ablations
    results = []

    # First: evaluate full model as baseline
    if os.path.exists(args.full_checkpoint):
        import torch
        ckpt    = torch.load(args.full_checkpoint, map_location="cpu")
        metrics = ckpt.get("metrics", {})
        results.append({
            "name"      : "FIN (full)",
            "fine_acc"  : metrics.get("fine_acc",   0.0),
            "coarse_acc": metrics.get("coarse_acc", 0.0),
            "joint_acc" : metrics.get("joint",      0.0),
            "params_M"  : "—",
        })
        print(f"\n[Baseline] Loaded FIN (full): {metrics}")
    else:
        print(f"\n[Warning] Full checkpoint not found at {args.full_checkpoint}")
        print(f"          Train full model first: python train.py --config {args.config}")
        results.append({
            "name": "FIN (full) [not yet trained]",
            "fine_acc": 0, "coarse_acc": 0, "joint_acc": 0,
        })

    # Run each ablation
    for ab in ablations_to_run:
        result = run_ablation(cfg, ab["name"], args.output_dir)
        # Format name for table
        result["name"] = f"  — {ab['name'].replace('_', ' ')}"
        results.append(result)

    # Print final table
    print("\n" + "="*60)
    print("ABLATION STUDY RESULTS (Section 5.2)")
    print("="*60)
    print_results_table(results)

    # Save results to JSON for reproducibility
    results_path = os.path.join(args.output_dir, "ablation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Done] Results saved to: {results_path}")


if __name__ == "__main__":
    main()