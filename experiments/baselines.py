"""
experiments/baselines.py

Baseline Models for FIN Comparison (Section 5.1)
=================================================

Trains two baselines for the paper's main results table:

    1. MobileNetV2 (fine only)
       Standard single-scale deep network, single global objective.
       The architectural foil for FIN — depth and width only, no
       hierarchical structure, no bandwidth constraint.
       ~3.5M params (vs FIN's 3.82M params)

    2. MobileNetV2 + auxiliary coarse head
       Same MobileNetV2 but with a second classification head on the
       penultimate layer predicting coarse superclass labels.
       Closest flat-network approximation of FIN's multi-scale
       objective — has fine + coarse objectives but no bandwidth
       constraint, no self-similarity, no bidirectional feedback.

Usage:
    # CIFAR-100 only (default)
    python experiments/baselines.py --config configs/fin_cifar100.yaml

    # CIFAR-10 only
    python experiments/baselines.py --config configs/fin_cifar100.yaml --dataset cifar10

    # Both datasets (train CIFAR-100 first, then CIFAR-10)
    python experiments/baselines.py --config configs/fin_cifar100.yaml --dataset both

    # Multiple seeds
    python experiments/baselines.py --config configs/fin_cifar100.yaml --seeds 1,2,3,4,5

    # Specific model variant
    python experiments/baselines.py --config configs/fin_cifar100.yaml \
                                    --model mobilenet_fine

Both models use identical training setup to FIN-v2 for fair comparison:
same optimizer, LR schedule, augmentation, batch size, and epochs.
"""

import os
import sys
import yaml
import argparse
import time
import math
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as T
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train import (
    set_seed, get_coarse_labels,
    save_checkpoint, CIFAR100_COARSE_LABELS
)


# =============================================================================
# CIFAR DataLoaders (supports both CIFAR-10 and CIFAR-100)
# =============================================================================

CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD  = [0.2470, 0.2435, 0.2616]
CIFAR100_MEAN = [0.5071, 0.4867, 0.4408]
CIFAR100_STD  = [0.2675, 0.2565, 0.2761]


def build_dataloaders(
    data_cfg    : dict,
    dataset_name: str = "cifar100",
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation dataloaders for CIFAR-10 or CIFAR-100.

    Args:
        data_cfg: Configuration dictionary with batch_size, num_workers, etc.
        dataset_name: 'cifar10' or 'cifar100'

    Returns:
        train_loader, val_loader
    """
    is_cifar10 = dataset_name.lower() == "cifar10"
    
    # Normalization values
    if is_cifar10:
        mean, std = CIFAR10_MEAN, CIFAR10_STD
        DatasetClass = torchvision.datasets.CIFAR10
    else:
        mean, std = CIFAR100_MEAN, CIFAR100_STD
        DatasetClass = torchvision.datasets.CIFAR100

    # Standard augmentation for training
    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    val_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    # Check if dataset already exists before setting download=True
    # This avoids unnecessary validation of 50k+ files on every run
    def dataset_exists(root, dataset_name, train=True):
        """Check if dataset files already exist locally."""
        if dataset_name == "CIFAR10":
            data_dir = os.path.join(root, "cifar-10-batches-py")
        else:  # CIFAR100
            data_dir = os.path.join(root, "cifar-100-python")
        
        if train:
            # Check for training files
            if dataset_name == "CIFAR10":
                train_files = [os.path.join(data_dir, f"data_batch_{i}") for i in range(1, 6)]
                train_files.append(os.path.join(data_dir, "meta"))
                return all(os.path.exists(f) for f in train_files)
            else:
                return os.path.exists(os.path.join(data_dir, "train"))
        else:
            # Check for test files
            if dataset_name == "CIFAR10":
                return os.path.exists(os.path.join(data_dir, "test_batch"))
            else:
                return os.path.exists(os.path.join(data_dir, "test"))

    root = data_cfg.get("root", "./data")
    train_exists = dataset_exists(root, dataset_name, train=True)
    val_exists = dataset_exists(root, dataset_name, train=False)

    train_dataset = DatasetClass(
        root     = root,
        train    = True,
        download = not train_exists,  # Only download if missing
        transform = train_transform,
    )

    val_dataset = DatasetClass(
        root     = root,
        train    = False,
        download = not val_exists,  # Only download if missing
        transform = val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size  = data_cfg["batch_size"],
        shuffle     = True,
        num_workers = data_cfg.get("num_workers", 4),
        pin_memory  = data_cfg.get("pin_memory", True),
        drop_last   = True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size  = data_cfg["batch_size"] * 2,
        shuffle     = False,
        num_workers = data_cfg.get("num_workers", 4),
        pin_memory  = data_cfg.get("pin_memory", True),
    )

    return train_loader, val_loader


def get_cifar10_coarse_labels(fine_labels: torch.Tensor) -> torch.Tensor:
    """
    For CIFAR-10, coarse labels equal fine labels (no hierarchy).
    """
    return fine_labels.clone()


# =============================================================================
# MobileNetV2 Baseline (Fine Only)
# =============================================================================

class MobileNetV2Fine(nn.Module):
    """
    MobileNetV2 for CIFAR classification (fine only).
    Single global objective — the architectural foil for FIN.
    
    Adapted for CIFAR (32x32 images) by:
      - Modifying first conv layer from 3x3 stride-2 to maintain more spatial info
      - Removing the initial maxpool
      - Final classifier: 1280 -> num_classes
    
    ~3.5M params (vs FIN's 3.82M params)
    """

    def __init__(
        self,
        num_classes      : int = 100,
        num_coarse_classes: int = 20,
        is_cifar10       : bool = False,
    ):
        super().__init__()
        self.is_cifar10 = is_cifar10
        self.num_classes = num_classes
        
        # Load MobileNetV2 backbone
        self.backbone = mobilenet_v2(weights=None)
        
        # CIFAR adaptation: modify first conv to work better with 32x32 images
        # Original MobileNetV2 uses 3x3 conv stride-2 + then more layers
        # For 32x32 images, we use stride=1 to preserve spatial information
        # Replace the entire first block with a simpler conv
        from torchvision.ops.misc import ConvNormActivation
        self.backbone.features[0] = ConvNormActivation(
            3, 32, kernel_size=3, stride=1, padding=1, norm_layer=nn.BatchNorm2d
        )
        
        # Update classifier for final classification
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes),
        )

    def forward(
        self,
        x            : torch.Tensor,
        fine_labels  : torch.Tensor,
        coarse_labels: torch.Tensor,
    ) -> dict:
        logits = self.backbone(x)
        loss   = F.cross_entropy(logits, fine_labels)
        acc    = (logits.argmax(1) == fine_labels).float().mean().item() * 100

        if self.is_cifar10:
            # For CIFAR-10, fine_acc = coarse_acc = overall accuracy
            coarse_acc = acc
        else:
            # Derive coarse accuracy by mapping fine predictions to coarse
            fine_preds   = logits.argmax(1).cpu()
            coarse_preds = torch.tensor(
                [CIFAR100_COARSE_LABELS[p] for p in fine_preds.tolist()]
            ).to(coarse_labels.device)
            coarse_acc = (coarse_preds == coarse_labels).float().mean().item() * 100

        return {
            "loss"      : loss,
            "fine_acc"  : acc,
            "coarse_acc": coarse_acc,
            "joint"     : acc + coarse_acc,
        }

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# MobileNetV2 + Auxiliary Coarse Head
# =============================================================================

class MobileNetV2Aux(nn.Module):
    """
    MobileNetV2 with auxiliary coarse classification head.

    Architecture:
        Shared backbone (MobileNetV2 features + pooling)
        Fine head:   1280 -> num_fine_classes  (fine classification)
        Coarse head: 1280 -> num_coarse_classes (coarse classification, auxiliary)

    Loss: lambda_fine * CE(fine) + lambda_coarse * CE(coarse)
    with lambda_fine=1.0, lambda_coarse=0.5 matching FIN-v2's weights.

    This is the strongest flat baseline: same multi-scale objective
    as FIN-v2 but no bandwidth constraint, no self-similarity condition,
    no bidirectional message passing. Isolates the contribution of
    FIN's structural properties vs simply having two objectives.
    """

    def __init__(
        self,
        num_fine        : int   = 100,
        num_coarse      : int   = 20,
        lambda_fine     : float = 1.0,
        lambda_coarse   : float = 0.5,
        is_cifar10      : bool  = False,
    ):
        super().__init__()

        self.lambda_fine   = lambda_fine
        self.lambda_coarse = lambda_coarse
        self.is_cifar10    = is_cifar10

        # Shared backbone — extract features before final pooling
        backbone = mobilenet_v2(weights=None)
        
        # CIFAR adaptation: use stride=1 for first conv to preserve spatial info
        from torchvision.ops.misc import ConvNormActivation
        backbone.features[0] = ConvNormActivation(
            3, 32, kernel_size=3, stride=1, padding=1, norm_layer=nn.BatchNorm2d
        )
        
        self.features = backbone.features
        self.pool     = nn.AdaptiveAvgPool2d(1)

        # Task-specific heads
        self.fine_head   = nn.Linear(1280, num_fine)
        if is_cifar10:
            # For CIFAR-10, coarse head predicts same classes as fine head
            self.coarse_head = nn.Linear(1280, num_fine)
        else:
            self.coarse_head = nn.Linear(1280, num_coarse)

    def forward(
        self,
        x            : torch.Tensor,
        fine_labels  : torch.Tensor,
        coarse_labels: torch.Tensor,
    ) -> dict:
        # Shared representation
        feat = self.features(x)
        feat = self.pool(feat).flatten(1)   # (B, 1280)

        # Task heads
        fine_logits   = self.fine_head(feat)
        coarse_logits = self.coarse_head(feat)

        # Losses
        loss_fine   = F.cross_entropy(fine_logits,   fine_labels)
        loss_coarse = F.cross_entropy(coarse_logits, coarse_labels)
        total_loss  = self.lambda_fine * loss_fine + \
                      self.lambda_coarse * loss_coarse

        # Accuracies
        fine_acc   = (fine_logits.argmax(1)   == fine_labels).float().mean().item()   * 100
        coarse_acc = (coarse_logits.argmax(1) == coarse_labels).float().mean().item() * 100

        return {
            "loss"      : total_loss,
            "fine_acc"  : fine_acc,
            "coarse_acc": coarse_acc,
            "joint"     : fine_acc + coarse_acc,
        }

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# Training Loop
# =============================================================================

def train_baseline(
    model         : nn.Module,
    cfg           : dict,
    model_name    : str,
    dataset_name  : str = "cifar100",
    seed          : int = 1,
) -> dict:
    """
    Train a baseline model using identical setup to FIN-v2.
    Same optimizer, LR schedule, augmentation, and epochs for fair comparison.
    
    Returns:
        dict with best metrics {'fine_acc': float, 'coarse_acc': float, 'joint': float}
    """
    set_seed(seed)

    # Device
    if torch.cuda.is_available():
        try:
            torch.zeros(1).cuda()
            device = torch.device("cuda")
        except Exception:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    print(f"[{model_name}] Device: {device}")
    print(f"[{model_name}] Parameters: {model.param_count():,}")

    model = model.to(device)

    # Data
    train_loader, val_loader = build_dataloaders(cfg["data"], dataset_name)

    # Optimizer — identical to FIN-v2
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)

    optimizer = optim.AdamW([
        {"params": decay,    "weight_decay": cfg["training"]["weight_decay"]},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=cfg["training"]["lr"])

    # LR schedule — identical to FIN-v2
    total_epochs  = cfg["training"]["epochs"]
    warmup_epochs = cfg["training"]["warmup_epochs"]

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Logging
    ckpt_dir = os.path.join(
        "checkpoints", 
        f"{model_name}_{dataset_name}_seed{seed}"
    )
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(
        log_dir=os.path.join(cfg["experiment"]["log_dir"], 
                           f"{model_name}_{dataset_name}_seed{seed}")
    )

    best_joint = 0.0
    best_metrics = {"fine_acc": 0.0, "coarse_acc": 0.0, "joint": 0.0}
    grad_clip = cfg["training"]["grad_clip"]

    print(f"[{model_name}] Starting training: {total_epochs} epochs\n")

    for epoch in range(total_epochs):
        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        train_metrics = {"loss": 0.0, "fine_acc": 0.0, "coarse_acc": 0.0, "joint": 0.0}
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d} [train]", leave=False)
        for batch in pbar:
            # Handle both CIFAR-10 (single label) and CIFAR-100 (single label)
            x, fine_labels = batch
            x             = x.to(device, non_blocking=True)
            fine_labels   = fine_labels.to(device, non_blocking=True)
            
            if dataset_name.lower() == "cifar10":
                coarse_labels = get_cifar10_coarse_labels(fine_labels).to(device)
            else:
                coarse_labels = get_coarse_labels(fine_labels, dataset_name).to(device)

            out = model(x, fine_labels, coarse_labels)

            optimizer.zero_grad(set_to_none=True)
            out["loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            for k in train_metrics:
                val = out[k].item() if torch.is_tensor(out[k]) else out[k]
                train_metrics[k] += val
            n_batches += 1

            pbar.set_postfix({
                "loss"   : f"{out['loss'].item():.3f}",
                "fine"   : f"{out['fine_acc']:.1f}%",
                "coarse" : f"{out['coarse_acc']:.1f}%",
            })

        scheduler.step()

        # Average
        for k in train_metrics:
            train_metrics[k] /= n_batches

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d}/{total_epochs} "
            f"| loss={train_metrics['loss']:.4f} "
            f"| fine={train_metrics['fine_acc']:.1f}% "
            f"| coarse={train_metrics['coarse_acc']:.1f}% "
            f"| joint={train_metrics['joint']:.2f} "
            f"| lr={current_lr:.2e}"
        )

        for k, v in train_metrics.items():
            writer.add_scalar(f"{model_name}/train/{k}", v, epoch)

        # ── Validate ──────────────────────────────────────────────────────
        if epoch % cfg["training"].get("eval_every", 1) == 0 or \
           epoch == total_epochs - 1:
            model.eval()
            val_metrics = {"loss": 0.0, "fine_acc": 0.0, "coarse_acc": 0.0, "joint": 0.0}
            n_val = 0

            with torch.no_grad():
                for batch in val_loader:
                    x, fine_labels = batch
                    x             = x.to(device)
                    fine_labels   = fine_labels.to(device)
                    
                    if dataset_name.lower() == "cifar10":
                        coarse_labels = get_cifar10_coarse_labels(fine_labels).to(device)
                    else:
                        coarse_labels = get_coarse_labels(fine_labels).to(device)

                    out = model(x, fine_labels, coarse_labels)
                    for k in val_metrics:
                        val = out[k].item() if torch.is_tensor(out[k]) else out[k]
                        val_metrics[k] += val
                    n_val += 1

            for k in val_metrics:
                val_metrics[k] /= n_val

            joint = val_metrics["joint"]
            print(
                f"  [Val] fine={val_metrics['fine_acc']:.2f}% "
                f"| coarse={val_metrics['coarse_acc']:.2f}% "
                f"| joint={joint:.2f} "
                f"| best={best_joint:.2f}"
            )

            for k, v in val_metrics.items():
                writer.add_scalar(f"{model_name}/val/{k}", v, epoch)

            if joint > best_joint:
                best_joint = joint
                best_metrics = {k: val_metrics[k] for k in ["fine_acc", "coarse_acc", "joint"]}
                save_checkpoint(
                    model, optimizer, scheduler, epoch, val_metrics,
                    os.path.join(ckpt_dir, "best.pt"),
                )
                print(f"  [Ckpt] New best saved (joint={joint:.2f})")

    writer.close()
    print(f"\n[{model_name}] Done. Best joint: {best_joint:.2f}")
    
    # Clear GPU memory before returning to avoid fragmentation
    if torch.cuda.is_available():
        del model, optimizer, scheduler
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    return best_metrics


def run_model_training(
    model_class,
    model_kwargs    : dict,
    cfg             : dict,
    model_name      : str,
    dataset_name    : str,
    seeds           : List[int],
) -> dict:
    """
    Run training for a single model across multiple seeds.
    
    Returns:
        dict with per-seed results and aggregate statistics
    """
    all_results = []
    
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Training {model_name} on {dataset_name} | Seed {seed}/{len(seeds)}")
        print(f"{'='*60}")
        
        # Create fresh model for each seed
        model = model_class(**model_kwargs)
        
        metrics = train_baseline(
            model, cfg, model_name, dataset_name, seed
        )
        metrics["seed"] = seed
        all_results.append(metrics)
    
    # Compute aggregate statistics
    fine_accs   = [r["fine_acc"]   for r in all_results]
    coarse_accs = [r["coarse_acc"] for r in all_results]
    joints      = [r["joint"]      for r in all_results]
    
    agg_results = {
        "per_seed": all_results,
        "mean_fine_acc"   : sum(fine_accs)   / len(fine_accs),
        "std_fine_acc"    : _std(fine_accs),
        "mean_coarse_acc" : sum(coarse_accs) / len(coarse_accs),
        "std_coarse_acc"  : _std(coarse_accs),
        "mean_joint"      : sum(joints)       / len(joints),
        "std_joint"       : _std(joints),
    }
    
    return agg_results


def _std(values: List[float]) -> float:
    """Compute sample standard deviation."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    return math.sqrt(variance)


def print_summary_table(
    results   : dict,
    dataset   : str,
    seeds     : List[int],
):
    """Print formatted summary table of results."""
    n_seeds = len(seeds)
    
    print(f"\n{'='*80}")
    print(f"RESULTS SUMMARY - {dataset.upper()} ({n_seeds} seeds: {seeds})")
    print(f"{'='*80}")
    print(f"{'Model':<20} {'Fine Acc':<15} {'Coarse Acc':<15} {'Joint':<15}")
    print(f"{'-'*80}")
    
    for model_name, model_results in results.items():
        per_seed = model_results["per_seed"]
        
        # Print per-seed results
        for r in per_seed:
            seed_str = str(r['seed'])
            print(
                f"{model_name}_s{seed_str:<15} "
                f"{r['fine_acc']:>6.2f}%          "
                f"{r['coarse_acc']:>6.2f}%          "
                f"{r['joint']:>6.2f}"
            )
        
        # Print aggregate (mean ± std)
        print(
            f"{model_name + ' (mean±std)':<20} "
            f"{model_results['mean_fine_acc']:>6.2f}±{model_results['std_fine_acc']:.2f}     "
            f"{model_results['mean_coarse_acc']:>6.2f}±{model_results['std_coarse_acc']:.2f}     "
            f"{model_results['mean_joint']:>6.2f}±{model_results['std_joint']:.2f}"
        )
        print(f"{'-'*80}")
    
    print("\nNote: For CIFAR-10, fine_acc = coarse_acc = overall accuracy")
    print(f"\nParameter counts:")
    print(f"  MobileNetV2-Fine:  ~2.35M params")
    print(f"  MobileNetV2-Aux:   ~2.35M + aux head params")
    print(f"  FIN-v2:            3.82M params")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train FIN baselines (MobileNetV2)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--model", type=str, default="both",
        choices=["mobilenet_fine", "mobilenet_aux", "both"],
        help="Which baseline to train"
    )
    parser.add_argument(
        "--dataset", type=str, default="cifar100",
        choices=["cifar10", "cifar100", "both"],
        help="Which dataset to train on"
    )
    parser.add_argument(
        "--seeds", type=str, default="1,2,3",
        help="Comma-separated list of random seeds (default: 1,2,3)"
    )
    args = parser.parse_args()

    # Parse seeds
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    
    # Parse datasets
    if args.dataset == "both":
        datasets = ["cifar100", "cifar10"]
    else:
        datasets = [args.dataset]

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Determine class counts based on dataset
    all_results = {}

    for dataset_name in datasets:
        print(f"\n{'#'*80}")
        print(f"# DATASET: {dataset_name.upper()}")
        print(f"{'#'*80}")
        
        is_cifar10 = dataset_name.lower() == "cifar10"
        
        # CIFAR-10: num_fine = num_coarse = 10
        # CIFAR-100: num_fine = 100, num_coarse = 20
        if is_cifar10:
            num_fine = num_coarse = 10
        else:
            num_fine = cfg["data"]["num_fine_classes"]
            num_coarse = cfg["data"]["num_coarse_classes"]
        
        dataset_results = {}
        
        if args.model in ("mobilenet_fine", "both"):
            print("\n" + "="*60)
            print(f"Baseline 1: MobileNetV2 (Fine Only) [{dataset_name}]")
            print("="*60)
            model_kwargs = {
                "num_classes"        : num_fine,
                "num_coarse_classes" : num_coarse,
                "is_cifar10"         : is_cifar10,
            }
            results = run_model_training(
                MobileNetV2Fine, model_kwargs, cfg,
                "mobilenet_fine", dataset_name, seeds
            )
            dataset_results["mobilenet_fine"] = results

        if args.model in ("mobilenet_aux", "both"):
            print("\n" + "="*60)
            print(f"Baseline 2: MobileNetV2 + Auxiliary Coarse Head [{dataset_name}]")
            print("="*60)
            model_kwargs = {
                "num_fine"        : num_fine,
                "num_coarse"      : num_coarse,
                "lambda_fine"     : cfg["loss"]["lambda_1"],
                "lambda_coarse"   : cfg["loss"]["lambda_2"],
                "is_cifar10"      : is_cifar10,
            }
            results = run_model_training(
                MobileNetV2Aux, model_kwargs, cfg,
                "mobilenet_aux", dataset_name, seeds
            )
            dataset_results["mobilenet_aux"] = results
        
        # Print summary for this dataset
        print_summary_table(dataset_results, dataset_name, seeds)
        
        # Store for overall summary
        all_results[dataset_name] = dataset_results

    # Final summary
    print(f"\n{'='*80}")
    print("FINAL BASELINE COMPARISON")
    print("="*80)
    
    for dataset_name, dataset_results in all_results.items():
        print(f"\n--- {dataset_name.upper()} ---")
        for model_name, results in dataset_results.items():
            print(
                f"  {model_name:<20}: "
                f"Fine={results['mean_fine_acc']:.2f}±{results['std_fine_acc']:.2f}%  "
                f"Coarse={results['mean_coarse_acc']:.2f}±{results['std_coarse_acc']:.2f}%  "
                f"Joint={results['mean_joint']:.2f}±{results['std_joint']:.2f}"
            )
    
    print("\n" + "="*80)
    print("REFERENCE (from paper):")
    print("="*80)
    print("  FIN-v2 (full)      : joint=106.80, CKA=0.625, 3.82M params")
    print("  FIN-v2 no bandwidth: joint=115.75, CKA=0.622")
    print("\n  MobileNetV2 baselines are ~2.35M params (comparable to FIN's 3.82M)")


if __name__ == "__main__":
    main()
