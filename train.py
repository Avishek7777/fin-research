"""
train.py

FIN Training Script — CIFAR-10/CIFAR-100
========================================

Usage:
    python train.py --config configs/fin_cifar100.yaml
    python train.py --config configs/fin_cifar100.yaml --dataset cifar10
    python train.py --config configs/fin_cifar100.yaml --seeds "1,2,3"
    python train.py --config configs/fin_cifar100.yaml --dataset both --seeds "1,2,3"
    python train.py --config configs/fin_cifar100.yaml --ablation no_bandwidth

What this script does:
    1. Loads and validates config (with optional ablation overrides)
    2. Builds dataloaders for CIFAR-10, CIFAR-100, or both sequentially
    3. Builds FIN model, optimizer, and LR scheduler
    4. Runs multi-seed training (default: 3 seeds)
    5. Runs training loop with:
         - Beta annealing (once per epoch)
         - Per-batch forward/backward/step
         - Gradient clipping
         - TensorBoard logging (every batch)
         - Epoch-level metric averaging and console output
    6. Evaluates on validation set every eval_every epochs
    7. Saves best checkpoint per seed (by joint fine+coarse accuracy)
    8. Computes and prints summary statistics across seeds

CIFAR-100 label handling:
    CIFAR-100 provides both fine labels (0-99) and coarse labels (0-19).
    torchvision's CIFAR100 dataset stores:
        target        = fine label
        targets_coarse is NOT provided directly — we derive it via
        a fixed fine->coarse mapping (CIFAR100_COARSE_LABELS below).
    This mapping is the official PyTorch/CIFAR-100 superclass assignment.

CIFAR-10 label handling:
    CIFAR-10 provides only 10 fine classes (0-9).
    For consistency, we set num_fine_classes=10, num_coarse_classes=10
    (no hierarchy — fine and coarse are equivalent).
"""

import os
import sys
import json
import math
import time
import yaml
import argparse
import random
from copy import deepcopy
from typing import Optional, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as T
from tqdm import tqdm

from fin.network.fin import build_fin, FIN
from fin.losses.hierarchical import MetricsTracker


# =============================================================================
# CIFAR-100 Fine -> Coarse Label Mapping
# =============================================================================
# Official mapping: fine class index -> coarse superclass index
# Source: https://www.cs.toronto.edu/~kriz/cifar.html
# 20 superclasses, each containing 5 fine classes.

CIFAR100_COARSE_LABELS = [
    4,  1,  14, 8,  0,  6,  7,  7,  18, 3,   # 0-9
    3,  14, 9,  18, 7,  11, 3,  9,  7,  11,  # 10-19
    6,  11, 5,  10, 7,  6,  13, 15, 3,  15,  # 20-29
    0,  11, 1,  10, 12, 14, 16, 9,  11, 5,   # 30-39
    5,  19, 8,  8,  15, 13, 14, 17, 18, 10,  # 40-49
    16, 4,  17, 4,  2,  0,  17, 4,  18, 17,  # 50-59
    10, 3,  2,  12, 12, 16, 12, 1,  9,  19,  # 60-69
    2,  10, 0,  1,  16, 12, 9,  13, 15, 13,  # 70-79
    16, 19, 2,  4,  6,  19, 5,  5,  8,  19,  # 80-89
    18, 1,  2,  15, 6,  0,  17, 8,  14, 13,  # 90-99
]

COARSE_LABEL_TENSOR = torch.tensor(CIFAR100_COARSE_LABELS, dtype=torch.long)


# =============================================================================
# Dataset Normalization Constants
# =============================================================================
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD  = [0.2470, 0.2435, 0.2616]

CIFAR100_MEAN = [0.5071, 0.4867, 0.4408]
CIFAR100_STD  = [0.2675, 0.2565, 0.2761]


# =============================================================================
# Utilities
# =============================================================================

def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def apply_ablation(cfg: dict, ablation: Optional[str]) -> dict:
    """
    Apply ablation overrides to the config.
    Maps ablation name to the override dict defined in config.ablations.

    Args:
        cfg     : full config dict
        ablation: ablation name (e.g. "no_bandwidth") or None

    Returns:
        Modified config dict (deep copy — original untouched)
    """
    if ablation is None:
        return cfg

    ablations = cfg.get("ablations", {})
    if ablation not in ablations:
        raise ValueError(
            f"Unknown ablation '{ablation}'. "
            f"Available: {list(ablations.keys())}"
        )

    cfg = deepcopy(cfg)
    overrides = ablations[ablation]

    for key_path, value in overrides.items():
        # key_path format: "section.subsection.key" e.g. "bandwidth.channel_01.gamma"
        keys  = key_path.split(".")
        node  = cfg
        for k in keys[:-1]:
            node = node[k]
        node[keys[-1]] = value

    print(f"[Ablation] Applied '{ablation}': {overrides}")
    return cfg


def parse_seeds(seeds_str: str) -> List[int]:
    """Parse comma-separated seed string into list of integers."""
    return [int(s.strip()) for s in seeds_str.split(",") if s.strip()]


# =============================================================================
# Dataset
# =============================================================================

def build_dataloaders(data_cfg: dict, dataset_name: str = "cifar100") -> tuple:
    """
    Build train and validation dataloaders for CIFAR-10 or CIFAR-100.

    Args:
        data_cfg     : data configuration dict from config file
        dataset_name : "cifar10" or "cifar100"

    Returns:
        train_loader, val_loader
    """
    # Select normalization constants based on dataset
    if dataset_name == "cifar10":
        mean, std = CIFAR10_MEAN, CIFAR10_STD
    else:  # cifar100
        mean, std = CIFAR100_MEAN, CIFAR100_STD

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

    # Build datasets based on dataset_name
    root = data_cfg["root"]
    
    # Check if dataset already exists before setting download=True
    # This avoids unnecessary validation of 50k+ files on every run
    def dataset_exists(root, dataset_name, train=True):
        """Check if dataset files already exist locally."""
        if dataset_name == "cifar10":
            data_dir = os.path.join(root, "cifar-10-batches-py")
        else:  # cifar100
            data_dir = os.path.join(root, "cifar-100-python")
        
        if train:
            # Check for training files
            train_files = [os.path.join(data_dir, f"data_batch_{i}") for i in range(1, 6)]
            train_files.append(os.path.join(data_dir, "meta"))
            return all(os.path.exists(f) for f in train_files) if dataset_name == "cifar10" else os.path.exists(os.path.join(data_dir, "train"))
        else:
            # Check for test files
            if dataset_name == "cifar10":
                return os.path.exists(os.path.join(data_dir, "test_batch"))
            else:
                return os.path.exists(os.path.join(data_dir, "test"))

    if dataset_name == "cifar10":
        train_exists = dataset_exists(root, "cifar10", train=True)
        val_exists = dataset_exists(root, "cifar10", train=False)
        
        train_dataset = torchvision.datasets.CIFAR10(
            root      = root,
            train     = True,
            download  = not train_exists,  # Only download if missing
            transform = train_transform,
        )
        val_dataset = torchvision.datasets.CIFAR10(
            root      = root,
            train     = False,
            download  = not val_exists,  # Only download if missing
            transform = val_transform,
        )
    else:  # cifar100
        train_exists = dataset_exists(root, "cifar100", train=True)
        val_exists = dataset_exists(root, "cifar100", train=False)
        
        train_dataset = torchvision.datasets.CIFAR100(
            root      = root,
            train     = True,
            download  = not train_exists,  # Only download if missing
            transform = train_transform,
        )
        val_dataset = torchvision.datasets.CIFAR100(
            root      = root,
            train     = False,
            download  = not val_exists,  # Only download if missing
            transform = val_transform,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size  = data_cfg["batch_size"],
        shuffle     = True,
        num_workers = data_cfg["num_workers"],
        pin_memory  = data_cfg["pin_memory"],
        drop_last   = True,    # keeps batch sizes consistent
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size  = data_cfg["batch_size"] * 2,   # no grad -> bigger batches
        shuffle     = False,
        num_workers = data_cfg["num_workers"],
        pin_memory  = data_cfg["pin_memory"],
    )

    return train_loader, val_loader


# Global cache for coarse label tensors on device
_COARSE_LABEL_TENSOR_CACHE = {}


def get_coarse_labels(fine_labels: torch.Tensor, dataset_name: str = "cifar100") -> torch.Tensor:
    """
    Derive coarse labels from fine labels.
    
    Optimized to avoid repeated CPU transfers and device mismatches.
    Caches COARSE_LABEL_TENSOR on the same device as fine_labels.

    Args:
        fine_labels  : (B,) fine class indices (on any device)
        dataset_name  : "cifar10" or "cifar100"

    Returns:
        coarse_labels: (B,) superclass indices (on same device as fine_labels)
    """
    if dataset_name == "cifar10":
        # CIFAR-10: no hierarchy, coarse = fine
        return fine_labels
    else:  # cifar100
        # Get device from fine_labels
        device = fine_labels.device
        
        # Check if we have cached tensor on this device
        if device not in _COARSE_LABEL_TENSOR_CACHE:
            # Create and cache on device
            _COARSE_LABEL_TENSOR_CACHE[device] = COARSE_LABEL_TENSOR.to(device)
        
        # Index on device (no CPU transfer!)
        coarse_tensor = _COARSE_LABEL_TENSOR_CACHE[device]
        return coarse_tensor[fine_labels]


# =============================================================================
# Optimizer and Scheduler
# =============================================================================

def build_optimizer(model: FIN, training_cfg: dict) -> optim.Optimizer:
    """
    Build AdamW optimizer.

    We use different weight decay for:
      - Bias terms and LayerNorm parameters: no decay (standard practice)
      - Everything else: weight_decay from config
    """
    decay_params    = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = optim.AdamW([
        {"params": decay_params,    "weight_decay": training_cfg["weight_decay"]},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=training_cfg["lr"])

    return optimizer


def build_lr_scheduler(
    optimizer   : optim.Optimizer,
    training_cfg: dict,
) -> optim.lr_scheduler._LRScheduler:
    """
    Build cosine LR scheduler with linear warmup.

    Warmup: LR linearly increases from 0 -> base_lr over warmup_epochs.
    Cosine: LR decays from base_lr -> 0 over remaining epochs.
    """
    warmup_epochs = training_cfg["warmup_epochs"]
    total_epochs  = training_cfg["epochs"]

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return float(epoch + 1) / float(warmup_epochs)
        else:
            # Cosine decay
            progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =============================================================================
# Checkpoint
# =============================================================================

def save_checkpoint(
    model    : FIN,
    optimizer: optim.Optimizer,
    scheduler,
    epoch    : int,
    metrics  : dict,
    path     : str,
):
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch"       : epoch,
        "model_state" : model.state_dict(),
        "optim_state" : optimizer.state_dict(),
        "sched_state"  : scheduler.state_dict(),
        "metrics"     : metrics,
    }, path)


def load_checkpoint(
    path     : str,
    model    : FIN,
    optimizer: optim.Optimizer,
    scheduler,
    device   : torch.device,
) -> int:
    """
    Load checkpoint and return the epoch to resume from.

    Returns:
        start_epoch: epoch to resume training from
    """
    ckpt        = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optim_state"])
    scheduler.load_state_dict(ckpt["sched_state"])
    start_epoch = ckpt["epoch"] + 1
    print(f"[Resume] Loaded checkpoint from epoch {ckpt['epoch']}")
    print(f"[Resume] Previous metrics: {ckpt['metrics']}")
    return start_epoch


# =============================================================================
# Training Loop — One Epoch
# =============================================================================

def train_one_epoch(
    model       : FIN,
    loader      : DataLoader,
    optimizer   : optim.Optimizer,
    scaler      : GradScaler,
    writer      : SummaryWriter,
    epoch       : int,
    cfg         : dict,
    global_step : int,
    device      : torch.device,
    dataset_name: str = "cifar100",
) -> tuple:
    """
    Train for one epoch with mixed precision training.

    Returns:
        tracker (MetricsTracker): accumulated metrics for this epoch
        global_step (int)       : updated global step count
    """
    model.train()
    tracker   = MetricsTracker()
    grad_clip = cfg["training"]["grad_clip"]

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [train]", leave=False)

    for x, fine_labels in pbar:
        x            = x.to(device, non_blocking=True)
        fine_labels  = fine_labels.to(device, non_blocking=True)
        coarse_labels= get_coarse_labels(fine_labels, dataset_name).to(device)

        # ── Forward ──────────────────────────────────────────────────────
        optimizer.zero_grad(set_to_none=True)
        with autocast('cuda' if device.type == 'cuda' else 'cpu'):
            out = model(x, fine_labels, coarse_labels)

        # ── Backward ─────────────────────────────────────────────────────
        scaler.scale(out.total_loss).backward()

        # Gradient clipping — important for stability with hierarchical losses
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        # ── Logging ──────────────────────────────────────────────────────
        tracker.update(out.breakdown)

        # TensorBoard: log every batch
        bd = out.breakdown.to_dict()
        for key, val in bd.items():
            writer.add_scalar(f"train/batch/{key}", val, global_step)

        # Update progress bar with key metrics
        pbar.set_postfix({
            "loss"      : f"{out.total_loss.item():.3f}",
            "fine_acc"  : f"{out.breakdown.fine_acc:.1f}%",
            "coarse_acc": f"{out.breakdown.coarse_acc:.1f}%",
        })

        global_step += 1

    return tracker, global_step


# =============================================================================
# Validation Loop
# =============================================================================

@torch.no_grad()
def validate(
    model       : FIN,
    loader      : DataLoader,
    writer      : SummaryWriter,
    epoch       : int,
    device      : torch.device,
    dataset_name: str = "cifar100",
) -> dict:
    """
    Run validation and return averaged metrics.

    Uses stochastic forward (not deterministic) to match training conditions.
    For representation analysis, use encode_deterministic() in evaluate.py.
    """
    model.eval()
    tracker = MetricsTracker()

    for x, fine_labels in tqdm(loader, desc=f"Epoch {epoch:03d} [val]", leave=False):
        x             = x.to(device, non_blocking=True)
        fine_labels   = fine_labels.to(device, non_blocking=True)
        coarse_labels = get_coarse_labels(fine_labels, dataset_name).to(device)

        out = model(x, fine_labels, coarse_labels)
        tracker.update(out.breakdown)

    avg = tracker.average()

    # TensorBoard: log epoch-level val metrics
    for key, val in avg.items():
        writer.add_scalar(f"val/epoch/{key}", val, epoch)

    return avg


# =============================================================================
# Main Training Function (Single Seed/Dataset)
# =============================================================================

def train_single(
    cfg         : dict,
    dataset_name: str,
    seed        : int,
    resume_path : Optional[str] = None,
) -> Dict[str, Any]:
    """
    Full training run for a single seed and dataset.

    Args:
        cfg         : full config dict (after ablation overrides applied)
        dataset_name: "cifar10" or "cifar100"
        seed        : random seed for this run
        resume_path : path to checkpoint to resume from, or None

    Returns:
        results dict with metrics for this run
    """
    # ── Setup ────────────────────────────────────────────────────────────────
    set_seed(seed)

    # Probe CUDA before committing
    if torch.cuda.is_available() and cfg["experiment"]["device"] == "cuda":
        try:
            torch.zeros(1).cuda()
            device = torch.device("cuda")
        except Exception as e:
            print(f"[Setup] CUDA probe failed: {e}")
            print("[Setup] Falling back to CPU.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    print(f"[Setup] Device: {device}")

    # Create directories with seed and dataset in path
    base_name = cfg["experiment"]["name"]
    log_dir = os.path.join(cfg["experiment"]["log_dir"], f"{base_name}_{dataset_name}_seed{seed}")
    ckpt_dir = os.path.join(cfg["experiment"]["checkpoint_dir"], f"{base_name}_{dataset_name}_seed{seed}")
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)

    # ── Data ─────────────────────────────────────────────────────────────────
    print(f"[Data] Building {dataset_name.upper()} dataloaders...")
    train_loader, val_loader = build_dataloaders(cfg["data"], dataset_name)
    print(f"[Data] Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── Model ────────────────────────────────────────────────────────────────
    # Update num_classes based on dataset
    if dataset_name == "cifar10":
        cfg["data"]["num_fine_classes"] = 10
        cfg["data"]["num_coarse_classes"] = 10
    else:  # cifar100
        cfg["data"]["num_fine_classes"] = 100
        cfg["data"]["num_coarse_classes"] = 20

    print("[Model] Building FIN...")
    print(f"[Model] Dataset: {dataset_name} | Fine classes: {cfg['data']['num_fine_classes']} | Coarse classes: {cfg['data']['num_coarse_classes']}")
    
    model = build_fin(cfg).to(device)
    print(model.architecture_summary())

    pc = model.param_count()
    print(f"[Model] Total parameters: {pc['total']:,} ({pc['total_M']}M)")
    writer.add_text("model/architecture", model.architecture_summary())
    writer.add_text("model/param_count", str(pc))

    # ── Optimizer + Scheduler ────────────────────────────────────────────────
    optimizer = build_optimizer(model, cfg["training"])
    scheduler = build_lr_scheduler(optimizer, cfg["training"])

    # ── Mixed Precision Training ─────────────────────────────────────────────
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    print(f"[Setup] GradScaler enabled: {scaler.is_enabled()}")

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch = 0
    if resume_path:
        start_epoch = load_checkpoint(resume_path, model, optimizer, scheduler, device)

    # ── Training State ───────────────────────────────────────────────────────
    total_epochs  = cfg["training"]["epochs"]
    eval_every    = cfg["training"]["eval_every"]
    best_joint    = 0.0      # best fine_acc + coarse_acc (joint metric)
    global_step   = 0

    print(f"\n[Train] Dataset: {dataset_name} | Seed: {seed}")
    print(f"[Train] Starting training: epochs {start_epoch} -> {total_epochs}")
    print(f"[Train] Loss weights: lambda_0={cfg['loss']['lambda_0']} | "
          f"lambda_1={cfg['loss']['lambda_1']} | lambda_2={cfg['loss']['lambda_2']}")
    print(f"[Train] Bandwidth: beta_01={cfg['bandwidth']['channel_01']['beta']} | "
          f"beta_12={cfg['bandwidth']['channel_12']['beta']}")
    print(f"[Train] Annealing: {cfg['annealing']['strategy']} | "
          f"warmup={cfg['annealing']['warmup_epochs']} epochs\n")

    # ── Main Loop ────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, total_epochs):

        epoch_start = time.time()

        # 1. Update beta annealing for this epoch
        current_betas = model.update_betas(epoch)
        writer.add_scalar("annealing/beta_01", current_betas["channel_01"], epoch)
        writer.add_scalar("annealing/beta_12", current_betas["channel_12"], epoch)

        # 2. Train one epoch
        train_tracker, global_step = train_one_epoch(
            model, train_loader, optimizer, scaler, writer,
            epoch, cfg, global_step, device, dataset_name
        )

        # 3. LR scheduler step
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("train/lr", current_lr, epoch)

        # 4. Log epoch-level train metrics
        train_avg = train_tracker.average()
        for key, val in train_avg.items():
            writer.add_scalar(f"train/epoch/{key}", val, epoch)

        # 5. Console output
        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch:03d}/{total_epochs} "
            f"| loss={train_avg.get('total', 0):.4f} "
            f"| fine={train_avg.get('fine_acc', 0):.1f}% "
            f"| coarse={train_avg.get('coarse_acc', 0):.1f}% "
            f"| bw_01={train_avg.get('bw_01', 0):.4f} "
            f"| bw_12={train_avg.get('bw_12', 0):.4f} "
            f"| beta_01={current_betas['channel_01']:.3f} "
            f"| lr={current_lr:.2e} "
            f"| {epoch_time:.1f}s"
        )

        # 6. Validation
        if epoch % eval_every == 0 or epoch == total_epochs - 1:
            val_avg = validate(model, val_loader, writer, epoch, device, dataset_name)
            val_fine   = val_avg.get("fine_acc",   0.0)
            val_coarse = val_avg.get("coarse_acc", 0.0)
            joint      = val_fine + val_coarse

            print(
                f"  [Val] fine={val_fine:.2f}% | coarse={val_coarse:.2f}% "
                f"| joint={joint:.2f} | best_joint={best_joint:.2f}"
            )

            # 7. Save best checkpoint
            if cfg["training"]["save_best"] and joint > best_joint:
                best_joint = joint
                save_checkpoint(
                    model, optimizer, scheduler, epoch,
                    {"fine_acc": val_fine, "coarse_acc": val_coarse, "joint": joint},
                    os.path.join(ckpt_dir, f"best_seed{seed}.pt"),
                )
                print(f"  [Ckpt] New best saved (joint={joint:.2f})")

        # 8. Always save last checkpoint (for resuming)
        save_checkpoint(
            model, optimizer, scheduler, epoch,
            {"fine_acc": train_avg.get("fine_acc", 0), "step": global_step},
            os.path.join(ckpt_dir, f"last_seed{seed}.pt"),
        )

    # ── Done ─────────────────────────────────────────────────────────────────
    writer.close()
    print(f"\n[Done] Dataset: {dataset_name} | Seed: {seed}")
    print(f"[Done] Best joint accuracy: {best_joint:.2f}")
    print(f"[Done] Best checkpoint: {os.path.join(ckpt_dir, f'best_seed{seed}.pt')}")
    print(f"[Done] TensorBoard logs: {log_dir}")
    
    # Clear GPU memory before returning to avoid fragmentation
    if torch.cuda.is_available():
        del model, optimizer, scheduler
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Return results for this run
    return {
        "dataset": dataset_name,
        "seed": seed,
        "best_joint": best_joint,
        "ckpt_dir": ckpt_dir,
        "log_dir": log_dir,
    }


# =============================================================================
# Multi-Seed Training
# =============================================================================

def train_multi_seed(
    cfg     : dict,
    seeds   : List[int],
    dataset : str,
) -> List[Dict[str, Any]]:
    """
    Run training for multiple seeds.

    Args:
        cfg     : full config dict
        seeds   : list of random seeds
        dataset : "cifar10", "cifar100", or "both"

    Returns:
        List of results dicts for each run
    """
    all_results = []

    # Determine datasets to run
    if dataset == "both":
        datasets_to_run = ["cifar100", "cifar10"]
    else:
        datasets_to_run = [dataset]

    for ds in datasets_to_run:
        print(f"\n{'='*80}")
        print(f"Starting training for {ds.upper()} dataset")
        print(f"{'='*80}\n")

        for seed in seeds:
            print(f"\n{'='*60}")
            print(f"Starting training: Dataset={ds} | Seed={seed}")
            print(f"{'='*60}\n")

            result = train_single(cfg, ds, seed)
            all_results.append(result)

    return all_results


def compute_summary_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute summary statistics across all runs.

    Args:
        results: list of result dicts from train_single

    Returns:
        summary dict with mean ± std for each metric
    """
    # Group by dataset
    by_dataset = {}
    for r in results:
        ds = r["dataset"]
        if ds not in by_dataset:
            by_dataset[ds] = []
        by_dataset[ds].append(r)

    summary = {}

    for ds, runs in by_dataset.items():
        seeds = [r["seed"] for r in runs]
        joints = [r["best_joint"] for r in runs]

        mean_joint = np.mean(joints)
        std_joint = np.std(joints)

        summary[ds] = {
            "seeds": seeds,
            "joints": joints,
            "mean_joint": float(mean_joint),
            "std_joint": float(std_joint),
            "mean_joint_str": f"{mean_joint:.2f} ± {std_joint:.2f}",
        }

    return summary


def print_summary_table(results: List[Dict[str, Any]], summary: Dict[str, Any]):
    """
    Print formatted summary table of all runs.
    """
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)

    # Group by dataset
    by_dataset = {}
    for r in results:
        ds = r["dataset"]
        if ds not in by_dataset:
            by_dataset[ds] = []
        by_dataset[ds].append(r)

    for ds in ["cifar100", "cifar10"]:  # Order: cifar100 first, then cifar10
        if ds not in by_dataset:
            continue

        runs = by_dataset[ds]
        print(f"\n{ds.upper()}:")
        print("-" * 60)
        print(f"{'Seed':<10} {'Best Joint Acc':<20}")
        print("-" * 60)

        for r in runs:
            print(f"{r['seed']:<10} {r['best_joint']:.2f}%")

        # Print mean ± std
        s = summary[ds]
        print("-" * 60)
        print(f"{'Mean ± Std':<10} {s['mean_joint_str']:<20}")
        print(f"{'Seeds':<10} {', '.join(map(str, s['seeds']))}")

    print("\n" + "=" * 80)


def save_results_json(
    results: List[Dict[str, Any]],
    summary: Dict[str, Any],
    output_path: str,
    cfg: dict,
):
    """
    Save complete results to JSON file.
    """
    output_data = {
        "config": {
            "experiment_name": cfg["experiment"]["name"],
            "loss_weights": cfg["loss"],
            "bandwidth": cfg["bandwidth"],
            "architecture": cfg["architecture"],
        },
        "summary": summary,
        "all_runs": [
            {
                "dataset": r["dataset"],
                "seed": r["seed"],
                "best_joint": r["best_joint"],
                "ckpt_dir": r["ckpt_dir"],
                "log_dir": r["log_dir"],
            }
            for r in results
        ],
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"[Results] Saved to {output_path}")


# =============================================================================
# Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Fractal Intelligence Network")

    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file (e.g. configs/fin_cifar100.yaml)"
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        choices=["cifar10", "cifar100", "both"],
        help="Dataset to train on. Overrides config default if specified."
    )
    parser.add_argument(
        "--seeds", type=str, default="1,2,3",
        help="Comma-separated list of random seeds (default: '1,2,3')"
    )
    parser.add_argument(
        "--ablation", type=str, default=None,
        help="Ablation name from config.ablations (e.g. no_bandwidth)"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume training from"
    )

    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Apply ablation overrides if specified
    cfg = apply_ablation(cfg, args.ablation)

    # Parse seeds
    seeds = parse_seeds(args.seeds)
    print(f"[Setup] Seeds: {seeds}")

    # Determine dataset (command-line overrides config)
    dataset = args.dataset if args.dataset else cfg["data"].get("dataset", "cifar100")
    print(f"[Setup] Dataset: {dataset}")

    # Update experiment name to include ablation tag and dataset
    base_name = cfg["experiment"]["name"]
    if args.ablation:
        base_name += f"_{args.ablation}"
    if args.dataset:
        base_name += f"_{args.dataset}"
    cfg["experiment"]["name"] = base_name

    # Create base checkpoint and log directories
    base_ckpt_dir = cfg["experiment"]["checkpoint_dir"]
    base_log_dir = cfg["experiment"]["log_dir"]

    # Results file path
    results_path = os.path.join(base_log_dir, f"{base_name}_results.json")

    # Run multi-seed training
    results = train_multi_seed(cfg, seeds, dataset)

    # Compute and print summary statistics
    summary = compute_summary_stats(results)
    print_summary_table(results, summary)

    # Save results to JSON
    save_results_json(results, summary, results_path, cfg)

    # Print TensorBoard command
    print(f"\n[Done] Run TensorBoard: tensorboard --logdir {base_log_dir}")


if __name__ == "__main__":
    main()
