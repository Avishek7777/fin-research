"""
experiments/baselines.py

Baseline Models for FIN Comparison (Section 5.1)
=================================================

Trains two baselines for the paper's main results table:

    1. ResNet-18 (fine only)
       Standard single-scale deep network, single global objective.
       The architectural foil for FIN — depth and width only, no
       hierarchical structure, no bandwidth constraint.

    2. ResNet-18 + auxiliary coarse head
       Same ResNet-18 but with a second classification head on the
       penultimate layer predicting coarse superclass labels.
       Closest flat-network approximation of FIN's multi-scale
       objective — has fine + coarse objectives but no bandwidth
       constraint, no self-similarity, no bidirectional feedback.

Usage:
    python experiments/baselines.py --config configs/fin_cifar100.yaml
    python experiments/baselines.py --config configs/fin_cifar100.yaml \
                                    --model resnet_aux

Both models use identical training setup to FIN-v2 for fair comparison:
same optimizer, LR schedule, augmentation, batch size, and epochs.
"""

import os
import sys
import yaml
import argparse
import time
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as T
from torchvision.models import resnet18
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train import (
    set_seed, build_dataloaders, get_coarse_labels,
    save_checkpoint, CIFAR100_COARSE_LABELS
)


# =============================================================================
# ResNet-18 Baseline (Fine Only)
# =============================================================================

class ResNet18Fine(nn.Module):
    """
    Standard ResNet-18 for CIFAR-100 fine classification.
    Single global objective — the architectural foil for FIN.

    Modifications from ImageNet ResNet-18:
      - First conv: 3x3, stride 1, no maxpool (standard CIFAR adaptation)
      - Final FC: 512 -> 100
    """

    def __init__(self, num_classes: int = 100):
        super().__init__()
        self.backbone = resnet18(weights=None)

        # CIFAR adaptation — replace 7x7 stride-2 conv + maxpool
        # with 3x3 stride-1 conv (images are 32x32, not 224x224)
        self.backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Linear(512, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        fine_labels: torch.Tensor,
        coarse_labels: torch.Tensor,
    ) -> dict:
        logits = self.backbone(x)
        loss   = F.cross_entropy(logits, fine_labels)
        acc    = (logits.argmax(1) == fine_labels).float().mean().item() * 100

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
# ResNet-18 + Auxiliary Coarse Head
# =============================================================================

class ResNet18Aux(nn.Module):
    """
    ResNet-18 with auxiliary coarse classification head.

    Architecture:
        Shared backbone (ResNet-18 up to penultimate layer)
        Fine head:   512 -> 100  (fine classification)
        Coarse head: 512 -> 20   (coarse classification, auxiliary)

    Loss: lambda_fine * CE(fine) + lambda_coarse * CE(coarse)
    with lambda_fine=1.0, lambda_coarse=0.5 matching FIN-v2's weights.

    This is the strongest flat baseline: same multi-scale objective
    as FIN-v2 but no bandwidth constraint, no self-similarity condition,
    no bidirectional message passing. Isolates the contribution of
    FIN's structural properties vs simply having two objectives.
    """

    def __init__(
        self,
        num_fine  : int   = 100,
        num_coarse: int   = 20,
        lambda_fine  : float = 1.0,
        lambda_coarse: float = 0.5,
    ):
        super().__init__()

        self.lambda_fine   = lambda_fine
        self.lambda_coarse = lambda_coarse

        # Shared backbone — extract features before final FC
        backbone = resnet18(weights=None)
        backbone.conv1   = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        backbone.maxpool = nn.Identity()

        # Remove final FC — we add our own heads
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.pool     = nn.AdaptiveAvgPool2d(1)

        # Task-specific heads
        self.fine_head   = nn.Linear(512, num_fine)
        self.coarse_head = nn.Linear(512, num_coarse)

    def forward(
        self,
        x            : torch.Tensor,
        fine_labels  : torch.Tensor,
        coarse_labels: torch.Tensor,
    ) -> dict:
        # Shared representation
        feat = self.features(x)
        feat = self.pool(feat).flatten(1)   # (B, 512)

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
    model     : nn.Module,
    cfg       : dict,
    model_name: str,
):
    """
    Train a baseline model using identical setup to FIN-v2.
    Same optimizer, LR schedule, augmentation, and epochs for fair comparison.
    """
    set_seed(cfg["experiment"]["seed"])

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

    # Data — identical to FIN-v2
    train_loader, val_loader = build_dataloaders(cfg["data"])

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
    ckpt_dir = os.path.join("checkpoints", model_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(
        log_dir=os.path.join(cfg["experiment"]["log_dir"], model_name)
    )

    best_joint = 0.0
    grad_clip  = cfg["training"]["grad_clip"]

    print(f"[{model_name}] Starting training: {total_epochs} epochs\n")

    for epoch in range(total_epochs):
        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        train_metrics = {"loss": 0, "fine_acc": 0, "coarse_acc": 0, "joint": 0}
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d} [train]", leave=False)
        for x, fine_labels in pbar:
            x             = x.to(device, non_blocking=True)
            fine_labels   = fine_labels.to(device, non_blocking=True)
            coarse_labels = get_coarse_labels(fine_labels).to(device)

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
        if epoch % cfg["training"]["eval_every"] == 0 or \
           epoch == total_epochs - 1:
            model.eval()
            val_metrics = {"loss": 0, "fine_acc": 0, "coarse_acc": 0, "joint": 0}
            n_val = 0

            with torch.no_grad():
                for x, fine_labels in val_loader:
                    x             = x.to(device)
                    fine_labels   = fine_labels.to(device)
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
                save_checkpoint(
                    model, optimizer, scheduler, epoch, val_metrics,
                    os.path.join(ckpt_dir, "best.pt"),
                )
                print(f"  [Ckpt] New best saved (joint={joint:.2f})")

    writer.close()
    print(f"\n[{model_name}] Done. Best joint: {best_joint:.2f}")
    return best_joint


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train FIN baselines")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--model", type=str, default="both",
        choices=["resnet_fine", "resnet_aux", "both"],
        help="Which baseline to train"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    results = {}

    if args.model in ("resnet_fine", "both"):
        print("\n" + "="*60)
        print("Baseline 1: ResNet-18 (Fine Only)")
        print("="*60)
        model = ResNet18Fine(num_classes=cfg["data"]["num_fine_classes"])
        joint = train_baseline(model, cfg, "resnet_fine")
        results["resnet_fine"] = joint

    if args.model in ("resnet_aux", "both"):
        print("\n" + "="*60)
        print("Baseline 2: ResNet-18 + Auxiliary Coarse Head")
        print("="*60)
        model = ResNet18Aux(
            num_fine   = cfg["data"]["num_fine_classes"],
            num_coarse = cfg["data"]["num_coarse_classes"],
            lambda_fine   = cfg["loss"]["lambda_1"],
            lambda_coarse = cfg["loss"]["lambda_2"],
        )
        joint = train_baseline(model, cfg, "resnet_aux")
        results["resnet_aux"] = joint

    print("\n" + "="*60)
    print("BASELINE RESULTS SUMMARY")
    print("="*60)
    for name, joint in results.items():
        print(f"  {name:<20}: joint={joint:.2f}")
    print()
    print("Compare against:")
    print("  FIN-v2 (full)      : joint=106.80, CKA=0.625, 3.82M params")
    print("  FIN-v2 no bandwidth: joint=115.75, CKA=0.622")


if __name__ == "__main__":
    main()