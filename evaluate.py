"""
evaluate.py

FIN Evaluation, Representation Analysis & Visualization
=======================================================

Usage:
    # Full evaluation on test set
    python evaluate.py --config configs/fin_cifar100.yaml \
                       --checkpoint checkpoints/best.pt

    # CIFAR-10 evaluation
    python evaluate.py --config configs/fin_cifar10.yaml \
                       --checkpoint checkpoints/best_cifar10.pt \
                       --dataset cifar10

    # Evaluate on both datasets
    python evaluate.py --config configs/fin_cifar100.yaml \
                       --checkpoint checkpoints/best.pt \
                       --dataset both

    # Multi-seed evaluation
    python evaluate.py --config configs/fin_cifar100.yaml \
                       --checkpoint_dir checkpoints \
                       --seeds 1,2,3,4,5

    # With t-SNE visualization
    python evaluate.py --config configs/fin_cifar100.yaml \
                       --checkpoint checkpoints/best.pt \
                       --tsne

    # Compare FIN vs ablation checkpoint
    python evaluate.py --config configs/fin_cifar100.yaml \
                       --checkpoint checkpoints/best.pt \
                       --compare checkpoints/no_bandwidth/best.pt \
                       --tsne

    # Efficiency analysis (Params vs Accuracy, FLOPs vs Accuracy)
    python evaluate.py --config configs/fin_cifar100.yaml \
                       --checkpoint_dir checkpoints \
                       --seeds 1,2,3 \
                       --efficiency_plots

What this script produces:
    1. Accuracy table (fine + coarse) for paper Section 5.1
    2. t-SNE plots of z0, z1, z2 — visual evidence for Theorem 1.1
    3. CKA similarity matrix between levels — quantitative evidence
       that each level learns a genuinely different representation
    4. Bandwidth usage report — how much of beta_k each channel used
    5. Alpha gate values — how much top-down influence each level learned
    6. Efficiency plots (Params/FLOPs vs Accuracy)
    7. Multi-seed analysis with mean ± std statistics

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
import json
import glob
import re
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch import nn
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
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from fin.network.fin import build_fin, FIN
from train import CIFAR100_COARSE_LABELS, COARSE_LABEL_TENSOR, get_coarse_labels


# =============================================================================
# CIFAR-10 and CIFAR-100 Class Names
# =============================================================================

CIFAR100_SUPERCLASSES = [
    "aquatic mammals", "fish", "flowers", "food containers",
    "fruit & vegetables", "household electrical devices", "household furniture",
    "insects", "large carnivores", "large man-made outdoor things",
    "large natural outdoor scenes", "large omnivores & herbivores",
    "medium-sized mammals", "non-insect invertebrates", "people",
    "reptiles", "small mammals", "trees", "vehicles 1", "vehicles 2",
]

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Normalization constants
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]
CIFAR100_MEAN = [0.5071, 0.4867, 0.4408]
CIFAR100_STD = [0.2675, 0.2565, 0.2761]


# =============================================================================
# Data Loading
# =============================================================================

def get_dataset_normalization(dataset: str) -> Tuple[List[float], List[float]]:
    """Get mean and std for dataset normalization."""
    if dataset == "cifar10":
        return CIFAR10_MEAN, CIFAR10_STD
    elif dataset == "cifar100":
        return CIFAR100_MEAN, CIFAR100_STD
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_num_classes(dataset: str) -> int:
    """Get number of classes for dataset."""
    if dataset == "cifar10":
        return 10
    elif dataset == "cifar100":
        return 100
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def load_dataset(
    dataset: str,
    data_root: str,
    batch_size: int,
    num_workers: int = 4,
) -> DataLoader:
    """Load CIFAR-10 or CIFAR-100 test set."""
    mean, std = get_dataset_normalization(dataset)
    
    val_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    
    # Check if dataset already exists before setting download=True
    # This avoids unnecessary validation of 50k+ files on every run
    def dataset_exists(root, dataset_name):
        """Check if test dataset files already exist locally."""
        if dataset_name == "cifar10":
            test_file = os.path.join(root, "cifar-10-batches-py", "test_batch")
        else:  # cifar100
            test_file = os.path.join(root, "cifar-100-python", "test")
        
        return os.path.exists(test_file)
    
    test_exists = dataset_exists(data_root, dataset)
    
    if dataset == "cifar10":
        val_dataset = torchvision.datasets.CIFAR10(
            root=data_root, train=False,
            download=not test_exists, transform=val_transform,  # Only download if missing
        )
    elif dataset == "cifar100":
        val_dataset = torchvision.datasets.CIFAR100(
            root=data_root, train=False,
            download=not test_exists, transform=val_transform,  # Only download if missing
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    return DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
    )


# =============================================================================
# FLOPs Counting
# =============================================================================

def count_flops(model: nn.Module, input_size: Tuple[int, int, int] = (1, 3, 32, 32)) -> Dict[str, float]:
    """
    Count FLOPs and parameters for the model.
    Tries thop library first, falls back to manual counting.
    
    Returns:
        dict with 'flops_M' (FLOPs in millions), 'params_M' (params in millions)
    """
    # Try thop first
    try:
        from thop import profile
        device = next(model.parameters()).device
        dummy_input = torch.randn(1, *input_size).to(device)
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        return {
            'flops_M': flops / 1e6,
            'params_M': params / 1e6,
        }
    except (ImportError, Exception) as e:
        # Fall back to manual counting
        pass
    
    # Manual FLOP counting
    total_flops = 0
    total_params = 0
    
    def count_conv2d_flops(module: nn.Conv2d, input_h: int, input_w: int) -> int:
        """Count FLOPs for Conv2d: 2 * Cin * Cout * K * K * H * W"""
        cin = module.in_channels
        cout = module.out_channels
        k = module.kernel_size[0] * module.kernel_size[1]
        h_out = (input_h + 2 * module.padding[0] - module.dilation[0] * (module.kernel_size[0] - 1) - 1) // module.stride[0] + 1
        w_out = (input_w + 2 * module.padding[1] - module.dilation[1] * (module.kernel_size[1] - 1) - 1) // module.stride[1] + 1
        return 2 * cin * cout * k * h_out * w_out
    
    def count_linear_flops(module: nn.Linear) -> int:
        """Count FLOPs for Linear: 2 * Cin * Cout"""
        return 2 * module.in_features * module.out_features
    
    def count_bn2d_flops(module: nn.BatchNorm2d, h: int, w: int) -> int:
        """Count FLOPs for BatchNorm2d: 4 * Cin * H * W"""
        return 4 * module.num_features * h * w
    
    def recursively_count_flops(module: nn.Module, input_shape: Tuple[int, ...]):
        nonlocal total_flops, total_params
        
        # Count parameters for this module
        if hasattr(module, 'weight') and module.weight is not None:
            total_params += module.weight.numel()
        if hasattr(module, 'bias') and module.bias is not None:
            total_params += module.bias.numel()
        
        # Guard against shapes without spatial dimensions
        if len(input_shape) < 3:
            # Shape is 1D or 2D (e.g., after Linear layer)
            # Recurse into children but don't try to count Conv/BN
            child_output_shape = input_shape
            for child in module.children():
                child_output_shape = recursively_count_flops(child, child_output_shape)
            return child_output_shape
        
        h, w = input_shape[1], input_shape[2]
        
        if isinstance(module, nn.Conv2d):
            if module.in_channels > 0:  # Skip uninitialized
                total_flops += count_conv2d_flops(module, h, w)
                # Compute output size for children
                h = (h + 2 * module.padding[0] - module.dilation[0] * (module.kernel_size[0] - 1) - 1) // module.stride[0] + 1
                w = (w + 2 * module.padding[1] - module.dilation[1] * (module.kernel_size[1] - 1) - 1) // module.stride[1] + 1
                return (module.out_channels, h, w)
        elif isinstance(module, nn.Linear):
            total_flops += count_linear_flops(module)
            return (module.out_features,)
        elif isinstance(module, nn.BatchNorm2d):
            total_flops += count_bn2d_flops(module, h, w)
        
        # Recurse into children
        child_output_shape = input_shape
        for child in module.children():
            child_output_shape = recursively_count_flops(child, child_output_shape)
        
        return child_output_shape if 'child_output_shape' in locals() else input_shape
    
    # Count for all submodules
    recursively_count_flops(model, (1, *input_size))
    
    return {
        'flops_M': total_flops / 1e6,
        'params_M': total_params / 1e6,
    }


# =============================================================================
# Load Model
# =============================================================================

def load_model(
    cfg: dict, 
    checkpoint_path: str, 
    device: torch.device,
    dataset: str = "cifar100"
):
    """Load model from checkpoint - supports FIN and baseline models."""
    from copy import deepcopy
    
    # Determine model type from checkpoint path
    checkpoint_name = os.path.basename(os.path.dirname(checkpoint_path))
    is_mobilenet_fine = "mobilenet_fine" in checkpoint_name
    is_mobilenet_aux = "mobilenet_aux" in checkpoint_name
    is_fin = not (is_mobilenet_fine or is_mobilenet_aux)
    
    # Load checkpoint first to inspect structure
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Extract state dict
    if "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    
    # Determine model type from state dict if path detection isn't clear
    has_fin_keys = any(k.startswith("level") for k in state_dict.keys())
    has_features_keys = any(k.startswith("features") for k in state_dict.keys())
    
    if has_features_keys:
        is_mobilenet_fine = "coarse_head" not in state_dict
        is_mobilenet_aux = "coarse_head" in state_dict
        is_fin = False
    elif has_fin_keys:
        is_fin = True
        is_mobilenet_fine = False
        is_mobilenet_aux = False
    
    # Load appropriate model type
    if is_mobilenet_fine:
        # Import baseline models
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "experiments"))
        from baselines import MobileNetV2Fine
        
        num_classes = get_num_classes(dataset)
        model = MobileNetV2Fine(num_classes=num_classes).to(device)
        print(f"[Load] Loaded MobileNetV2Fine")
        
    elif is_mobilenet_aux:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "experiments"))
        from baselines import MobileNetV2Aux
        
        num_classes = get_num_classes(dataset)
        model = MobileNetV2Aux(num_classes=num_classes).to(device)
        print(f"[Load] Loaded MobileNetV2Aux")
        
    else:  # FIN model
        model_cfg = deepcopy(cfg)
        
        # Update num_classes in data section
        if "data" not in model_cfg:
            model_cfg["data"] = {}
        
        if dataset == "cifar10":
            model_cfg["data"]["num_fine_classes"] = 10
            model_cfg["data"]["num_coarse_classes"] = 10
        else:  # cifar100
            model_cfg["data"]["num_fine_classes"] = 100
            model_cfg["data"]["num_coarse_classes"] = 20
        
        model = build_fin(model_cfg).to(device)
        print(f"[Load] Loaded FIN")
    
    # Load state dict
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[Load] Loaded checkpoint: {checkpoint_path}")
    if "metrics" in ckpt:
        print(f"[Load] Checkpoint metrics: {ckpt['metrics']}")
    return model


# =============================================================================
# Checkpoint Path Utilities
# =============================================================================

def find_checkpoint_paths(
    checkpoint_dir: str,
    checkpoint_file: Optional[str],
    model_prefix: str = "fin",
    seeds: List[int] = None
) -> Dict[int, str]:
    """
    Find checkpoint paths for given seeds.
    Supports both old format: checkpoints/best.pt
    And new format: checkpoints/fin_cifar100_seed1/best.pt
    
    Returns:
        dict mapping seed -> checkpoint_path
    """
    paths = {}
    
    if checkpoint_file:
        # Single checkpoint provided
        if os.path.isfile(checkpoint_file):
            paths[1] = checkpoint_file
        return paths
    
    if not checkpoint_dir or not os.path.exists(checkpoint_dir):
        return paths
    
    # If seeds provided, look for pattern: {prefix}_seed{n}/best.pt
    if seeds:
        for seed in seeds:
            # Try various patterns
            patterns = [
                # Actual train.py output: experiment name has dataset appended once in main(),
                # so FIN on cifar100 produces "fin_cifar100_cifar100_seed{n}"
                os.path.join(checkpoint_dir, f"{model_prefix}_cifar100_cifar100_seed{seed}", f"best_seed{seed}.pt"),
                os.path.join(checkpoint_dir, f"{model_prefix}_cifar10_cifar10_seed{seed}", f"best_seed{seed}.pt"),
                # Standard patterns (baselines and fallbacks)
                os.path.join(checkpoint_dir, f"{model_prefix}_cifar100_seed{seed}", "best.pt"),
                os.path.join(checkpoint_dir, f"{model_prefix}_cifar100_seed{seed}", f"best_seed{seed}.pt"),
                os.path.join(checkpoint_dir, f"{model_prefix}_cifar10_seed{seed}", "best.pt"),
                os.path.join(checkpoint_dir, f"{model_prefix}_cifar10_seed{seed}", f"best_seed{seed}.pt"),
                os.path.join(checkpoint_dir, f"{model_prefix}_seed{seed}", "best.pt"),
                os.path.join(checkpoint_dir, f"{model_prefix}_seed{seed}", f"best_seed{seed}.pt"),
                os.path.join(checkpoint_dir, f"seed{seed}", "best.pt"),
                os.path.join(checkpoint_dir, f"best_seed{seed}.pt"),
            ]
            for pattern in patterns:
                if os.path.exists(pattern):
                    paths[seed] = pattern
                    break
    else:
        # Auto-detect seeds from directory structure
        # Pattern: *_seed{n} or seed{n}
        dirs = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
        for d in dirs:
            match = re.search(r'seed(\d+)', d)
            if match:
                seed = int(match.group(1))
                # Try both best.pt and best_seed{n}.pt
                ckpt_path = os.path.join(checkpoint_dir, d, "best.pt")
                if not os.path.exists(ckpt_path):
                    ckpt_path = os.path.join(checkpoint_dir, d, f"best_seed{seed}.pt")
                if os.path.exists(ckpt_path):
                    paths[seed] = ckpt_path
        
        # Also check for best_seed{n}.pt pattern in root
        files = glob.glob(os.path.join(checkpoint_dir, "best_seed*.pt"))
        for f in files:
            match = re.search(r'seed(\d+)', f)
            if match:
                seed = int(match.group(1))
                if seed not in paths:  # Prefer directory version
                    paths[seed] = f
        
        # If no seed directories found, look for best.pt
        if not paths:
            best_path = os.path.join(checkpoint_dir, "best.pt")
            if os.path.exists(best_path):
                paths[1] = best_path
    
    return paths


# =============================================================================
# Extract Representations with Diagnostics
# =============================================================================

@torch.no_grad()
def extract_representations_with_diagnostics(
    model      : FIN,
    loader     : DataLoader,
    device     : torch.device,
    max_samples: int = 5000,
    dataset    : str = "cifar100",
) -> Dict:
    """
    Extract z0, z1, z2 representations and labels from the dataset.
    Also extract channel diagnostics for bandwidth analysis.
    
    Args:
        model      : FIN in eval mode
        loader     : DataLoader (shuffled=False for consistency)
        device     : compute device
        max_samples: cap at this many samples
        dataset    : "cifar10" or "cifar100"
    
    Returns:
        dict with keys: z0, z1, z2, fine_labels, coarse_labels, 
                       channel_kl, activation_stats
    """
    z0_list, z1_list, z2_list = [], [], []
    fine_list, coarse_list = [], []
    channel_kl_list = []
    activation_stats = defaultdict(list)
    total = 0
    
    # Enable diagnostics mode
    if hasattr(model, 'enable_diagnostics'):
        model.enable_diagnostics()
    
    for x, fine_labels in tqdm(loader, desc="Extracting representations"):
        if total >= max_samples:
            break
        
        x = x.to(device)
        fine_labels = fine_labels.to(device)
        coarse_labels = get_coarse_labels(fine_labels, dataset).to(device)
        
        # Forward pass with diagnostics
        if hasattr(model, 'encode_with_diagnostics'):
            z0, z1, z2, diagnostics = model.encode_with_diagnostics(x)
            
            # Collect KL divergences
            if 'channel_kl' in diagnostics:
                channel_kl_list.append(diagnostics['channel_kl'])
            
            # Collect activation statistics
            for level in ['z0', 'z1', 'z2']:
                if level in diagnostics:
                    acts = diagnostics[level]
                    activation_stats[f'{level}_mean'].append(acts.mean().item())
                    activation_stats[f'{level}_std'].append(acts.std().item())
                    activation_stats[f'{level}_sparsity'].append((acts.abs() < 0.01).float().mean().item())
        else:
            if hasattr(model, 'encode_deterministic'):
                z0, z1, z2 = model.encode_deterministic(x)
            else:
                feat = model.features(x).mean([2, 3])
                z0 = z1 = z2 = feat
        
        z0_list.append(z0.cpu().numpy())
        z1_list.append(z1.cpu().numpy())
        z2_list.append(z2.cpu().numpy())
        fine_list.append(fine_labels.cpu().numpy())
        coarse_list.append(coarse_labels.cpu().numpy())
        
        total += x.size(0)
    
    # Disable diagnostics
    if hasattr(model, 'disable_diagnostics'):
        model.disable_diagnostics()
    
    result = {
        "z0"           : np.concatenate(z0_list, axis=0)[:max_samples],
        "z1"           : np.concatenate(z1_list, axis=0)[:max_samples],
        "z2"           : np.concatenate(z2_list, axis=0)[:max_samples],
        "fine_labels"  : np.concatenate(fine_list, axis=0)[:max_samples],
        "coarse_labels": np.concatenate(coarse_list, axis=0)[:max_samples],
    }
    
    # Add channel diagnostics
    if channel_kl_list:
        result["channel_kl"] = torch.stack(channel_kl_list).cpu().numpy()
    
    # Add activation statistics
    if activation_stats:
        result["activation_stats"] = {
            k: np.mean(v) for k, v in activation_stats.items()
        }
    
    return result


@torch.no_grad()
def evaluate_accuracy_and_representations(
    model      : FIN,
    loader     : DataLoader,
    device     : torch.device,
    max_samples: int = 5000,
    dataset    : str = "cifar100",
) -> tuple:
    """
    Compute accuracy AND extract representations in a single forward pass.
    This eliminates the redundant second forward pass.
    
    Returns:
        tuple: (accuracies_dict, representations_dict)
    """
    model.eval()
    fine_correct = 0
    coarse_correct = 0
    total = 0
    
    z0_list, z1_list, z2_list = [], [], []
    fine_list, coarse_list = [], []
    samples_count = 0
    
    for x, fine_labels in tqdm(loader, desc="Evaluating accuracy & extracting representations"):
        x = x.to(device)
        fine_labels = fine_labels.to(device)
        coarse_labels = get_coarse_labels(fine_labels, dataset).to(device)
        
        # FIN and baseline models have different forward signatures
        if hasattr(model, 'level0'):  # FIN model
            out = model(x, fine_labels, coarse_labels)
            fine_logits   = out.fine_logits
            coarse_logits = out.coarse_logits
        else:  # Baseline MobileNet
            fine_logits = model(x)
            if hasattr(model, 'coarse_head'):
                coarse_logits = model.coarse_head(model.features(x).mean([2, 3]))
            else:
                coarse_logits = fine_logits
        
        fine_correct   += (fine_logits.argmax(1)   == fine_labels).sum().item()
        coarse_correct += (coarse_logits.argmax(1) == coarse_labels).sum().item()
        total += x.size(0)
        
        # Also extract representations for analysis (deterministic paths)
        if samples_count < max_samples:
            if hasattr(model, 'encode_deterministic'):
                z0, z1, z2 = model.encode_deterministic(x)
            else:
                # For baselines, use the feature extractor output for all three levels
                feat = model.features(x).mean([2, 3])  # global avg pool → (B, C)
                z0 = z1 = z2 = feat
            
            remaining = max_samples - samples_count
            z0_list.append(z0[:remaining].cpu().numpy())
            z1_list.append(z1[:remaining].cpu().numpy())
            z2_list.append(z2[:remaining].cpu().numpy())
            fine_list.append(fine_labels[:remaining].cpu().numpy())
            coarse_list.append(coarse_labels[:remaining].cpu().numpy())
            
            samples_count += min(x.size(0), remaining)
    
    fine_acc = 100.0 * fine_correct / total
    coarse_acc = 100.0 * coarse_correct / total
    
    # For CIFAR-10, coarse = fine (no hierarchy)
    if dataset == "cifar10":
        coarse_acc = fine_acc
    
    joint = fine_acc + coarse_acc
    
    accuracies = {
        "fine_acc"  : fine_acc,
        "coarse_acc": coarse_acc,
        "joint_acc" : joint,
    }
    
    representations = {
        "z0"           : np.concatenate(z0_list, axis=0)[:max_samples],
        "z1"           : np.concatenate(z1_list, axis=0)[:max_samples],
        "z2"           : np.concatenate(z2_list, axis=0)[:max_samples],
        "fine_labels"  : np.concatenate(fine_list, axis=0)[:max_samples],
        "coarse_labels": np.concatenate(coarse_list, axis=0)[:max_samples],
    }
    
    return accuracies, representations


@torch.no_grad()
def extract_representations(
    model      : FIN,
    loader     : DataLoader,
    device     : torch.device,
    max_samples: int = 5000,
    dataset    : str = "cifar100",
) -> Dict:
    """
    Extract z0, z1, z2 representations and labels from the dataset.
    Uses deterministic forward (channel means, no sampling) for
    stable, reproducible embeddings suitable for analysis.
    """
    z0_list, z1_list, z2_list = [], [], []
    fine_list, coarse_list = [], []
    total = 0
    
    for x, fine_labels in tqdm(loader, desc="Extracting representations"):
        if total >= max_samples:
            break
        
        x = x.to(device)
        fine_labels = fine_labels.to(device)
        coarse_labels = get_coarse_labels(fine_labels, dataset).to(device)
        
        if hasattr(model, 'encode_deterministic'):
            z0, z1, z2 = model.encode_deterministic(x)
        else:
            # For baselines, use the feature extractor output for all three levels
            feat = model.features(x).mean([2, 3])  # global avg pool → (B, C)
            z0 = z1 = z2 = feat
        
        z0_list.append(z0.cpu().numpy())
        z1_list.append(z1.cpu().numpy())
        z2_list.append(z2.cpu().numpy())
        fine_list.append(fine_labels.cpu().numpy())
        coarse_list.append(coarse_labels.cpu().numpy())
        
        total += x.size(0)
    
    return {
        "z0"           : np.concatenate(z0_list, axis=0)[:max_samples],
        "z1"           : np.concatenate(z1_list, axis=0)[:max_samples],
        "z2"           : np.concatenate(z2_list, axis=0)[:max_samples],
        "fine_labels"  : np.concatenate(fine_list, axis=0)[:max_samples],
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
    dataset: str = "cifar100",
) -> Dict[str, float]:
    """
    Compute fine and coarse top-1 accuracy on the full test set.
    
    For CIFAR-10: coarse_acc = fine_acc (no hierarchy)
    
    Returns:
        dict: fine_acc, coarse_acc, joint_acc (all in %)
    """
    model.eval()
    fine_correct = 0
    coarse_correct = 0
    total = 0
    
    num_classes = get_num_classes(dataset)
    
    for x, fine_labels in tqdm(loader, desc="Evaluating accuracy"):
        x = x.to(device)
        fine_labels = fine_labels.to(device)
        coarse_labels = get_coarse_labels(fine_labels, dataset).to(device)
        
        # FIN and baseline models have different forward signatures
        if hasattr(model, 'level0'):  # FIN model
            out = model(x, fine_labels, coarse_labels)
            fine_logits   = out.fine_logits
            coarse_logits = out.coarse_logits
        else:  # Baseline MobileNet
            fine_logits = model(x)
            # MobileNetV2Aux has a separate coarse head; MobileNetV2Fine doesn't
            if hasattr(model, 'coarse_head'):
                coarse_logits = model.coarse_head(model.features(x).mean([2, 3]))
            else:
                coarse_logits = fine_logits
        
        fine_correct   += (fine_logits.argmax(1)   == fine_labels).sum().item()
        coarse_correct += (coarse_logits.argmax(1) == coarse_labels).sum().item()
        total += x.size(0)
    
    fine_acc = 100.0 * fine_correct / total
    coarse_acc = 100.0 * coarse_correct / total
    
    # For CIFAR-10, coarse = fine (no hierarchy)
    if dataset == "cifar10":
        coarse_acc = fine_acc
    
    joint = fine_acc + coarse_acc
    
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
        tsne = TSNE(
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
    label_type     : str = "coarse",
    dataset        : str = "cifar100",
):
    """
    Plot t-SNE embeddings for all three levels side by side.
    This is Figure X in the paper — direct visual evidence for Theorem 1.1.
    
    Color = class label. Tighter clusters at higher levels = lower entropy
    = more abstract, more organized representation space.
    """
    if label_type == "coarse":
        labels = representations["coarse_labels"]
        n_classes = 20 if dataset == "cifar100" else 10
        if dataset == "cifar100":
            class_names = CIFAR100_SUPERCLASSES
        else:
            class_names = CIFAR10_CLASSES
        title_suffix = f"({n_classes} superclasses)"
    else:
        labels = representations["fine_labels"]
        n_classes = 100 if dataset == "cifar100" else 10
        if dataset == "cifar100":
            class_names = [str(i) for i in range(100)]
        else:
            class_names = CIFAR10_CLASSES
        title_suffix = f"({n_classes} fine classes)"
    
    # Color palette
    if n_classes <= 20:
        cmap = plt.cm.get_cmap("tab20", n_classes)
        colors = [cmap(i) for i in range(n_classes)]
    else:
        cmap = plt.cm.get_cmap("nipy_spectral", n_classes)
        colors = [cmap(i) for i in range(n_classes)]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"FIN Representation t-SNE — {title_suffix}\n"
        f"Theorem 1.1: each level should show tighter clustering than the one below",
        fontsize=13, y=1.02
    )
    
    level_info = [
        ("tsne_z0", "L0 — Raw Signal\n(CNN, d=512)", "z0"),
        ("tsne_z1", "L1 — Local Patterns\n(Transformer, d=128)", "z1"),
        ("tsne_z2", "L2 — Abstract Concepts\n(MLP Apex, d=32)", "z2"),
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
        
        # Compute and display cluster tightness
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
    
    # Legend (only for coarse — classes is readable)
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


def compute_cross_seed_cka(
    representations_list: List[Dict],
    subsample: int = 500,
) -> np.ndarray:
    """
    Compute CKA matrix between levels across seeds.
    Measures stability of representations across different random initializations.
    
    Returns:
        (n_seeds, 3, 3) array of CKA matrices
    """
    n_seeds = len(representations_list)
    all_ckas = np.zeros((n_seeds, 3, 3))
    
    for seed_idx, reps in enumerate(representations_list):
        all_ckas[seed_idx] = compute_cka_matrix(reps, subsample)
    
    return all_ckas


def plot_cka_matrix(cka_matrix: np.ndarray, save_path: str, title: str = None):
    """
    Plot CKA similarity matrix as a heatmap.
    """
    if title is None:
        title = (
            "CKA Representation Similarity Matrix\n"
            "Low off-diagonal = genuinely different representations per level"
        )
    
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
    
    ax.set_title(title, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] CKA matrix saved: {save_path}")


# =============================================================================
# Representation Statistics
# =============================================================================

def compute_representation_stats(representations: Dict) -> Dict:
    """
    Compute statistics that directly test Theorem 1.1's predictions.
    """
    stats = {}
    for key in ["z0", "z1", "z2"]:
        z = representations[key]
        stats[key] = {
            "dim"           : z.shape[1],
            "var_mean"      : float(z.var(axis=0).mean()),
            "var_total"     : float(z.var()),
            "norm_mean"     : float(np.linalg.norm(z, axis=1).mean()),
            "norm_std"      : float(np.linalg.norm(z, axis=1).std()),
            "intrinsic_dim" : two_nn_intrinsic_dim(z),
            "participation_ratio": participation_ratio(z),
        }
    
    # Add activation stats if available
    if "activation_stats" in representations:
        stats["activation"] = representations["activation_stats"]
    
    return stats


def participation_ratio(Z: np.ndarray) -> float:
    """
    Compute participation ratio as intrinsic dimensionality estimate.
    PR = (sum lambda_i)^2 / sum(lambda_i^2)
    where lambda_i are eigenvalues of covariance matrix.
    """
    Z_centered = Z - Z.mean(axis=0)
    cov = np.cov(Z_centered.T)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 0)  # Numerical stability
    total = eigvals.sum()
    if total == 0:
        return 0.0
    return float((total ** 2) / (eigvals ** 2).sum())


def two_nn_intrinsic_dim(Z: np.ndarray, subsample: int = 2000) -> float:
    """
    Two-NN intrinsic dimensionality estimator (Facco et al., 2017).
    """
    n = min(subsample, len(Z))
    Z = Z[:n]
    
    nbrs = NearestNeighbors(n_neighbors=2).fit(Z)
    distances, _ = nbrs.kneighbors(Z)
    
    r1 = distances[:, 0]
    r2 = distances[:, 1]
    
    mu = r2 / (r1 + 1e-10)
    mu = mu[mu > 1]
    
    if len(mu) == 0:
        return 0.0
    
    intrinsic_dim = 1.0 / (np.log(mu).mean())
    return float(intrinsic_dim)


def print_representation_report(
    stats       : Dict,
    accuracies  : Dict,
    cka_matrix  : np.ndarray,
    model       : FIN,
    dataset     : str = "cifar100",
):
    """
    Print a full evaluation report suitable for the paper appendix.
    """
    dataset_name = "CIFAR-100" if dataset == "cifar100" else "CIFAR-10"
    
    print("\n" + "=" * 65)
    print(f"FIN EVALUATION REPORT ({dataset_name})")
    print("=" * 65)
    
    print("\n── Accuracy (Section 5.1) ──────────────────────────────────")
    print(f"  Fine   accuracy (L1, {get_num_classes(dataset)} classes) : {accuracies['fine_acc']:.2f}%")
    print(f"  Coarse accuracy (L2, {'20' if dataset == 'cifar100' else '10'} classes) : {accuracies['coarse_acc']:.2f}%")
    print(f"  Joint  accuracy (fine + coarse)   : {accuracies['joint_acc']:.2f}")
    
    print("\n── Representation Statistics (Theorem 1.1 evidence) ────────")
    print(f"  {'Level':<8} {'Dim':<8} {'Var(mean)':<14} {'||z|| mean':<14} {'Intrinsic Dim':<14} {'Part. Ratio':<14}")
    print(f"  {'-'*80}")
    for key, label in [("z0","L0"), ("z1","L1"), ("z2","L2")]:
        s = stats[key]
        pr = s.get('participation_ratio', 0)
        print(f"  {label:<8} {s['dim']:<8} {s['var_mean']:<14.4f} "
            f"{s['norm_mean']:<14.4f} {s['intrinsic_dim']:<14.4f} {pr:<14.4f}")
    print()
    
    # Verify Theorem 1.1 prediction
    id_z0 = stats["z0"]["intrinsic_dim"]
    id_z1 = stats["z1"]["intrinsic_dim"]
    id_z2 = stats["z2"]["intrinsic_dim"]
    
    if id_z0 > id_z1 > id_z2:
        print("  [PASS] ID(z0) > ID(z1) > ID(z2) — Theorem 1.1 confirmed")
    else:
        print(f"  [NOTE] Intrinsic dim not strictly decreasing:")
        print(f"         ID_z0={id_z0:.2f}, ID_z1={id_z1:.2f}, ID_z2={id_z2:.2f}")
    
    # Participation ratio check
    pr_z0 = stats["z0"]["participation_ratio"]
    pr_z1 = stats["z1"]["participation_ratio"]
    pr_z2 = stats["z2"]["participation_ratio"]
    print(f"\n  Participation ratio: z0={pr_z0:.2f}, z1={pr_z1:.2f}, z2={pr_z2:.2f}")
    if pr_z0 > pr_z1 > pr_z2:
        print("  [PASS] Participation ratio decreasing — compressed hierarchy")
    
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
    if model is not None and hasattr(model, '_current_betas'):
        for k, v in model._current_betas.items():
            print(f"  {k}: beta_max={v:.3f}")
    
    print("\n── Alpha Gate Values (top-down influence) ──────────────────")
    if model is not None:
        if hasattr(model.level0, "alpha"):
            a0 = model.level0.alpha.item()
            print(f"  alpha_0 (L1->L0 influence): {a0:.4f}")
        if hasattr(model.level1, "alpha"):
            a1 = model.level1.alpha.item()
            print(f"  alpha_1 (L2->L1 influence): {a1:.4f}")
    else:
        print("  [Model not available for alpha analysis]")
    
    print("\n── Parameter Count ─────────────────────────────────────────")
    if model is not None:
        pc = model.param_count()
        for k, v in pc.items():
            if k != "total_M":
                print(f"  {k:<16}: {v:>10,}")
        print(f"  {'Total (M)':<16}: {pc['total_M']:>10.2f}M")
    else:
        print("  [Model not available for parameter count]")
    
    # Activation statistics
    if "activation" in stats:
        print("\n── Activation Statistics ──────────────────────────────────")
        act_stats = stats["activation"]
        for level in ["z0", "z1", "z2"]:
            mean_key = f"{level}_mean"
            std_key = f"{level}_std"
            sparsity_key = f"{level}_sparsity"
            if mean_key in act_stats:
                print(f"  {level}: mean={act_stats[mean_key]:.4f}, std={act_stats[std_key]:.4f}, "
                      f"sparsity={act_stats[sparsity_key]*100:.1f}%")
    
    print("=" * 65)


# =============================================================================
# Multi-Seed Evaluation
# =============================================================================

def evaluate_single_seed(
    cfg: dict,
    checkpoint_path: str,
    dataset: str,
    device: torch.device,
    max_samples: int = 5000,
    val_loader: DataLoader = None,
) -> Dict[str, Any]:
    """
    Evaluate a single checkpoint on a single dataset.
    
    Args:
        cfg: Configuration dict
        checkpoint_path: Path to model checkpoint
        dataset: Dataset name ("cifar10" or "cifar100")
        device: Compute device
        max_samples: Max representations to extract
        val_loader: Optional pre-loaded DataLoader (cached from evaluate())
                   If not provided, will load dataset
    """
    # Load dataset once (or use cached version)
    if val_loader is None:
        val_loader = load_dataset(
            dataset=dataset,
            data_root=cfg["data"]["root"],
            batch_size=cfg["data"]["batch_size"],
            num_workers=cfg["data"]["num_workers"],
        )
    
    # Load model
    model = load_model(cfg, checkpoint_path, device, dataset)
    
    # Count FLOPs and params
    model_info = count_flops(model, input_size=(1, 3, 32, 32))
    
    # Evaluate accuracy AND extract representations in a single forward pass
    accuracies, representations = evaluate_accuracy_and_representations(
        model, val_loader, device, max_samples, dataset
    )
    
    # Compute CKA
    cka_matrix = compute_cka_matrix(representations)
    
    # Compute representation stats
    stats = compute_representation_stats(representations)
    
    return {
        "checkpoint_path": checkpoint_path,
        "dataset": dataset,
        "accuracies": accuracies,
        "cka_matrix": cka_matrix,
        "representation_stats": stats,
        "model_info": model_info,
        "param_count": model.param_count() if hasattr(model, 'param_count') else {
            "total": sum(p.numel() for p in model.parameters()),
            "total_M": f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}",
        },
    }


def compute_multi_seed_statistics(results: List[Dict]) -> Dict[str, Any]:
    """
    Compute mean ± std statistics across multiple seeds.
    """
    # Collect metric arrays
    fine_accs = [r["accuracies"]["fine_acc"] for r in results]
    coarse_accs = [r["accuracies"]["coarse_acc"] for r in results]
    joint_accs = [r["accuracies"]["joint_acc"] for r in results]
    
    # CKA values
    cka_01 = [r["cka_matrix"][0, 1] for r in results]
    cka_12 = [r["cka_matrix"][1, 2] for r in results]
    cka_02 = [r["cka_matrix"][0, 2] for r in results]
    
    # Intrinsic dimensions
    id_z0 = [r["representation_stats"]["z0"]["intrinsic_dim"] for r in results]
    id_z1 = [r["representation_stats"]["z1"]["intrinsic_dim"] for r in results]
    id_z2 = [r["representation_stats"]["z2"]["intrinsic_dim"] for r in results]
    
    # FLOPs and params (should be same across seeds)
    flops = results[0]["model_info"]["flops_M"]
    params = results[0]["model_info"]["params_M"]
    
    stats = {
        "fine_acc": {"mean": np.mean(fine_accs), "std": np.std(fine_accs),
                     "min": np.min(fine_accs), "max": np.max(fine_accs)},
        "coarse_acc": {"mean": np.mean(coarse_accs), "std": np.std(coarse_accs),
                       "min": np.min(coarse_accs), "max": np.max(coarse_accs)},
        "joint_acc": {"mean": np.mean(joint_accs), "std": np.std(joint_accs),
                      "min": np.min(joint_accs), "max": np.max(joint_accs)},
        "cka": {
            "01": {"mean": np.mean(cka_01), "std": np.std(cka_01)},
            "12": {"mean": np.mean(cka_12), "std": np.std(cka_12)},
            "02": {"mean": np.mean(cka_02), "std": np.std(cka_02)},
        },
        "intrinsic_dim": {
            "z0": {"mean": np.mean(id_z0), "std": np.std(id_z0)},
            "z1": {"mean": np.mean(id_z1), "std": np.std(id_z1)},
            "z2": {"mean": np.mean(id_z2), "std": np.std(id_z2)},
        },
        "model_info": {"flops_M": flops, "params_M": params},
        "n_seeds": len(results),
    }
    
    return stats


def print_multi_seed_report(
    seed_results: List[Dict],
    stats: Dict,
    dataset: str,
):
    """Print summary report for multi-seed evaluation."""
    dataset_name = "CIFAR-100" if dataset == "cifar100" else "CIFAR-10"
    
    print("\n" + "=" * 80)
    print(f"MULTI-SEED EVALUATION REPORT ({dataset_name})")
    print(f"Seeds evaluated: {stats['n_seeds']}")
    print("=" * 80)
    
    print("\n── Accuracy Summary ─────────────────────────────────────────")
    print(f"  Fine Acc:   {stats['fine_acc']['mean']:.2f} ± {stats['fine_acc']['std']:.2f} "
          f"(min={stats['fine_acc']['min']:.2f}, max={stats['fine_acc']['max']:.2f})")
    print(f"  Coarse Acc: {stats['coarse_acc']['mean']:.2f} ± {stats['coarse_acc']['std']:.2f} "
          f"(min={stats['coarse_acc']['min']:.2f}, max={stats['coarse_acc']['max']:.2f})")
    print(f"  Joint Acc:  {stats['joint_acc']['mean']:.2f} ± {stats['joint_acc']['std']:.2f}")
    
    print("\n── Per-Seed Accuracy ────────────────────────────────────────")
    print(f"  {'Seed':<8} {'Fine Acc':>10} {'Coarse Acc':>12} {'Joint':>10}")
    print(f"  {'-'*44}")
    for i, result in enumerate(seed_results):
        acc = result["accuracies"]
        print(f"  {i+1:<8} {acc['fine_acc']:>10.2f} {acc['coarse_acc']:>12.2f} {acc['joint_acc']:>10.2f}")
    
    print("\n── CKA Similarity (mean ± std) ─────────────────────────────")
    print(f"  CKA(L0, L1): {stats['cka']['01']['mean']:.3f} ± {stats['cka']['01']['std']:.3f}")
    print(f"  CKA(L1, L2): {stats['cka']['12']['mean']:.3f} ± {stats['cka']['12']['std']:.3f}")
    print(f"  CKA(L0, L2): {stats['cka']['02']['mean']:.3f} ± {stats['cka']['02']['std']:.3f}")
    
    print("\n── Intrinsic Dimensionality (mean ± std) ───────────────────")
    print(f"  ID(L0): {stats['intrinsic_dim']['z0']['mean']:.2f} ± {stats['intrinsic_dim']['z0']['std']:.2f}")
    print(f"  ID(L1): {stats['intrinsic_dim']['z1']['mean']:.2f} ± {stats['intrinsic_dim']['z1']['std']:.2f}")
    print(f"  ID(L2): {stats['intrinsic_dim']['z2']['mean']:.2f} ± {stats['intrinsic_dim']['z2']['std']:.2f}")
    
    print("\n── Model Efficiency ────────────────────────────────────────")
    print(f"  Parameters: {stats['model_info']['params_M']:.2f}M")
    print(f"  FLOPs: {stats['model_info']['flops_M']:.2f}M")
    
    print("=" * 80)


# =============================================================================
# Efficiency Analysis Plots
# =============================================================================

def plot_efficiency_analysis(
    all_results: Dict[str, List[Dict]],
    save_dir: str,
):
    """
    Generate efficiency analysis plots: Params vs Accuracy, FLOPs vs Accuracy.
    all_results: {model_name: [list of seed results]}
    """
    # Collect data points
    model_names = []
    params_list = []
    flops_list = []
    fine_accs_mean = []
    fine_accs_std = []
    
    for model_name, results in all_results.items():
        if not results:
            continue
        stats = compute_multi_seed_statistics(results)
        
        model_names.append(model_name)
        params_list.append(stats["model_info"]["params_M"])
        flops_list.append(stats["model_info"]["flops_M"])
        fine_accs_mean.append(stats["fine_acc"]["mean"])
        fine_accs_std.append(stats["fine_acc"]["std"])
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
    
    # Plot 1: Params vs Accuracy
    ax1 = axes[0]
    for i, (name, params, acc, acc_std) in enumerate(zip(
        model_names, params_list, fine_accs_mean, fine_accs_std
    )):
        ax1.errorbar(params, acc, yerr=acc_std, fmt='o', markersize=10,
                     color=colors[i], label=name, capsize=5, capthick=2)
    ax1.set_xlabel('Parameters (M)', fontsize=12)
    ax1.set_ylabel('Fine Accuracy (%)', fontsize=12)
    ax1.set_title('Parameters vs Accuracy', fontsize=14)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: FLOPs vs Accuracy
    ax2 = axes[1]
    for i, (name, flops, acc, acc_std) in enumerate(zip(
        model_names, flops_list, fine_accs_mean, fine_accs_std
    )):
        ax2.errorbar(flops, acc, yerr=acc_std, fmt='o', markersize=10,
                     color=colors[i], label=name, capsize=5, capthick=2)
    ax2.set_xlabel('FLOPs (M)', fontsize=12)
    ax2.set_ylabel('Fine Accuracy (%)', fontsize=12)
    ax2.set_title('FLOPs vs Accuracy', fontsize=14)
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "efficiency_analysis.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Efficiency analysis saved: {save_path}")
    
    # Print efficiency table
    print("\n── Efficiency Comparison Table ──────────────────────────────")
    print(f"  {'Model':<20} {'Params (M)':>10} {'FLOPs (M)':>10} {'Fine Acc':>10} "
          f"{'Coarse Acc':>12} {'Acc/Param':>10} {'Acc/FLOP':>10}")
    print(f"  {'-'*86}")
    
    for model_name, results in all_results.items():
        if not results:
            continue
        stats = compute_multi_seed_statistics(results)
        fine_acc = stats["fine_acc"]["mean"]
        coarse_acc = stats["coarse_acc"]["mean"]
        params = stats["model_info"]["params_M"]
        flops = stats["model_info"]["flops_M"]
        acc_per_param = fine_acc / params if params > 0 else 0
        acc_per_flop = fine_acc / flops if flops > 0 else 0
        
        print(f"  {model_name:<20} {params:>10.2f} {flops:>10.2f} {fine_acc:>10.2f} "
              f"{coarse_acc:>12.2f} {acc_per_param:>10.2f} {acc_per_flop:>10.2f}")
    
    print()


def plot_multi_seed_comparison(
    seed_results: List[Dict],
    save_dir: str,
    dataset: str,
    model_name: str = "FIN",
):
    """
    Generate plots showing results across seeds with error bars.
    """
    # Metrics to plot
    metrics = ["fine_acc", "coarse_acc", "joint_acc"]
    metric_names = ["Fine Accuracy", "Coarse Accuracy", "Joint Accuracy"]
    
    # Extract values
    n_seeds = len(seed_results)
    seeds = list(range(1, n_seeds + 1))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, metric, name in zip(axes, metrics, metric_names):
        values = [r["accuracies"][metric] for r in seed_results]
        colors = plt.cm.viridis(np.linspace(0, 1, n_seeds))
        
        bars = ax.bar(seeds, values, color=colors, edgecolor='black', linewidth=1)
        ax.axhline(y=np.mean(values), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(values):.2f}')
        ax.fill_between([0.5, n_seeds + 0.5], 
                        np.mean(values) - np.std(values),
                        np.mean(values) + np.std(values),
                        color='red', alpha=0.2, label=f'±1 std: {np.std(values):.2f}')
        
        ax.set_xlabel('Seed', fontsize=12)
        ax.set_ylabel(f'{name} (%)', fontsize=12)
        ax.set_title(f'{name} Across Seeds', fontsize=14)
        ax.set_xticks(seeds)
        ax.legend(loc='lower right')
        ax.set_ylim([0, 100])
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'{model_name} - {dataset.upper()} Multi-Seed Evaluation', fontsize=14, y=1.02)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"multi_seed_{dataset}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Multi-seed comparison saved: {save_path}")


# =============================================================================
# Results Table
# =============================================================================

def print_results_table(results: List[Dict]):
    """
    Print a formatted results table for the paper.
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
# JSON Reporting
# =============================================================================

def save_json_results(
    seed_results: Dict[str, Dict],
    output_dir: str,
    config: dict,
    args: argparse.Namespace,
):
    """
    Save comprehensive JSON results file.
    Handles both flat structure {model_name -> [results]} 
    and nested structure {model_name -> {dataset -> [results]}}
    """
    json_results = {
        "config": {
            "dataset": args.dataset,
            "seeds": args.seeds,
            "checkpoint_dir": args.checkpoint_dir,
            "checkpoint_file": args.checkpoint,
            "model_config": config.get("model", {}),
        },
        "per_model": {},
        "summary": {},
    }
    
    # Per-model results
    for model_name, results_or_dict in seed_results.items():
        if not results_or_dict:
            continue
        
        # Handle nested structure: {dataset -> [results]}
        if isinstance(results_or_dict, dict):
            # results_or_dict is {dataset -> [results]}
            for dataset_key, results_list in results_or_dict.items():
                if not results_list:
                    continue
                
                # Create a key that includes dataset for clarity
                results_key = f"{model_name}_{dataset_key}" if len(results_or_dict) > 1 else model_name
                
                if isinstance(results_list, list) and len(results_list) > 0:
                    dataset = results_list[0].get("dataset", dataset_key)
                    stats = compute_multi_seed_statistics(results_list)
                    
                    json_results["per_model"][results_key] = {
                        "dataset": dataset,
                        "n_seeds": len(results_list),
                        "per_seed": [
                            {
                                "seed": i + 1,
                                "checkpoint": r.get("checkpoint_path", ""),
                                "accuracies": r.get("accuracies", {}),
                                "cka_matrix": r.get("cka_matrix", np.eye(3)).tolist() if hasattr(r.get("cka_matrix", []), "tolist") else [],
                                "representation_stats": {
                                    k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv 
                                        for kk, vv in v.items()}
                                    for k, v in r.get("representation_stats", {}).items()
                                },
                            }
                            for i, r in enumerate(results_list)
                        ],
                        "summary_statistics": {
                            "accuracies": {
                                "fine_acc": stats.get("fine_acc", {}),
                                "coarse_acc": stats.get("coarse_acc", {}),
                                "joint_acc": stats.get("joint_acc", {}),
                            },
                            "cka": stats.get("cka", {}),
                            "intrinsic_dim": stats.get("intrinsic_dim", {}),
                            "model_info": stats.get("model_info", {}),
                        }
                    }
         # Handle flat structure: [results] (backward compatibility)
        elif isinstance(results_or_dict, list) and len(results_or_dict) > 0:
            dataset = results_or_dict[0].get("dataset", args.dataset)
            stats = compute_multi_seed_statistics(results_or_dict)
            
            json_results["per_model"][model_name] = {
                "dataset": dataset,
                "n_seeds": len(results_or_dict),
                "per_seed": [
                    {
                        "seed": i + 1,
                        "checkpoint": r.get("checkpoint_path", ""),
                        "accuracies": r.get("accuracies", {}),
                        "cka_matrix": r.get("cka_matrix", np.eye(3)).tolist() if hasattr(r.get("cka_matrix", []), "tolist") else [],
                        "representation_stats": {
                            k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv 
                                for kk, vv in v.items()}
                            for k, v in r.get("representation_stats", {}).items()
                        },
                    }
                    for i, r in enumerate(results_or_dict)
                ],
                "summary_statistics": {
                    "accuracies": {
                        "fine_acc": stats.get("fine_acc", {}),
                        "coarse_acc": stats.get("coarse_acc", {}),
                        "joint_acc": stats.get("joint_acc", {}),
                    },
                    "cka": stats.get("cka", {}),
                    "intrinsic_dim": stats.get("intrinsic_dim", {}),
                    "model_info": stats.get("model_info", {}),
                }
            }
    
    # Summary across all models - simplified
    if len(seed_results) > 1:
        # Build summary data by extracting first dataset results from each model
        model_summaries = {}
        for model_name, results_or_dict in seed_results.items():
            try:
                if isinstance(results_or_dict, dict):
                    # Get first dataset's results
                    first_results = list(results_or_dict.values())[0]
                elif isinstance(results_or_dict, list):
                    first_results = results_or_dict
                else:
                    continue
                    
                if first_results:
                    stats = compute_multi_seed_statistics(first_results)
                    model_summaries[model_name] = stats
            except:
                continue
        
        if model_summaries:
            json_results["summary"] = {
                "best_model_by_accuracy": max(
                    model_summaries.keys(),
                    key=lambda k: model_summaries[k].get("fine_acc", {}).get("mean", 0)
                ),
                "best_model_by_efficiency": max(
                    model_summaries.keys(),
                    key=lambda k: model_summaries[k].get("fine_acc", {}).get("mean", 0) /
                                  (model_summaries[k].get("model_info", {}).get("params_M", 1) or 1)
                ),
            }
    
    # Save JSON
    json_path = os.path.join(output_dir, "evaluation_results.json")
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"[Save] JSON results saved: {json_path}")
    
    return json_results


def generate_markdown_report(
    seed_results: Dict[str, List[Dict]],
    output_dir: str,
    config: dict,
    args: argparse.Namespace,
):
    """
    Generate markdown report with all tables and figures.
    """
    md_lines = []
    md_lines.append("# FIN Evaluation Report\n")
    md_lines.append(f"**Dataset**: {args.dataset.upper()}")
    md_lines.append(f"**Seeds**: {args.seeds}")
    md_lines.append(f"**Models Evaluated**: {len(seed_results)}\n")
    
    # Results table
    md_lines.append("## Accuracy Results\n")
    md_lines.append("| Model | Fine Acc | Coarse Acc | Joint Acc | Params (M) | FLOPs (M) |")
    md_lines.append("|-------|----------|------------|-----------|------------|-----------|")
    
    for model_name, results_or_dict in seed_results.items():
        if not results_or_dict:
            continue
        
        # Handle nested structure: {dataset -> [results]}
        if isinstance(results_or_dict, dict):
            # Get first dataset's results for this model
            results = list(results_or_dict.values())[0]
        else:
            # Already a flat list
            results = results_or_dict
        
        if not results:
            continue
            
        stats = compute_multi_seed_statistics(results)
        fine_acc = f"{stats['fine_acc']['mean']:.2f} ± {stats['fine_acc']['std']:.2f}"
        coarse_acc = f"{stats['coarse_acc']['mean']:.2f} ± {stats['coarse_acc']['std']:.2f}"
        joint_acc = f"{stats['joint_acc']['mean']:.2f} ± {stats['joint_acc']['std']:.2f}"
        params = f"{stats['model_info']['params_M']:.2f}"
        flops = f"{stats['model_info']['flops_M']:.2f}"
        md_lines.append(f"| {model_name} | {fine_acc} | {coarse_acc} | {joint_acc} | {params} | {flops} |")
    
    md_lines.append("")
    
    # CKA matrix
    md_lines.append("## CKA Representation Similarity\n")
    md_lines.append("Mean ± std across seeds:\n")
    
    for model_name, results_or_dict in seed_results.items():
        if not results_or_dict:
            continue
        
        # Handle nested structure: {dataset -> [results]}
        if isinstance(results_or_dict, dict):
            results = list(results_or_dict.values())[0]
        else:
            results = results_or_dict
        
        if not results:
            continue
        
        stats = compute_multi_seed_statistics(results)
        md_lines.append(f"### {model_name}\n")
        md_lines.append(f"- CKA(L0, L1): {stats['cka']['01']['mean']:.3f} ± {stats['cka']['01']['std']:.3f}")
        md_lines.append(f"- CKA(L1, L2): {stats['cka']['12']['mean']:.3f} ± {stats['cka']['12']['std']:.3f}")
        md_lines.append(f"- CKA(L0, L2): {stats['cka']['02']['mean']:.3f} ± {stats['cka']['02']['std']:.3f}\n")
    
    # Intrinsic dimensionality
    md_lines.append("## Intrinsic Dimensionality\n")
    md_lines.append("| Model | ID(L0) | ID(L1) | ID(L2) |")
    md_lines.append("|-------|--------|--------|-------|")
    
    for model_name, results_or_dict in seed_results.items():
        if not results_or_dict:
            continue
        
        # Handle nested structure: {dataset -> [results]}
        if isinstance(results_or_dict, dict):
            results = list(results_or_dict.values())[0]
        else:
            results = results_or_dict
        
        if not results:
            continue
        
        stats = compute_multi_seed_statistics(results)
        id_z0 = f"{stats['intrinsic_dim']['z0']['mean']:.2f} ± {stats['intrinsic_dim']['z0']['std']:.2f}"
        id_z1 = f"{stats['intrinsic_dim']['z1']['mean']:.2f} ± {stats['intrinsic_dim']['z1']['std']:.2f}"
        id_z2 = f"{stats['intrinsic_dim']['z2']['mean']:.2f} ± {stats['intrinsic_dim']['z2']['std']:.2f}"
        md_lines.append(f"| {model_name} | {id_z0} | {id_z1} | {id_z2} |")
    
    md_lines.append("")
    
    # Figures
    md_lines.append("## Figures\n")
    md_lines.append(f"- [CKA Matrix](cka_matrix.png)")
    md_lines.append(f"- [Multi-seed Comparison](multi_seed_{args.dataset}.png)")
    md_lines.append(f"- [Efficiency Analysis](efficiency_analysis.png)")
    if args.tsne:
        md_lines.append(f"- [t-SNE (coarse)](tsne_coarse.png)")
        md_lines.append(f"- [t-SNE (fine)](tsne_fine.png)")
    
    # Save markdown
    md_path = os.path.join(output_dir, "evaluation_report.md")
    with open(md_path, 'w') as f:
        f.write("\n".join(md_lines))
    print(f"[Save] Markdown report saved: {md_path}")


# =============================================================================
# Main Evaluation Function
# =============================================================================

def evaluate(
    cfg           : dict,
    checkpoint_path: Optional[str],
    run_tsne      : bool = False,
    compare_path  : Optional[str] = None,
    output_dir    : str = "eval_outputs",
    max_samples   : int = 5000,
    dataset       : str = "cifar100",
    seeds         : List[int] = None,
    checkpoint_dir: str = None,
    model_prefix  : str = "fin",
    generate_efficiency_plots: bool = False,
):
    """
    Full evaluation pipeline with multi-seed and multi-dataset support.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Device: {device}")
    
    # Determine datasets to evaluate
    if dataset == "both":
        datasets_to_eval = ["cifar100", "cifar10"]
    else:
        datasets_to_eval = [dataset]
    
    # Find checkpoint paths for each seed
    if seeds is None:
        seeds = [1, 2, 3]
    
    all_seed_results = {}  # model_name -> {dataset -> [results]}
    
    # Cache dataloaders to avoid reloading same dataset across seeds
    dataloader_cache = {}
    
    def get_cached_dataloader(ds):
        """Get or create cached dataloader for dataset."""
        if ds not in dataloader_cache:
            print(f"[Cache] Loading {ds.upper()} dataset...")
            dataloader_cache[ds] = load_dataset(
                dataset=ds,
                data_root=cfg["data"]["root"],
                batch_size=cfg["data"]["batch_size"],
                num_workers=cfg["data"]["num_workers"],
            )
        return dataloader_cache[ds]

    
    # If checkpoint_dir provided, auto-discover all models and seeds
    if checkpoint_dir and os.path.exists(checkpoint_dir):
        # Find all checkpoint directories with seed numbers
        import re
        
        model_checkpoints = {}  # {model_name: {seed: path}}
        
        # List all directories in checkpoint_dir
        dirs = [d for d in os.listdir(checkpoint_dir) 
                if os.path.isdir(os.path.join(checkpoint_dir, d))]
        
        for directory in dirs:
            # Extract seed number from directory name (e.g., "fin_cifar100_cifar100_cifar100_seed1" -> seed=1)
            seed_match = re.search(r'seed(\d+)', directory)
            if seed_match:
                seed = int(seed_match.group(1))
                # Extract model name by removing the trailing dataset + seed suffix.
                # train.py appends the dataset name once in main(), so FIN on cifar100
                # produces "fin_cifar100_cifar100_seed1" → strip "_cifar100_seed1" → "fin_cifar100"
                # Baselines produce "mobilenet_fine_cifar100_seed1" → "mobilenet_fine"
                model_name = re.sub(r'_(cifar100|cifar10)_seed\d+$', '', directory)
                
                if model_name not in model_checkpoints:
                    model_checkpoints[model_name] = {}
                
                # Try both best_seed{N}.pt and best.pt patterns
                ckpt_path = os.path.join(checkpoint_dir, directory, f"best_seed{seed}.pt")
                if not os.path.exists(ckpt_path):
                    ckpt_path = os.path.join(checkpoint_dir, directory, "best.pt")
                
                if os.path.exists(ckpt_path):
                    model_checkpoints[model_name][seed] = ckpt_path
                    print(f"[Found] {model_name} seed {seed}: {ckpt_path}")
        
        # Evaluate each model and seed
        for model_name, seed_paths in model_checkpoints.items():
            for seed, ckpt_path in sorted(seed_paths.items()):
                if model_name not in all_seed_results:
                    all_seed_results[model_name] = {}
                
                for ds in datasets_to_eval:
                    print(f"\n{'='*60}")
                    print(f"Evaluating {model_name} (seed {seed}) on {ds.upper()}")
                    print(f"Checkpoint: {ckpt_path}")
                    print("=" * 60)
                    
                    # Use cached dataloader
                    val_loader = get_cached_dataloader(ds)
                    
                    result = evaluate_single_seed(
                        cfg, ckpt_path, ds, device, max_samples, val_loader=val_loader
                    )
                    
                    if ds not in all_seed_results[model_name]:
                        all_seed_results[model_name][ds] = []
                    all_seed_results[model_name][ds].append(result)
                    
                    # Print single-seed report
                    print_representation_report(
                        result["representation_stats"],
                        result["accuracies"],
                        result["cka_matrix"],
                        None,  # model not available after eval
                        dataset=ds,
                    )

    
    # If single checkpoint provided
    elif checkpoint_path and os.path.exists(checkpoint_path):
        model_name = f"{model_prefix}_seed1"
        all_seed_results[model_name] = {}
        
        for ds in datasets_to_eval:
            print(f"\n{'='*60}")
            print(f"Evaluating {model_name} on {ds.upper()}")
            print(f"Checkpoint: {checkpoint_path}")
            print("=" * 60)
            
            # Use cached dataloader
            val_loader = get_cached_dataloader(ds)
            
            result = evaluate_single_seed(
                cfg, checkpoint_path, ds, device, max_samples, val_loader=val_loader
            )
            
            all_seed_results[model_name][ds] = [result]
            
            # Print single-seed report
            print_representation_report(
                result["representation_stats"],
                result["accuracies"],
                result["cka_matrix"],
                None,
                dataset=ds,
            )
    
    # Generate efficiency plots if requested
    if generate_efficiency_plots and len(all_seed_results) > 0:
        # Flatten results for efficiency plotting
        flat_results = {}
        for model_name, dataset_results in all_seed_results.items():
            for ds, results in dataset_results.items():
                key = f"{model_name}_{ds}"
                flat_results[key] = results
        plot_efficiency_analysis(flat_results, output_dir)
    
    # Save JSON results
    json_results = save_json_results(
        all_seed_results, output_dir, cfg,
        argparse.Namespace(
            dataset=dataset,
            seeds=seeds,
            checkpoint_dir=checkpoint_dir,
            checkpoint=checkpoint_path,
        )
    )
    
    # Generate markdown report
    generate_markdown_report(
        all_seed_results, output_dir, cfg,
        argparse.Namespace(
            dataset=dataset,
            seeds=seeds,
            checkpoint_dir=checkpoint_dir,
            checkpoint=checkpoint_path,
            tsne=run_tsne,
        )
    )
    
    print(f"\n[Done] All outputs saved to: {output_dir}/")


def evaluate_single_checkpoint(
    cfg           : dict,
    checkpoint_path: str,
    dataset       : str = "cifar100",
    run_tsne      : bool = False,
    output_dir    : str = "eval_outputs",
    max_samples   : int = 5000,
):
    """
    Evaluate a single checkpoint on a single dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Device: {device}")
    print(f"[Eval] Dataset: {dataset.upper()}")
    
    # Load dataset
    val_loader = load_dataset(
        dataset=dataset,
        data_root=cfg["data"]["root"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )
    
    # Load model
    model = load_model(cfg, checkpoint_path, device, dataset)
    
    # Count FLOPs and params
    model_info = count_flops(model, input_size=(1, 3, 32, 32))
    print(f"[Model] Parameters: {model_info['params_M']:.2f}M, FLOPs: {model_info['flops_M']:.2f}M")
    
    # Evaluate accuracy
    print("\n[Eval] Computing accuracy...")
    accuracies = evaluate_accuracy(model, val_loader, device, dataset)
    
    # Extract representations
    print("\n[Eval] Extracting representations...")
    representations = extract_representations(model, val_loader, device, max_samples, dataset)
    
    # CKA
    print("\n[Eval] Computing CKA similarity matrix...")
    cka_matrix = compute_cka_matrix(representations)
    plot_cka_matrix(cka_matrix, os.path.join(output_dir, f"cka_matrix_{dataset}.png"))
    
    # Representation statistics
    stats = compute_representation_stats(representations)
    
    # Full report
    print_representation_report(stats, accuracies, cka_matrix, model, dataset)
    
    # t-SNE
    if run_tsne:
        print("\n[Eval] Computing t-SNE (this takes a few minutes)...")
        tsne_results = compute_tsne(representations)
        
        plot_tsne(
            representations, tsne_results,
            save_path=os.path.join(output_dir, f"tsne_{dataset}_coarse.png"),
            label_type="coarse",
            dataset=dataset,
        )
        plot_tsne(
            representations, tsne_results,
            save_path=os.path.join(output_dir, f"tsne_{dataset}_fine.png"),
            label_type="fine",
            dataset=dataset,
        )
    
    print(f"\n[Done] All outputs saved to: {output_dir}/")
    
    return {
        "accuracies": accuracies,
        "cka_matrix": cka_matrix,
        "representation_stats": stats,
        "model_info": model_info,
    }


# =============================================================================
# Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate Fractal Intelligence Network")
    
    # Dataset selection
    parser.add_argument(
        "--dataset", type=str, default="cifar100", choices=["cifar10", "cifar100", "both"],
        help="Dataset to evaluate on (default: cifar100)"
    )
    
    # Checkpoint paths
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint (best.pt)"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default=None,
        help="Directory containing seed-specific checkpoints"
    )
    parser.add_argument(
        "--model_prefix", type=str, default="fin",
        help="Model name prefix for checkpoint detection (default: fin)"
    )
    
    # Seeds
    parser.add_argument(
        "--seeds", type=str, default="1,2,3",
        help="Comma-separated list of seeds to evaluate (default: 1,2,3)"
    )
    
    # Visualization
    parser.add_argument(
        "--tsne", action="store_true",
        help="Compute and save t-SNE visualizations (takes ~5 min)"
    )
    
    # Comparison
    parser.add_argument(
        "--compare", type=str, default=None,
        help="Optional: path to a second checkpoint to compare against"
    )
    
    # Output
    parser.add_argument(
        "--output_dir", type=str, default="eval_outputs",
        help="Directory to save plots and reports"
    )
    parser.add_argument(
        "--max_samples", type=int, default=5000,
        help="Max samples for t-SNE and CKA (default: 5000)"
    )
    
    # Efficiency
    parser.add_argument(
        "--efficiency_plots", action="store_true",
        help="Generate efficiency analysis plots (Params/FLOPs vs Accuracy)"
    )
    
    # Config
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file"
    )
    
    args = parser.parse_args()
    
    # Parse seeds
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    # Handle both single checkpoint and multi-seed evaluation
    if args.checkpoint and not args.checkpoint_dir:
        # Single checkpoint evaluation
        if args.dataset == "both":
            # Evaluate on both datasets
            for ds in ["cifar100", "cifar10"]:
                ds_output = os.path.join(args.output_dir, ds)
                evaluate_single_checkpoint(
                    cfg=cfg,
                    checkpoint_path=args.checkpoint,
                    dataset=ds,
                    run_tsne=args.tsne,
                    output_dir=ds_output,
                    max_samples=args.max_samples,
                )
        else:
            evaluate_single_checkpoint(
                cfg=cfg,
                checkpoint_path=args.checkpoint,
                dataset=args.dataset,
                run_tsne=args.tsne,
                output_dir=args.output_dir,
                max_samples=args.max_samples,
            )
    else:
        # Multi-seed evaluation
        evaluate(
            cfg=cfg,
            checkpoint_path=args.checkpoint,
            run_tsne=args.tsne,
            compare_path=args.compare,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            dataset=args.dataset,
            seeds=seeds,
            checkpoint_dir=args.checkpoint_dir,
            model_prefix=args.model_prefix,
            generate_efficiency_plots=args.efficiency_plots,
        )


if __name__ == "__main__":
    main()