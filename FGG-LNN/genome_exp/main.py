"""
Genome sequence classification training script.
Based on cifar_exp/main.py structure but adapted for genomic data.
"""

from pathlib import Path
import sys
import time
import math
import random
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import wandb
from sklearn.metrics import matthews_corrcoef
from geoopt.optim import RiemannianSGD, RiemannianAdam
from geoopt import ManifoldParameter
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from genome_exp.model import GenomeHyperbolicCNN, EuclideanCNN


# ============================================================================
# Data Loading (adapted from HGE/utils/data_utils.py)
# ============================================================================

def get_param_groups(model, lr_manifold, weight_decay_manifold):
    no_decay = ["scale"]
    k_params = ["manifold.k"]

    parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad
                and not any(nd in n for nd in no_decay)
                and not isinstance(p, ManifoldParameter)
                and not any(nd in n for nd in k_params)
            ],
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad
                and isinstance(p, ManifoldParameter)
            ],
            'lr' : lr_manifold,
            "weight_decay": weight_decay_manifold
        },
        {  # k parameters
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad
                and any(nd in n for nd in k_params)
            ], 
            "weight_decay": 0,
            "lr": 1e-4
        }
    ]

    return parameters


def coin_flip():
    return random.random() > 0.5


STRING_COMPLEMENT_MAP = {
    'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'a': 't', 'c': 'g', 'g': 'c', 't': 'a'
}


def string_reverse_complement(seq):
    rev_comp = ''
    for base in seq[::-1]:
        if base in STRING_COMPLEMENT_MAP:
            rev_comp += STRING_COMPLEMENT_MAP[base]
        else:
            rev_comp += base
    return rev_comp


class GenomeDataset(Dataset):
    """Dataset for Genomic Sequences."""

    BASE_TO_INT = {
        'A': 0, 'C': 1, 'T': 2, 'G': 3, 'N': 4,
        'a': 0, 'c': 1, 't': 2, 'g': 3, 'n': 4,
        # IUPAC ambiguity codes -> map to N (unknown)
        'R': 4, 'Y': 4, 'S': 4, 'W': 4, 'K': 4, 'M': 4,
        'B': 4, 'D': 4, 'H': 4, 'V': 4,
        'r': 4, 'y': 4, 's': 4, 'w': 4, 'k': 4, 'm': 4,
        'b': 4, 'd': 4, 'h': 4, 'v': 4,
    }

    def __init__(self, file_path: str, length: int, rc_aug: bool = False):
        super().__init__()

        with open(file_path, "r") as f:
            data = list(csv.reader(f))[1:]  # Skip header
            self.sequences = [d[0] for d in data]
            self.labels = [int(d[1]) for d in data]

        self.max_length = length
        self.rc_aug = rc_aug

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]

        # Data augmentation: random reverse-complement
        if self.rc_aug and coin_flip():
            seq = string_reverse_complement(seq)

        # One-hot encode
        one_hot = self._one_hot_encode(seq)

        return one_hot, torch.tensor(label, dtype=torch.long)

    def _one_hot_encode(self, sequence):
        # Truncate or pad to max_length
        sequence = sequence[:self.max_length]

        integer_encoded = [self.BASE_TO_INT.get(base, 4) for base in sequence]
        one_hot = np.zeros((5, self.max_length), dtype=np.float32)
        one_hot[integer_encoded, np.arange(len(sequence))] = 1

        return torch.from_numpy(one_hot)


def get_dataloaders(
    data_path: str,
    dataset_name: str,
    benchmark: str,
    length: int,
    batch_size: int,
    rc_aug: bool = False,
    num_workers: int = 2,
):
    """
    Create train/val/test dataloaders for genome data.

    Args:
        data_path: Base path to dataset directory
        dataset_name: Name of specific dataset
        benchmark: "GUE" or "TEB" format
        length: Sequence length (truncate/pad to this)
        batch_size: Batch size
        rc_aug: Whether to use reverse-complement augmentation
        num_workers: Number of data loader workers
    """
    if benchmark == "TEB":
        train_path = f"{data_path}/train_{dataset_name}.csv"
        val_path = f"{data_path}/valid_{dataset_name}.csv"
        test_path = f"{data_path}/test_{dataset_name}.csv"
    elif benchmark == "GUE":
        train_path = f"{data_path}/{dataset_name}/train.csv"
        val_path = f"{data_path}/{dataset_name}/dev.csv"
        test_path = f"{data_path}/{dataset_name}/test.csv"
    else:
        raise ValueError(f"Unknown benchmark format: {benchmark}")

    train_ds = GenomeDataset(train_path, length, rc_aug=rc_aug)
    val_ds = GenomeDataset(val_path, length, rc_aug=False)
    test_ds = GenomeDataset(test_path, length, rc_aug=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader


# ============================================================================
# Utility Functions
# ============================================================================

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_mcc(y_true, y_pred):
    """Compute Matthews Correlation Coefficient."""
    return matthews_corrcoef(y_true, y_pred)


def _tensor_has_nan_inf(t: torch.Tensor) -> bool:
    return not torch.isfinite(t).all().item()


def _any_nan_inf(obj) -> bool:
    if torch.is_tensor(obj):
        return _tensor_has_nan_inf(obj)
    if isinstance(obj, (list, tuple)):
        return any(_any_nan_inf(x) for x in obj)
    if isinstance(obj, dict):
        return any(_any_nan_inf(v) for v in obj.values())
    return False


def register_nan_checks(model: nn.Module):
    handles = []

    def hook(module, inputs, outputs):
        if _any_nan_inf(outputs):
            raise RuntimeError(f"NaN/Inf detected in forward output of module: {module.__class__.__name__}")

    for m in model.modules():
        handles.append(m.register_forward_hook(hook))

    return handles


def check_model_finiteness(model: nn.Module, *, check_grads: bool = False):
    for name, param in model.named_parameters():
        if param is None:
            continue
        if param.data is not None and _tensor_has_nan_inf(param.data):
            raise RuntimeError(f"NaN/Inf detected in parameter: {name}")
        if check_grads and param.grad is not None and _tensor_has_nan_inf(param.grad):
            raise RuntimeError(f"NaN/Inf detected in gradient: {name}")


class EarlyStopping:
    """Early stopping based on validation metric."""

    def __init__(self, patience=10, min_delta=0.0, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = float('-inf') if mode == 'max' else float('inf')
        self.counter = 0
        self.should_stop = False

    def step(self, score):
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(
    model,
    train_loader,
    optimizer,
    device='cuda',
    grad_clip=1.0,
    nan_check: bool = False,
    epoch_idx: int = 0,
):
    """Train for one epoch, return avg loss, accuracy, and MCC."""
    model.train()
    running_loss, total_correct, total_samples = 0.0, 0, 0
    all_preds, all_labels = [], []

    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)

        if nan_check:
            if _any_nan_inf(x):
                raise RuntimeError(
                    f"NaN/Inf detected in input batch (epoch {epoch_idx}, batch {batch_idx})"
                )
            if _any_nan_inf(y):
                raise RuntimeError(
                    f"NaN/Inf detected in labels (epoch {epoch_idx}, batch {batch_idx})"
                )

        optimizer.zero_grad()
        logits = model(x)
        if nan_check and _any_nan_inf(logits):
            raise RuntimeError(
                f"NaN/Inf detected in logits (epoch {epoch_idx}, batch {batch_idx})"
            )
        loss = F.cross_entropy(logits, y)
        if nan_check and _any_nan_inf(loss):
            raise RuntimeError(
                f"NaN/Inf detected in loss (epoch {epoch_idx}, batch {batch_idx})"
            )
        loss.backward()

        if nan_check:
            check_model_finiteness(model, check_grads=True)

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        if nan_check:
            check_model_finiteness(model, check_grads=False)

        preds = logits.argmax(dim=1)
        running_loss += loss.item() * x.size(0)
        total_correct += (preds == y).sum().item()
        total_samples += x.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    avg_loss = running_loss / total_samples
    accuracy = total_correct / total_samples
    mcc = compute_mcc(all_labels, all_preds)

    return avg_loss, accuracy, mcc


def evaluate(model, loader, device='cuda', nan_check: bool = False, epoch_idx: int = 0):
    """Evaluate on a dataset, return avg loss, accuracy, and MCC."""
    model.eval()
    running_loss, total_correct, total_samples = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)

            if nan_check:
                if _any_nan_inf(x):
                    raise RuntimeError(
                        f"NaN/Inf detected in eval input (epoch {epoch_idx}, batch {batch_idx})"
                    )
                if _any_nan_inf(y):
                    raise RuntimeError(
                        f"NaN/Inf detected in eval labels (epoch {epoch_idx}, batch {batch_idx})"
                    )

            logits = model(x)
            if nan_check and _any_nan_inf(logits):
                raise RuntimeError(
                    f"NaN/Inf detected in eval logits (epoch {epoch_idx}, batch {batch_idx})"
                )
            loss = F.cross_entropy(logits, y, reduction='sum')
            if nan_check and _any_nan_inf(loss):
                raise RuntimeError(
                    f"NaN/Inf detected in eval loss (epoch {epoch_idx}, batch {batch_idx})"
                )
            preds = logits.argmax(dim=1)

            running_loss += loss.item()
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = running_loss / total_samples
    accuracy = total_correct / total_samples
    mcc = compute_mcc(all_labels, all_preds)

    return avg_loss, accuracy, mcc


# ============================================================================
# Main Training Function
# ============================================================================

def train(config=None):
    """Main training function - callable by W&B sweeps."""
    if config is None:
        config = wandb.config

    def get_config(key, default=None):
        if hasattr(config, key):
            return getattr(config, key)
        elif isinstance(config, dict):
            return config.get(key, default)
        else:
            return default

    def resolve_length(dataset_name: str) -> int:
        if not dataset_name:
            raise ValueError("Specify the length")
        name = dataset_name.lower()
        name_sp = name.replace("_", " ")

        if "pseudogene" in name_sp:
            return 1000

        mapping = {
            "dna_cmc": 200,
            "dna hat ac": 1000,
            "sines": 500,
            "lines": 1000,
            "ltr copia": 500,
        }
        if name in mapping:
            return mapping[name]
        else: # name_sp in mapping:
            return mapping[name_sp]

    def resolve_bn_mode(mode: str):
        mode = (mode or "normal").lower()
        if mode == "centering_only":
            return {
                "fix_gamma": True,
                "normalize_variance": False,
                "clamp_scale": False,
            }
        if mode == "clamp_scale":
            return {
                "fix_gamma": get_config("fix_gamma", False),
                "normalize_variance": True,
                "clamp_scale": True,
            }
        return {
            "fix_gamma": get_config("fix_gamma", False),
            "normalize_variance": True,
            "clamp_scale": False,
        }

    # Reproducibility
    seed_everything(get_config('seed', 42))
    device = get_config('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    # Data
    data_path = get_config('data_path', './HGE/GUE')
    length = get_config('length', 0)
    if length == 0:
        length = resolve_length(get_config('dataset_name', ''))

    train_loader, val_loader, test_loader = get_dataloaders(
        data_path=data_path,
        dataset_name=get_config('dataset_name', 'prom_300_all'),
        benchmark=get_config('benchmark', 'GUE'),
        length=length,
        batch_size=get_config('batch_size', 256),
        rc_aug=get_config('rc_aug', False),
        num_workers=get_config('num_workers', 2),
    )

    # Model
    manifold_type = get_config('manifold', 'lorentz')
    bn_mode_cfg = resolve_bn_mode(get_config("bn_mode", "normal"))

    if manifold_type == 'lorentz':
        model = GenomeHyperbolicCNN(
            num_classes=get_config('num_classes', 2),
            length=length,
            model_dim=get_config('num_channels', 32),
            fc_dim=get_config('embedding_dim', 528),
            num_layers=get_config('num_layers', 3),
            kernel_size=get_config('kernel_size', 9),
            learnable_k=get_config('learnable_k', False),
            k=get_config('curvature', 1.0),
            use_bn=get_config('use_bn', True),
            fix_gamma=bn_mode_cfg["fix_gamma"],
            clamp_scale=bn_mode_cfg["clamp_scale"],
            normalize_variance=bn_mode_cfg["normalize_variance"],
            mlr_type=get_config('mlr_type', 'fc_mlr'),
            use_weight_norm=get_config('use_weight_norm', False),
        ).to(device)
    else:
        model = EuclideanCNN(
            num_classes=get_config('num_classes', 2),
            length=length,
            model_dim=get_config('num_channels', 32),
            fc_dim=get_config('embedding_dim', 528),
            num_layers=get_config('num_layers', 3),
            kernel_size=get_config('kernel_size', 9),
        ).to(device)

    # Optional NaN/Inf debugging
    if get_config("nan_debug", False):
        torch.autograd.set_detect_anomaly(True)
        _nan_handles = register_nan_checks(model)

    # Log model size
    total_params = sum(p.numel() for p in model.parameters())
    wandb.config.update({"total_params": total_params}, allow_val_change=True)

    # Compile model if available
    if get_config('compile', False) and hasattr(torch, 'compile'):
        model = torch.compile(model)

    # Optimizer
    optimizer_name = get_config('optimizer', 'adam').lower()
    lr = get_config('learning_rate', 1e-4)
    weight_decay = get_config('weight_decay', 0.1)

    model_parameters = get_param_groups(model, get_config('manifold_lr', 2e-2), get_config('manifold_weight_decay', 5e-4))

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model_parameters,
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_name == "riemannian_adam":
        optimizer = RiemannianAdam(
            model_parameters,
            lr=lr,
            weight_decay=weight_decay,
            stabilize=1
        )
    elif optimizer_name == "sgd":
        momentum = get_config('momentum', 0.9)
        optimizer = torch.optim.SGD(
            model_parameters,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True
        )
    elif optimizer_name == "riemannian_sgd":
        momentum = get_config('momentum', 0.9)
        optimizer = RiemannianSGD(
            model_parameters,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True,
            stabilize=1
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Learning rate scheduler
    scheduler_type = get_config('scheduler', 'multistep').lower()
    num_epochs = get_config('num_epochs', 150)
    scheduler = None

    if scheduler_type == 'multistep':
        from torch.optim.lr_scheduler import MultiStepLR
        milestones = get_config('milestones', [60, 85])
        gamma = get_config('lr_decay', 0.1)
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    elif scheduler_type == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.01)

    elif scheduler_type == 'steplr':
        from torch.optim.lr_scheduler import StepLR
        step_size = get_config('step_size', 30)
        gamma = get_config('lr_decay', 0.1)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Early stopping
    early_stopping = None
    if get_config('early_stopping', False):
        early_stopping = EarlyStopping(
            patience=get_config('early_stopping_patience', 15),
            min_delta=0.0,
            mode='max'  # Maximize MCC
        )

    # Checkpointing
    checkpoint_dir = Path(get_config('checkpoint_dir', './checkpoints'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"best_model_{wandb.run.id}.pt"

    # Training loop
    best_val_mcc = -1.0

    for epoch in range(num_epochs):
        start = time.time()

        train_loss, train_acc, train_mcc = train_epoch(
            model, train_loader, optimizer, device,
            grad_clip=get_config('grad_clip', 1.0),
            nan_check=get_config("nan_debug", False),
            epoch_idx=epoch + 1,
        )
        val_loss, val_acc, val_mcc = evaluate(
            model,
            val_loader,
            device,
            nan_check=get_config("nan_debug", False),
            epoch_idx=epoch + 1,
        )

        if scheduler:
            scheduler.step()

        epoch_time = time.time() - start

        # Check for NaN
        if not all(map(math.isfinite, [train_loss, val_loss])):
            msg = f"NaN/Inf detected at epoch {epoch + 1}"
            print(msg)
            wandb.run.summary["nan_detected"] = True
            wandb.run.summary["nan_epoch"] = epoch + 1
            wandb.finish(exit_code=1)
            raise RuntimeError(msg)

        # Log metrics
        metrics = {
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/acc": train_acc,
            "train/mcc": train_mcc,
            "val/loss": val_loss,
            "val/acc": val_acc,
            "val/mcc": val_mcc,
            "epoch_time": epoch_time,
            "learning_rate": optimizer.param_groups[0]['lr']
        }
        wandb.log(metrics)

        # Track best model (by MCC)
        if val_mcc > best_val_mcc:
            best_val_mcc = val_mcc
            wandb.run.summary["best_val_mcc"] = best_val_mcc

            # Save checkpoint
            if isinstance(config, dict):
                config_dict = config
            elif hasattr(config, '_items'):
                config_dict = dict(config._items)
            else:
                config_dict = dict(config)

            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mcc': val_mcc,
                'val_acc': val_acc,
                'config': config_dict
            }
            if scheduler:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()

            torch.save(checkpoint, checkpoint_path)
            print(f"  -> Saved checkpoint (best val_mcc: {val_mcc:.4f})")

        print(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)")
        print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.4f}, mcc={train_mcc:.4f}")
        print(f"  Val:   loss={val_loss:.4f}, acc={val_acc:.4f}, mcc={val_mcc:.4f}")

        # Early stopping
        if early_stopping and early_stopping.step(val_mcc):
            print(f"Early stopping triggered at epoch {epoch+1}")
            wandb.run.summary["early_stopped_epoch"] = epoch + 1
            break

    # Final test evaluation
    if get_config('evaluate_test', True):
        # Load best model
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        test_loss, test_acc, test_mcc = evaluate(model, test_loader, device)
        wandb.run.summary["test_loss"] = test_loss
        wandb.run.summary["test_acc"] = test_acc
        wandb.run.summary["test_mcc"] = test_mcc
        print(f"Test: loss={test_loss:.4f}, acc={test_acc:.4f}, mcc={test_mcc:.4f}")

    return best_val_mcc


# ============================================================================
# Multi-Dataset Training
# ============================================================================

# All TEB datasets with their sequence lengths
TEB_DATASETS = {
    "processed_pseudogenes": 1000,
    "unprocessed_pseudogenes": 1000,
    "sines": 500,
    "lines": 1000,
    "ltr_copia": 500,
    "dna_cmc": 200,
    "dna_hat_ac": 1000,
}


def train_all_datasets(config):
    """
    Train on all TEB datasets and compute aggregate metrics.

    This function trains a fresh model on each dataset with the same
    hyperparameters, then computes aggregate metrics for hyperparameter
    selection.

    Returns:
        dict: Results containing per-dataset and aggregate metrics
    """
    def get_config(key, default=None):
        if hasattr(config, key):
            return getattr(config, key)
        elif isinstance(config, dict):
            return config.get(key, default)
        else:
            return default

    results = {}
    val_mccs = []
    test_mccs = []

    for dataset_name, length in TEB_DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Training on dataset: {dataset_name}")
        print(f"{'='*60}\n")

        # Create a modified config for this dataset
        dataset_config = dict(config) if isinstance(config, dict) else dict(config._items)
        dataset_config['dataset_name'] = dataset_name
        dataset_config['length'] = length

        # Train on this dataset
        try:
            best_val_mcc = train(dataset_config)

            # Get test MCC from wandb summary
            test_mcc = wandb.run.summary.get(f"test_mcc", 0.0)

            results[dataset_name] = {
                'val_mcc': best_val_mcc,
                'test_mcc': test_mcc,
            }
            val_mccs.append(best_val_mcc)
            test_mccs.append(test_mcc)

            # Log per-dataset metrics
            wandb.log({
                f"dataset/{dataset_name}/val_mcc": best_val_mcc,
                f"dataset/{dataset_name}/test_mcc": test_mcc,
            })

        except Exception as e:
            print(f"Error training on {dataset_name}: {e}")
            results[dataset_name] = {'val_mcc': 0.0, 'test_mcc': 0.0, 'error': str(e)}
            val_mccs.append(0.0)
            test_mccs.append(0.0)

    # Compute aggregate metrics
    if val_mccs:
        # Mean MCC
        mean_val_mcc = sum(val_mccs) / len(val_mccs)
        mean_test_mcc = sum(test_mccs) / len(test_mccs)

        # Min MCC (worst-case performance)
        min_val_mcc = min(val_mccs)
        min_test_mcc = min(test_mccs)

        # Harmonic mean (penalizes low values more)
        def harmonic_mean(values):
            # Shift by small epsilon to avoid division by zero for negative MCCs
            shifted = [max(v + 1.0, 1e-6) for v in values]  # MCC in [-1, 1] -> [0, 2]
            hm = len(shifted) / sum(1.0/v for v in shifted)
            return hm - 1.0  # Shift back

        harmonic_val_mcc = harmonic_mean(val_mccs)
        harmonic_test_mcc = harmonic_mean(test_mccs)

        # Log aggregate metrics
        wandb.log({
            "aggregate/mean_mcc": mean_val_mcc,
            "aggregate/min_mcc": min_val_mcc,
            "aggregate/harmonic_mcc": harmonic_val_mcc,
            "aggregate/mean_test_mcc": mean_test_mcc,
            "aggregate/min_test_mcc": min_test_mcc,
            "aggregate/harmonic_test_mcc": harmonic_test_mcc,
        })

        # Also set in summary for sweep optimization
        wandb.run.summary["aggregate/mean_mcc"] = mean_val_mcc
        wandb.run.summary["aggregate/min_mcc"] = min_val_mcc
        wandb.run.summary["aggregate/harmonic_mcc"] = harmonic_val_mcc

        print(f"\n{'='*60}")
        print("AGGREGATE RESULTS")
        print(f"{'='*60}")
        print(f"Mean Val MCC:     {mean_val_mcc:.4f}")
        print(f"Min Val MCC:      {min_val_mcc:.4f}")
        print(f"Harmonic Val MCC: {harmonic_val_mcc:.4f}")
        print(f"Mean Test MCC:    {mean_test_mcc:.4f}")

    return results


def main():
    """Entry point for standalone runs and W&B sweeps."""
    default_config = {
        # Data
        "data_path": "./TEB/",
        "dataset_name": "processed_pseudogenes",
        "benchmark": "TEB",
        "length": 0,
        "num_classes": 2,
        "rc_aug": False,

        # Model
        "manifold": "lorentz",  # "lorentz" or "euclidean"
        "num_channels": 32,
        "embedding_dim": 528,
        "num_layers": 3,
        "kernel_size": 9,
        "curvature": 1.0,
        "learnable_k": False,
        "use_bn": True,
        "fix_gamma": False,
        "mlr_type": "fc_mlr",  # "lorentz_mlr" or "fc_mlr"
        "use_weight_norm": True,
        "bn_mode": "centering_only",

        # Optimization
        "optimizer": "riemannian_adam",
        "learning_rate": 1e-4,
        "weight_decay": 0.1,
        "momentum": 0.9,
        "batch_size": 100,
        "num_epochs": 100,
        "grad_clip": 1.0,
        "manifold_lr": 2e-2,
        "manifold_weight_decay": 5e-4,

        # Scheduler
        "scheduler": "multistep",
        "milestones": [60, 85],
        "lr_decay": 0.1,

        # Early stopping
        "early_stopping": False,
        "early_stopping_patience": 15,

        # Misc
        "seed": 42,
        "device": "cuda",
        "compile": True,
        "evaluate_test": True,
        "checkpoint_dir": "./checkpoints",
        "num_workers": 2,
        "nan_debug": False,
    }

    wandb.init(
        project="Genome_Hyperbolic",
        config=default_config
    )

    # Check if we should train on all datasets
    dataset_name = wandb.config.get('dataset_name', default_config['dataset_name'])
    if dataset_name == "all":
        train_all_datasets(wandb.config)
    else:
        train(wandb.config)

    wandb.finish()


if __name__ == "__main__":
    main()
