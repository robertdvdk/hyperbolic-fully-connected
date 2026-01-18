# cifar100_hyperbolic.py
from pathlib import Path
import sys
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Subset, DataLoader
import wandb

parent_dir = Path(__file__).parent
sys.path.insert(0, str(parent_dir.parent))
from layers import lorentz_resnet18, Lorentz


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataloaders(
    batch_size,
    data_dir,
    val_fraction=0.1,
    train_subset_fraction=1.0,
    seed=42,
):
    """
    Create CIFAR-100 train/val/test dataloaders.

    Args:
        batch_size: Batch size for all loaders
        data_dir: Directory to store/load CIFAR data
        val_fraction: Fraction of training set to use for validation (default 10%)
        train_subset_fraction: Fraction of training set to use (after val split) for faster sweeps
        seed: Random seed for reproducible splits

    Returns:
        train_loader, val_loader, test_loader
    """
    mean = (0.5074, 0.4867, 0.4411)
    std = (0.267, 0.256, 0.276)

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std)
    ])

    val_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std)
    ])

    # Load full training set (will be split into train/val)
    full_trainset = torchvision.datasets.CIFAR100(
        data_dir, train=True, download=True, transform=train_transform
    )

    # For validation, we need the same data but without augmentation
    full_trainset_val = torchvision.datasets.CIFAR100(
        data_dir, train=True, download=True, transform=val_transform
    )

    # Test set is completely separate
    testset = torchvision.datasets.CIFAR100(
        data_dir, train=False, download=True, transform=val_transform
    )

    # Create reproducible train/val split
    num_train_full = len(full_trainset)
    indices = list(range(num_train_full))

    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    val_size = int(num_train_full * val_fraction)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    # Optionally use only a subset of training data (for faster sweeps)
    if train_subset_fraction < 1.0:
        num_train_subset = int(len(train_indices) * train_subset_fraction)
        train_indices = train_indices[:num_train_subset]

    train_subset = Subset(full_trainset, train_indices)
    val_subset = Subset(full_trainset_val, val_indices)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )

    return train_loader, val_loader, test_loader


def train_epoch(model, train_loader, optimizer, device='cuda', grad_clip=1.0):
    """Train for one epoch, return avg loss and accuracy."""
    model.train()
    running_loss, total_correct, total_samples = 0.0, 0, 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x).squeeze()
        loss = F.cross_entropy(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += x.size(0)

    return running_loss / total_samples, total_correct / total_samples


def evaluate(model, loader, device='cuda'):
    """Evaluate on a dataset, return avg loss and accuracy."""
    model.eval()
    running_loss, total_correct, total_samples = 0.0, 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            logits = model(x).squeeze()
            loss = F.cross_entropy(logits, y, reduction='sum')

            running_loss += loss.item()
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_samples += x.size(0)

    return running_loss / total_samples, total_correct / total_samples


class EarlyStopping:
    """Early stopping based on validation loss."""

    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


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

    # Reproducibility
    seed_everything(get_config('seed', 0))
    device = 'cuda'

    # Data
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=get_config('batch_size', 128),
        data_dir="./data/cifar",
        val_fraction=get_config('val_fraction', 0.1),
        train_subset_fraction=get_config('train_subset_fraction', 1.0),
        seed=get_config('data_split_seed', 42),
    )

    # Model
    manifold = Lorentz(k=get_config('curvature', 1.0))
    model = lorentz_resnet18(
        num_classes=100,
        base_dim=get_config('hidden_dim', 64),
        manifold=manifold,
        activation=nn.ReLU,
        init_method=get_config('init_method', 'kaiming'),
    ).to(device)

    if get_config('compile', True):
        model = torch.compile(model)

    # Log model size
    total_params = sum(p.numel() for p in model.parameters())
    wandb.config.update({"total_params": total_params}, allow_val_change=True)

    # Optimizer
    optimizer_name = get_config('optimizer', 'adam').lower()
    lr = get_config('learning_rate', 1e-3)
    weight_decay = get_config('weight_decay', 0.0)

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_name == "sgd":
        momentum = get_config('momentum', 0.9)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Learning rate scheduler
    scheduler = None
    scheduler_type = get_config('scheduler', 'none').lower()
    num_epochs = get_config('num_epochs', 100)
    warmup_epochs = get_config('warmup_epochs', 0)

    if scheduler_type == 'steplr':
        from torch.optim.lr_scheduler import SequentialLR, MultiStepLR, LinearLR

        # Milestones at ~40%, 70%, 90% of training
        milestones = get_config('milestones', [int(num_epochs * 0.4), int(num_epochs * 0.7), int(num_epochs * 0.9)])
        gamma = get_config('lr_decay', 0.2)

        if warmup_epochs > 0:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_epochs
            )
            step_scheduler = MultiStepLR(
                optimizer,
                milestones=[m - warmup_epochs for m in milestones if m > warmup_epochs],
                gamma=gamma
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, step_scheduler],
                milestones=[warmup_epochs]
            )
        else:
            scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    elif scheduler_type == 'cosine':
        from torch.optim.lr_scheduler import SequentialLR, CosineAnnealingLR, LinearLR

        if warmup_epochs > 0:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_epochs
            )
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=num_epochs - warmup_epochs,
                eta_min=lr * 0.01
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs]
            )
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.01)

    # Early stopping
    early_stopping = None
    if get_config('early_stopping', False):
        early_stopping = EarlyStopping(
            patience=get_config('early_stopping_patience', 10),
            min_delta=get_config('early_stopping_min_delta', 0.0)
        )

    # Training loop
    best_val_acc = 0.0
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        start = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device,
            grad_clip=get_config('grad_clip', 1.0)
        )
        val_loss, val_acc = evaluate(model, val_loader, device)

        if scheduler:
            scheduler.step()

        epoch_time = time.time() - start

        # Log metrics
        metrics = {
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/acc": train_acc,
            "val/loss": val_loss,
            "val/acc": val_acc,
            "epoch_time": epoch_time,
            "learning_rate": optimizer.param_groups[0]['lr']
        }
        wandb.log(metrics)

        # Track best validation metrics
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            wandb.run.summary["best_val_acc"] = best_val_acc

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wandb.run.summary["best_val_loss"] = best_val_loss

        print(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)")
        print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.4f}")
        print(f"  Val:   loss={val_loss:.4f}, acc={val_acc:.4f}")

        # Early stopping check
        if early_stopping is not None:
            if early_stopping.step(val_loss):
                print(f"Early stopping triggered at epoch {epoch+1}")
                wandb.run.summary["early_stopped_epoch"] = epoch + 1
                break

    # Final test evaluation (only if not a sweep or if explicitly requested)
    if get_config('evaluate_test', False):
        test_loss, test_acc = evaluate(model, test_loader, device)
        wandb.run.summary["test_loss"] = test_loss
        wandb.run.summary["test_acc"] = test_acc
        print(f"Test: loss={test_loss:.4f}, acc={test_acc:.4f}")

    return best_val_acc


def main():
    """
    Entry point for both standalone runs and W&B sweeps.

    For sweeps: wandb.init() connects to the sweep and populates wandb.config
    For standalone: wandb.init() uses the default config below
    """
    default_config = {
        # Model
        "hidden_dim": 64,
        "curvature": 1.0,
        "init_method": "kaiming",

        # Optimization
        "optimizer": "adam",
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "momentum": 0.9,
        "batch_size": 128,
        "num_epochs": 100,
        "grad_clip": 1.0,

        # Scheduler
        "scheduler": "none",
        "warmup_epochs": 0,
        "lr_decay": 0.2,

        # Data
        "val_fraction": 0.1,
        "train_subset_fraction": 1.0,
        "data_split_seed": 42,

        # Early stopping
        "early_stopping": False,
        "early_stopping_patience": 10,

        # Misc
        "seed": 0,
        "compile": True,
        "evaluate_test": True,
    }

    # wandb.init() will use sweep config if run by wandb agent,
    # otherwise uses default_config
    wandb.init(
        project="ICML_Hyperbolic",
        config=default_config,
    )

    train(wandb.config)
    wandb.finish()


if __name__ == "__main__":
    main()
