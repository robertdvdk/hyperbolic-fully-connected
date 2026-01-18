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


def get_dataloaders(batch_size=128, data_dir="./cifar"):
    """Create CIFAR-100 train/val dataloaders with standard augmentations."""
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
    
    trainset = torchvision.datasets.CIFAR100(
        data_dir, train=True, download=True, transform=train_transform
    )
    valset = torchvision.datasets.CIFAR100(
        data_dir, train=False, download=True, transform=val_transform
    )
    
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    
    return train_loader, val_loader


def train_epoch(model, train_loader, optimizer, device='cuda'):
    """Train for one epoch, return avg loss and accuracy."""
    model.train()
    running_loss, total_correct, total_samples = 0.0, 0, 0
    
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(x).squeeze()
        loss = F.cross_entropy(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        running_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += x.size(0)
    
    return running_loss / total_samples, total_correct / total_samples


def validate(model, val_loader, device='cuda'):
    """Evaluate on validation set, return avg loss and accuracy."""
    model.eval()
    running_loss, total_correct, total_samples = 0.0, 0, 0
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            
            logits = model(x).squeeze()
            loss = F.cross_entropy(logits, y, reduction='sum')
            
            running_loss += loss.item()
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_samples += x.size(0)
    
    return running_loss / total_samples, total_correct / total_samples


def train(config=None):
    """Main training function - callable by W&B sweeps."""
    if config is None:
        # Get config from W&B
        config = wandb.config
    
    # Helper to get values from either dict or wandb.config
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
    train_loader, val_loader = get_dataloaders(
        batch_size=get_config('batch_size', 128),
        data_dir="./cifar"
    )
    
    # Model
    manifold = Lorentz(k=get_config('curvature', 1.0))
    model = lorentz_resnet18(
        num_classes=100,
        base_dim=get_config('hidden_dim', 64),
        manifold=manifold,
        activation=nn.ReLU
    ).to(device)
    model.compile()
    
    # Log model size
    total_params = sum(p.numel() for p in model.parameters())
    wandb.config.update({"total_params": total_params}, allow_val_change=True)
    
    # Optimizer
    if get_config('optimizer', 'adam') == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=get_config('learning_rate', 1e-3),
            weight_decay=get_config('weight_decay', 0.0)
        )
    elif get_config('optimizer', 'adam') == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=get_config('learning_rate', 1e-3),
            momentum=get_config('momentum', 0.9),
            weight_decay=get_config('weight_decay', 0.0)
        )
    
    # Learning rate scheduler (optional)
    scheduler = None
    if get_config('use_scheduler', False):
        from torch.optim.lr_scheduler import SequentialLR, MultiStepLR, LinearLR
        warmup_epochs = get_config('warmup_epochs', 10)
        milestones = get_config('milestones', [60, 120, 160])
        
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        step_scheduler = MultiStepLR(
            optimizer,
            milestones=[m - warmup_epochs for m in milestones],
            gamma=get_config('lr_decay', 0.2)
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, step_scheduler],
            milestones=[warmup_epochs]
        )
    
    # Training loop
    best_val_acc = 0.0
    num_epochs = get_config('num_epochs', 100)
    
    for epoch in range(num_epochs):
        start = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, device)
        
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
        
        # Track best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            wandb.run.summary["best_val_acc"] = best_val_acc
        
        print(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)")
        print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.4f}")
        print(f"  Val:   loss={val_loss:.4f}, acc={val_acc:.4f}")


def main():
    """Run a single training run with default config."""
    config = {
        # Model
        "hidden_dim": 64,  # Changed from 16 to match ResNet18
        "curvature": 1.0,
        
        # Optimization
        "optimizer": "adam",
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "momentum": 0.9,  # for SGD
        "batch_size": 128,
        "num_epochs": 100,
        
        # Scheduler
        "use_scheduler": False,
        "warmup_epochs": 10,
        "milestones": [60, 120, 160],
        "lr_decay": 0.2,
        
        # Misc
        "seed": 0,
    }
    
    wandb.init(
        project="hyperbolic-cifar100",
        config=config,
        name="baseline-run"
    )
    
    train(wandb.config)  # Pass wandb.config instead of dict


if __name__ == "__main__":
    main()