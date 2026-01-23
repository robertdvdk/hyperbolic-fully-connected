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
import torchvision
from geoopt import ManifoldParameter
from geoopt.optim import RiemannianSGD

parent_dir = Path(__file__).parent
sys.path.insert(0, str(parent_dir.parent))
from layers import lorentz_resnet18, Lorentz


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

def select_optimizer(model, lr, weight_decay, warmup_epochs):
    """ Selects and sets up an available optimizer and returns it. """

    model_parameters = get_param_groups(model, lr*0.2, weight_decay)
    optimizer = RiemannianSGD(model_parameters, lr=lr, weight_decay=weight_decay, momentum=0.9, nesterov=True, stabilize=1)
    from torch.optim.lr_scheduler import SequentialLR, MultiStepLR, LinearLR
    warmup_scheduler = LinearLR(optimizer,
                                start_factor=0.01,
                                end_factor=1.0,
                                total_iters=warmup_epochs)
    step_scheduler = MultiStepLR(
        optimizer, milestones=[m - warmup_epochs for m in [60, 120, 160]], gamma=0.2
    )
    lr_scheduler = SequentialLR(optimizer,
                                schedulers=[warmup_scheduler, step_scheduler],
                                milestones=[warmup_epochs])
    
    return optimizer, lr_scheduler

    # if args.optimizer == "RiemannianSGD":
    #     optimizer = RiemannianSGD(model_parameters, lr=args.lr, weight_decay=args.weight_decay, momentum=0.9, nesterov=True, stabilize=1)
    # elif args.optimizer == "SGD":
    #     optimizer = torch.optim.SGD(model_parameters, lr=args.lr, weight_decay=args.weight_decay, momentum=0.9, nesterov=True)
    # else:
    #     raise "Optimizer not found. Wrong optimizer in configuration... -> " + args.model

    # lr_scheduler = None
    # if args.scheduler == "cosine":
    #     from torch.optim.lr_scheduler import SequentialLR, CosineAnnealingLR, LinearLR
    #     if args.warmup_epochs > 0:
    #         warmup_scheduler = LinearLR(optimizer,
    #                                     start_factor = 0.01,
    #                                     end_factor=1.0,
    #                                     total_iters=args.warmup_epochs)
    #         cosine_scheduler = CosineAnnealingLR(optimizer,
    #                                              T_max = args.num_epochs - args.warmup_epochs,
    #                                              eta_min=0)
    #         lr_scheduler = SequentialLR(optimizer,
    #                                     schedulers=[warmup_scheduler, cosine_scheduler],
    #                                     milestones=[args.warmup_epochs])
    #     else:
    #         lr_scheduler = CosineAnnealingLR(
    #             optimizer,
    #             T_max=args.num_epochs,
    #             eta_min=0  # LR goes down to 0
    #         )
    # else:
    #     from torch.optim.lr_scheduler import SequentialLR, MultiStepLR, LinearLR
    #     if args.warmup_epochs > 0:
    #         warmup_scheduler = LinearLR(optimizer,
    #                                     start_factor=0.01,
    #                                     end_factor=1.0,
    #                                     total_iters=args.warmup_epochs)
    #         step_scheduler = MultiStepLR(
    #             optimizer, milestones=[m - args.warmup_epochs for m in [60, 120, 160]], gamma=0.2
    #         )
    #         lr_scheduler = SequentialLR(optimizer,
    #                                     schedulers=[warmup_scheduler, step_scheduler],
    #                                     milestones=[args.warmup_epochs])
    #     else:
    #         lr_scheduler = MultiStepLR(
    #             optimizer, milestones=[60, 120, 160], gamma=0.2
    #         )
        

    # return optimizer, lr_scheduler

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device='cuda'):
    """
    Load a checkpoint and restore model, optimizer, and scheduler states.

    Returns:
        start_epoch: The epoch to resume from
        checkpoint: The full checkpoint dict for inspection
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle compiled models (state dict keys may have '_orig_mod.' prefix)
    state_dict = checkpoint['model_state_dict']

    # Try loading directly first
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # If model is compiled, keys might have _orig_mod prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('_orig_mod.', '')
            new_state_dict[new_key] = v
        model.load_state_dict(new_state_dict)

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint.get('epoch', 0)
    print(f"Loaded checkpoint from epoch {start_epoch}")
    print(f"  Val loss: {checkpoint.get('val_loss', 'N/A')}")
    print(f"  Val acc: {checkpoint.get('val_acc', 'N/A')}")

    return start_epoch, checkpoint


def inspect_model(model, dataloader, device='cuda'):
    """
    Inspect model state for debugging NaN issues.
    Checks U_norm values in all LorentzFullyConnected layers and logit magnitudes.
    """
    model.eval()
    print("\n" + "="*60)
    print("MODEL INSPECTION")
    print("="*60)

    # Check U_norm in all layers
    print("\n--- U_norm values in LorentzFullyConnected layers ---")
    for name, module in model.named_modules():
        if hasattr(module, 'U'):
            U_norm = module.U.norm(dim=0)
            print(f"{name}:")
            print(f"  U_norm: min={U_norm.min().item():.6f}, max={U_norm.max().item():.6f}, mean={U_norm.mean().item():.6f}")
            if hasattr(module, 'a'):
                a_vals = module.a
                print(f"  a: min={a_vals.min().item():.6f}, max={a_vals.max().item():.6f}, mean={a_vals.mean().item():.6f}")
                # Check effective bias magnitude (a / U_norm)
                effective = a_vals / U_norm
                print(f"  a/U_norm: min={effective.min().item():.6f}, max={effective.max().item():.6f}")

    # Check z and a in LorentzMLR
    print("\n--- LorentzMLR parameters ---")
    for name, module in model.named_modules():
        if hasattr(module, 'z') and hasattr(module, 'a') and not hasattr(module, 'U'):
            print(f"{name}:")
            print(f"  z norm: min={module.z.norm(dim=-1).min().item():.6f}, max={module.z.norm(dim=-1).max().item():.6f}")
            print(f"  a: min={module.a.min().item():.6f}, max={module.a.max().item():.6f}")

    # Forward pass to check activations
    print("\n--- Forward pass inspection ---")
    with torch.no_grad():
        x, y = next(iter(dataloader))
        x = x.to(device)

        # Get intermediate activations
        x_proj = model.input_proj(x)
        print(f"After input_proj: time=[{x_proj[:,0].min().item():.2f}, {x_proj[:,0].max().item():.2f}], "
              f"space_norm={x_proj[:,1:].norm(dim=1).mean().item():.2f}")

        x1 = model.stage1(x_proj)
        print(f"After stage1: time=[{x1[:,0].min().item():.2f}, {x1[:,0].max().item():.2f}], "
              f"space_norm={x1[:,1:].norm(dim=1).mean().item():.2f}")

        x2 = model.stage2(x1)
        print(f"After stage2: time=[{x2[:,0].min().item():.2f}, {x2[:,0].max().item():.2f}], "
              f"space_norm={x2[:,1:].norm(dim=1).mean().item():.2f}")

        x3 = model.stage3(x2)
        print(f"After stage3: time=[{x3[:,0].min().item():.2f}, {x3[:,0].max().item():.2f}], "
              f"space_norm={x3[:,1:].norm(dim=1).mean().item():.2f}")

        x4 = model.stage4(x3)
        print(f"After stage4: time=[{x4[:,0].min().item():.2f}, {x4[:,0].max().item():.2f}], "
              f"space_norm={x4[:,1:].norm(dim=1).mean().item():.2f}")

        x_pool = model._global_pool(x4)
        print(f"After pool: time=[{x_pool[:,0].min().item():.2f}, {x_pool[:,0].max().item():.2f}], "
              f"space_norm={x_pool[:,1:].norm(dim=-1).mean().item():.2f}")

        logits = model.classifier(x_pool)
        print(f"Logits: min={logits.min().item():.2f}, max={logits.max().item():.2f}, "
              f"std={logits.std().item():.2f}, mean_abs={logits.abs().mean().item():.2f}")

    print("="*60 + "\n")
    model.train()


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
        # torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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
        init_method=get_config('init_method', 'lorentz_kaiming'),
        input_proj_type=get_config('input_proj_type', 'conv_bn_relu'),
        mlr_init=get_config('mlr_init', 'mlr'),
    ).to(device)

    # Log model size
    total_params = sum(p.numel() for p in model.parameters())
    wandb.config.update({"total_params": total_params}, allow_val_change=True)

    # Optimizer
    optimizer_name = get_config('optimizer', 'adam').lower()
    lr = get_config('learning_rate', 1e-3)
    weight_decay = get_config('weight_decay', 0.0)
    warmup_epochs = get_config('warmup_epochs', 0)
    num_epochs = get_config('num_epochs', 100)
    optimizer, scheduler = select_optimizer(model, lr=lr, weight_decay=weight_decay, warmup_epochs=warmup_epochs)

    # Load checkpoint if specified
    start_epoch = 0
    checkpoint_path_load = get_config("resume_checkpoint", None)
    if checkpoint_path_load:
        start_epoch, ckpt = load_checkpoint(
            checkpoint_path_load, model, optimizer, scheduler, device
        )
        # Inspect mode: just analyze and exit
        if get_config('inspect_only', False):
            inspect_model(model, train_loader, device)
            return 0.0

    if get_config('compile', True):
        model = torch.compile(model)

    # if optimizer_name == "adam":
    #     optimizer = torch.optim.Adam(
    #         model.parameters(),
    #         lr=lr,
    #         weight_decay=weight_decay
    #     )
    # elif optimizer_name == "sgd":
    #     momentum = get_config('momentum', 0.9)
    #     optimizer = torch.optim.SGD(
    #         model.parameters(),
    #         lr=lr,
    #         momentum=momentum,
    #         weight_decay=weight_decay,
    #         nesterov=True
    #     )
    #     # optimizer = RiemannianSGD(params=model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True, stabilize=1)
    # else:
    #     raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Learning rate scheduler

    # optimizer, scheduler = select_optimizer(model, lr=lr, weight_decay=weight_decay, warmup_epochs=warmup_epochs)
    # scheduler = None
    # scheduler_type = get_config('scheduler', 'none').lower()
    # num_epochs = get_config('num_epochs', 100)
    # warmup_epochs = get_config('warmup_epochs', 0)

    # if scheduler_type == 'steplr':
    #     from torch.optim.lr_scheduler import SequentialLR, MultiStepLR, LinearLR

    #     # Milestones at ~40%, 70%, 90% of training
    #     milestones = get_config('milestones', [int(num_epochs * 0.3), int(num_epochs * 0.6), int(num_epochs * 0.8)])
    #     gamma = get_config('lr_decay', 0.2)

    #     if warmup_epochs > 0:
    #         warmup_scheduler = LinearLR(
    #             optimizer,
    #             start_factor=0.01,
    #             end_factor=1.0,
    #             total_iters=warmup_epochs
    #         )
    #         step_scheduler = MultiStepLR(
    #             optimizer,
    #             milestones=[m - warmup_epochs for m in milestones if m > warmup_epochs],
    #             gamma=gamma
    #         )
    #         scheduler = SequentialLR(
    #             optimizer,
    #             schedulers=[warmup_scheduler, step_scheduler],
    #             milestones=[warmup_epochs]
    #         )
    #     else:
    #         scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    # elif scheduler_type == 'cosine':
    #     from torch.optim.lr_scheduler import SequentialLR, CosineAnnealingLR, LinearLR

    #     if warmup_epochs > 0:
    #         warmup_scheduler = LinearLR(
    #             optimizer,
    #             start_factor=0.01,
    #             end_factor=1.0,
    #             total_iters=warmup_epochs
    #         )
    #         cosine_scheduler = CosineAnnealingLR(
    #             optimizer,
    #             T_max=num_epochs - warmup_epochs,
    #             eta_min=lr * 0.01
    #         )
    #         scheduler = SequentialLR(
    #             optimizer,
    #             schedulers=[warmup_scheduler, cosine_scheduler],
    #             milestones=[warmup_epochs]
    #         )
    #     else:
    #         scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.01)

    # Early stopping
    early_stopping = None
    if get_config('early_stopping', False):
        early_stopping = EarlyStopping(
            patience=get_config('early_stopping_patience', 10),
            min_delta=get_config('early_stopping_min_delta', 0.0)
        )

    # Create checkpoint directory
    checkpoint_dir = Path(get_config('checkpoint_dir', './checkpoints'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"best_model_{wandb.run.id}.pt"

    # Training loop
    best_val_acc = 0.0
    best_val_loss = float('inf')
    debug_interval = get_config('debug_interval', 0)  # Set to e.g. 10 to inspect every 10 epochs

    for epoch in range(start_epoch, num_epochs):
        start = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device,
            grad_clip=get_config('grad_clip', 1.0)
        )
        val_loss, val_acc = evaluate(model, val_loader, device)

        if scheduler:
            scheduler.step()

        epoch_time = time.time() - start

        # Debug inspection at intervals
        if debug_interval > 0 and (epoch + 1) % debug_interval == 0:
            inspect_model(model, train_loader, device)

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

            # Save model checkpoint
            # Convert wandb config to dict safely
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
                'val_loss': val_loss,
                'val_acc': val_acc,
                'config': config_dict
            }
            if scheduler is not None:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()

            torch.save(checkpoint, checkpoint_path)
            print(f"  → Saved checkpoint (val_loss: {val_loss:.4f})")

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
        "init_method": "lorentz_kaiming",
        "input_proj_type": "conv_bn_relu",
        "mlr_init": "mlr",

        # Optimization
        "optimizer": "sgd",
        "learning_rate": 1e-1,
        "weight_decay": 1e-3,
        "momentum": 0.9,
        "batch_size": 128,
        "num_epochs": 200,
        "grad_clip": 1.0,

        # Scheduler
        "scheduler": "steplr",
        "warmup_epochs": 10,
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
        "evaluate_test": False,

        # Checkpointing
        "checkpoint_dir": "./checkpoints",
        "resume_checkpoint": None,  # Path to checkpoint to resume from
        "inspect_only": False,      # If True, just inspect the model and exit
        "debug_interval": 0,        # Inspect model every N epochs (0 to disable)
    }

    # wandb.init() will use sweep config if run by wandb agent,
    # otherwise uses default_config
    wandb.init(
        project="ICML_Hyperbolic",
        config=default_config,
    )

    train(wandb.config)
    wandb.finish()


def inspect_checkpoint(checkpoint_path):
    """
    Standalone function to inspect a checkpoint without training.
    Usage: python main.py --inspect path/to/checkpoint.pt
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load checkpoint to get config
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})

    print(f"Checkpoint from epoch {checkpoint.get('epoch', '?')}")
    print(f"Val loss: {checkpoint.get('val_loss', '?')}")
    print(f"Val acc: {checkpoint.get('val_acc', '?')}")
    print(f"Config: {config}")

    # Create model
    manifold = Lorentz(k=config.get('curvature', 1.0))
    model = lorentz_resnet18(
        num_classes=100,
        base_dim=config.get('hidden_dim', 64),
        manifold=manifold,
        init_method=config.get('init_method', 'lorentz_kaiming'),
        input_proj_type=config.get('input_proj_type', 'conv_bn_relu'),
        mlr_init=config.get('mlr_init', 'mlr'),
    ).to(device)

    # Load weights
    load_checkpoint(checkpoint_path, model, device=device)

    # Create minimal dataloader for inspection
    _, val_loader, _ = get_dataloaders(
        batch_size=32,
        data_dir="./data/cifar",
        val_fraction=0.1,
        train_subset_fraction=1.0,
        seed=42,
    )

    # Inspect
    inspect_model(model, val_loader, device)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--inspect', type=str, default=None,
                        help='Path to checkpoint to inspect (skips training)')
    args = parser.parse_args()

    if args.inspect:
        inspect_checkpoint(args.inspect)
    else:
        main()
