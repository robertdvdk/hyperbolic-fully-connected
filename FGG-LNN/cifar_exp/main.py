# cifar100_hyperbolic.py
from pathlib import Path
import sys
import time
import math
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


def get_param_groups(model, lr_manifold, weight_decay_manifold, verbose=False):
    no_decay = ["scale"]
    k_params = ["manifold.k"]

    # Group 0: standard params with weight decay
    group0_params = []
    group0_names = []
    # Group 1: ManifoldParameters
    group1_params = []
    group1_names = []
    # Group 2: k parameters (no weight decay)
    group2_params = []
    group2_names = []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(nd in n for nd in k_params):
            group2_params.append(p)
            group2_names.append(n)
        elif isinstance(p, ManifoldParameter):
            group1_params.append(p)
            group1_names.append(n)
        elif not any(nd in n for nd in no_decay):
            group0_params.append(p)
            group0_names.append(n)
        else:
            # This would be params matching no_decay but not other conditions
            if verbose:
                print(f"  WARNING: param {n} excluded from all groups (no weight decay)")

    if verbose:
        print(f"\n--- Parameter Groups ---")
        print(f"Group 0 (standard, with WD): {len(group0_params)} params")
        gamma_params = [n for n in group0_names if 'gamma' in n]
        print(f"  gamma params in group 0: {gamma_params}")
        print(f"Group 1 (ManifoldParam, reduced LR): {len(group1_params)} params")
        print(f"Group 2 (k params, no WD): {len(group2_params)} params")

    parameters = [
        {"params": group0_params},
        {"params": group1_params, 'lr': lr_manifold, "weight_decay": weight_decay_manifold},
        {"params": group2_params, "weight_decay": 0, "lr": 1e-4}
    ]

    return parameters

# def select_optimizer(model, lr, weight_decay, warmup_epochs):
#     """ Selects and sets up an available optimizer and returns it. """

#     model_parameters = get_param_groups(model, lr*0.2, weight_decay, verbose=True)
#     optimizer = RiemannianSGD(model_parameters, lr=lr, weight_decay=weight_decay, momentum=0.9, nesterov=True, stabilize=1)

    # # Verify weight decay is being applied to gamma
    # print(f"\n--- Optimizer weight decay verification ---")
    # print(f"Optimizer-level weight_decay: {weight_decay}")
    # for i, group in enumerate(optimizer.param_groups):
    #     wd = group.get('weight_decay', 'NOT SET')
    #     print(f"Group {i}: weight_decay={wd}, num_params={len(group['params'])}")
    # from torch.optim.lr_scheduler import SequentialLR, MultiStepLR, LinearLR
    # warmup_scheduler = LinearLR(optimizer,
    #                             start_factor=0.01,
    #                             end_factor=1.0,
    #                             total_iters=warmup_epochs)
    # step_scheduler = MultiStepLR(
    #     optimizer, milestones=[m - warmup_epochs for m in [10, 20, 30]], gamma=0.2
    # )
    # lr_scheduler = SequentialLR(optimizer,
    #                             schedulers=[warmup_scheduler, step_scheduler],
    #                             milestones=[warmup_epochs])
    
    # return optimizer, lr_scheduler

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
    # print("\n--- U_norm values in LorentzFullyConnected layers ---")
    # for name, module in model.named_modules():
    #     if hasattr(module, 'U'):
    #         U_norm = module.U.norm(dim=0)
    #         print(f"{name}:")
    #         print(f"  U_norm: min={U_norm.min().item():.6f}, max={U_norm.max().item():.6f}, mean={U_norm.mean().item():.6f}")
    #         if hasattr(module, 'a'):
    #             a_vals = module.a
    #             print(f"  a: min={a_vals.min().item():.6f}, max={a_vals.max().item():.6f}, mean={a_vals.mean().item():.6f}")
    #             # Check effective bias magnitude (a / U_norm)
    #             effective = a_vals / U_norm
    #             print(f"  a/U_norm: min={effective.min().item():.6f}, max={effective.max().item():.6f}")

    # Check z and a in LorentzMLR
    # print("\n--- LorentzMLR parameters ---")
    # for name, module in model.named_modules():
    #     if hasattr(module, 'z') and hasattr(module, 'a') and not hasattr(module, 'U'):
    #         print(f"{name}:")
    #         print(f"  z norm: min={module.z.norm(dim=-1).min().item():.6f}, max={module.z.norm(dim=-1).max().item():.6f}")
    #         print(f"  a: min={module.a.min().item():.6f}, max={module.a.max().item():.6f}")

    # Check BatchNorm parameters - especially stage4.1.bn2
    print("\n--- BatchNorm parameters (gamma/beta for all BN layers) ---")
    for name, module in model.named_modules():
        if 'bn' in name.lower() or 'batchnorm' in name.lower() or 'BatchNorm' in type(module).__name__:
            print(f"{name} ({type(module).__name__}):")
            # LorentzBatchNorm uses gamma/beta, standard BN uses weight/bias
            if hasattr(module, 'gamma') and module.gamma is not None:
                print(f"  gamma: {module.gamma.item():.6f}")
            if hasattr(module, 'weight') and module.weight is not None:
                print(f"  gamma (weight): min={module.weight.min().item():.6f}, max={module.weight.max().item():.6f}")
            if hasattr(module, 'beta') and module.beta is not None:
                beta = module.beta
                print(f"  beta: time={beta[0].item():.4f}, space_norm={beta[1:].norm().item():.4f}")
            if hasattr(module, 'bias') and module.bias is not None:
                print(f"  beta (bias): min={module.bias.min().item():.6f}, max={module.bias.max().item():.6f}")
            if hasattr(module, 'running_mean') and module.running_mean is not None:
                rm = module.running_mean
                print(f"  running_mean: norm={rm.norm().item():.6f}")
            if hasattr(module, 'running_var') and module.running_var is not None:
                rv = module.running_var
                print(f"  running_var: {rv.item():.6f}")

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

        # Detailed stage4 inspection - find where explosion happens
        print("\n  --- Detailed Stage4 inspection ---")
        x_stage = x3
        for block_idx, block in enumerate(model.stage4):
            print(f"  Stage4.{block_idx} input: time=[{x_stage[:,0].min().item():.2f}, {x_stage[:,0].max().item():.2f}], "
                  f"space_norm={x_stage[:,1:].norm(dim=1).mean().item():.2f}")

            # Through layer1
            x_l1 = block.layer1(x_stage)
            print(f"    after layer1: time=[{x_l1[:,0].min().item():.2f}, {x_l1[:,0].max().item():.2f}], "
                  f"space_norm={x_l1[:,1:].norm(dim=1).mean().item():.2f}")

            # Through bn1
            x_bn1 = block.bn1(x_l1)
            print(f"    after bn1: time=[{x_bn1[:,0].min().item():.2f}, {x_bn1[:,0].max().item():.2f}], "
                  f"space_norm={x_bn1[:,1:].norm(dim=1).mean().item():.2f}")

            # Through relu
            x_relu1 = block.manifold.relu(x_bn1, manifold_dim=1)
            print(f"    after relu1: time=[{x_relu1[:,0].min().item():.2f}, {x_relu1[:,0].max().item():.2f}], "
                  f"space_norm={x_relu1[:,1:].norm(dim=1).mean().item():.2f}")

            # Through layer2
            x_l2 = block.layer2(x_relu1)
            print(f"    after layer2: time=[{x_l2[:,0].min().item():.2f}, {x_l2[:,0].max().item():.2f}], "
                  f"space_norm={x_l2[:,1:].norm(dim=1).mean().item():.2f}")

            # Through bn2 - with detailed internal tracing (if bn2 exists)
            if block.bn2 is not None:
                x_bn2 = block.bn2(x_l2)
                print(f"    after bn2: time=[{x_bn2[:,0].min().item():.2f}, {x_bn2[:,0].max().item():.2f}], "
                      f"space_norm={x_bn2[:,1:].norm(dim=1).mean().item():.2f}")

                # Detailed bn2 tracing if there's explosion
                if x_bn2[:,0].max().item() > 1e6 or torch.isnan(x_bn2).any():
                    print(f"\n    !!! EXPLOSION in bn2 - detailed trace !!!")
                    bn = block.bn2
                    m = bn.manifold

                    # Reshape like bn2 does
                    bs, c, h, w = x_l2.shape
                    x_flat = x_l2.permute(0, 2, 3, 1).reshape(bs, -1, c)

                    # Get running mean on manifold
                    running_mean = m.expmap0(bn.running_mean)
                    print(f"      running_mean: time={running_mean[0].item():.4f}, space_norm={running_mean[1:].norm().item():.4f}")

                    # Logmap: map input to tangent space at running_mean
                    x_T = m.logmap(running_mean, x_flat)
                    tangent_norms = x_T.norm(dim=-1)  # [bs, H*W]
                    print(f"      logmap tangent norms: min={tangent_norms.min().item():.4f}, max={tangent_norms.max().item():.4f}, "
                          f"mean={tangent_norms.mean().item():.4f}, num>10={((tangent_norms > 10).sum().item())}")

                    # Transp0back: transport to origin
                    x_T = m.transp0back(running_mean, x_T)
                    tangent_norms_after = x_T.norm(dim=-1)
                    print(f"      after transp0back: norms min={tangent_norms_after.min().item():.4f}, max={tangent_norms_after.max().item():.4f}")

                    # Scaling
                    scale_factor = bn.gamma / (bn.running_var + bn.eps)
                    print(f"      scale factor: {scale_factor.item():.4f} (gamma={bn.gamma.item():.4f}, var={bn.running_var.item():.4f})")
                    x_T_scaled = x_T * scale_factor
                    scaled_norms = x_T_scaled.norm(dim=-1)
                    print(f"      after scaling: norms min={scaled_norms.min().item():.4f}, max={scaled_norms.max().item():.4f}")

                    # Transport to beta
                    beta = bn.beta
                    print(f"      beta: time={beta[0].item():.4f}, space_norm={beta[1:].norm().item():.4f}")
                    x_T_at_beta = m.transp0(beta, x_T_scaled)
                    beta_norms = x_T_at_beta.norm(dim=-1)
                    print(f"      after transp0 to beta: norms min={beta_norms.min().item():.4f}, max={beta_norms.max().item():.4f}")

                    # Expmap: this is where explosion happens
                    output = m.expmap(beta, x_T_at_beta)
                    print(f"      after expmap: time=[{output[...,0].min().item():.2f}, {output[...,0].max().item():.2f}]")

                    # Find the specific positions that exploded
                    time_vals = output[..., 0]
                    max_idx = time_vals.argmax()
                    batch_idx = max_idx // (h * w)
                    spatial_idx = max_idx % (h * w)
                    print(f"      Worst explosion at batch={batch_idx}, spatial={spatial_idx}")
                    print(f"        input time: {x_flat[batch_idx, spatial_idx, 0].item():.4f}")
                    print(f"        tangent norm after logmap: {tangent_norms[batch_idx, spatial_idx].item():.4f}")
                    print(f"        scaled tangent norm: {scaled_norms[batch_idx, spatial_idx].item():.4f}")
                    print()
            else:
                x_bn2 = x_l2  # No bn2, just pass through
                print(f"    bn2: SKIPPED (skip_bn2=True)")

            # Through proj (shortcut)
            x_proj = block.proj(x_stage)
            print(f"    after proj: time=[{x_proj[:,0].min().item():.2f}, {x_proj[:,0].max().item():.2f}], "
                  f"space_norm={x_proj[:,1:].norm(dim=1).mean().item():.2f}")

            # Residual addition (space only)
            out_space = x_proj[:, 1:, :, :] + x_bn2[:, 1:, :, :]
            print(f"    after space add: space_norm={out_space.norm(dim=1).mean().item():.2f}, "
                  f"space_max={out_space.abs().max().item():.2f}")

            # Projection to get time
            x_out = block.manifold.projection_space_orthogonal(out_space, manifold_dim=1)
            print(f"    after proj_orth: time=[{x_out[:,0].min().item():.2f}, {x_out[:,0].max().item():.2f}], "
                  f"space_norm={x_out[:,1:].norm(dim=1).mean().item():.2f}")

            # Final relu
            x_stage = block.manifold.relu(x_out, manifold_dim=1)
            print(f"    after relu2: time=[{x_stage[:,0].min().item():.2f}, {x_stage[:,0].max().item():.2f}], "
                  f"space_norm={x_stage[:,1:].norm(dim=1).mean().item():.2f}")

        x4 = x_stage
        print(f"After stage4: time=[{x4[:,0].min().item():.2f}, {x4[:,0].max().item():.2f}], "
              f"space_norm={x4[:,1:].norm(dim=1).mean().item():.2f}")

        x_tokens = x4.permute(0, 2, 3, 1).reshape(x4.shape[0], -1, x4.shape[1])
        print(f"After flatten tokens: shape={tuple(x_tokens.shape)}, "
            f"time=[{x_tokens[...,0].min().item():.2f}, {x_tokens[...,0].max().item():.2f}], "
            f"space_norm={x_tokens[...,1:].norm(dim=-1).mean().item():.2f}")

        token_logits = model.classifier(x_tokens)
        print(f"After classifier (per-token): shape={tuple(token_logits.shape)}, "
            f"min={token_logits.min().item():.2f}, max={token_logits.max().item():.2f}, "
            f"std={token_logits.std().item():.2f}, mean_abs={token_logits.abs().mean().item():.2f}")

        logits = token_logits.mean(dim=1)
        print(f"Logits (mean over tokens): min={logits.min().item():.2f}, max={logits.max().item():.2f}, "
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


def train_epoch(model, train_loader, optimizer, device='cuda', grad_clip=1.0, debug_gamma=False):
    """Train for one epoch, return avg loss and accuracy."""
    model.train()
    running_loss, total_correct, total_samples = 0.0, 0, 0

    # For tracking gamma gradients
    gamma_grad_accum = {}
    gamma_values = {}
    batch_count = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x).squeeze()
        loss = F.cross_entropy(logits, y)
        loss.backward()

        # Track gamma gradients after backward
        if debug_gamma and batch_count < 10:  # First 10 batches
            for name, param in model.named_parameters():
                if 'gamma' in name and param.grad is not None:
                    if name not in gamma_grad_accum:
                        gamma_grad_accum[name] = []
                    gamma_grad_accum[name].append(param.grad.item())
                    gamma_values[name] = param.item()
        

        # torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += x.size(0)
        batch_count += 1

    # Print gamma gradient summary
    if debug_gamma and gamma_grad_accum:
        print("\n  --- Gamma gradients (first 10 batches) ---")
        for name in sorted(gamma_grad_accum.keys()):
            grads = gamma_grad_accum[name]
            avg_grad = sum(grads) / len(grads)
            print(f"  {name}: value={gamma_values[name]:.4f}, avg_grad={avg_grad:.6f}, "
                  f"all_positive={all(g > 0 for g in grads)}, all_negative={all(g < 0 for g in grads)}")

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
    manifold = Lorentz(k_value=get_config('curvature', 1.0))
    normalisation_mode = get_config('normalisation_mode', get_config('bn_mode', 'normal'))
    mlr_type = get_config('mlr_type', get_config('classifier_type', 'lorentz_mlr'))
    model = lorentz_resnet18(
        num_classes=100,
        base_dim=get_config('hidden_dim', 64),
        manifold=manifold,
        init_method=get_config('init_method', 'lorentz_kaiming'),
        input_proj_type=get_config('input_proj_type', 'conv_bn_relu'),
        mlr_init=get_config('mlr_init', 'mlr'),
        normalisation_mode=normalisation_mode,  # "normal", "fix_gamma", "skip_final_bn2", "clamp_scale", "mean_only", or "centering_only"
        mlr_type=mlr_type,  # "lorentz_mlr" or "fc_mlr"
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
    # optimizer, scheduler = select_optimizer(model, lr=lr, weight_decay=weight_decay, warmup_epochs=warmup_epochs)

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

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_name == "sgd":
        momentum = get_config('momentum', 0.9)
        optimizer = RiemannianSGD(params=model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True, stabilize=1)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Learning rate scheduler
    scheduler_type = get_config('scheduler', 'none').lower()
    num_epochs = get_config('num_epochs', 100)
    warmup_epochs = get_config('warmup_epochs', 0)

    if scheduler_type == 'steplr':
        from torch.optim.lr_scheduler import SequentialLR, MultiStepLR, LinearLR

        # Milestones at ~40%, 70%, 90% of training
        milestones = get_config('milestones', [int(num_epochs * 0.3), int(num_epochs * 0.6), int(num_epochs * 0.8)])
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

        # Debug gamma gradients at debug intervals
        should_debug_gamma = debug_interval > 0 and (epoch + 1) % debug_interval == 0

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device,
            grad_clip=get_config('grad_clip', 1.0),
            debug_gamma=should_debug_gamma
        )
        val_loss, val_acc = evaluate(model, val_loader, device)

        if not all(map(math.isfinite, [train_loss, train_acc, val_loss, val_acc])):
            msg = (
                f"NaN/Inf detected at epoch {epoch + 1}: "
                f"train_loss={train_loss}, train_acc={train_acc}, "
                f"val_loss={val_loss}, val_acc={val_acc}"
            )
            print(msg)
            wandb.run.summary["nan_detected"] = True
            wandb.run.summary["nan_epoch"] = epoch + 1
            wandb.log({"nan_detected": 1, "nan_epoch": epoch + 1})
            wandb.finish(exit_code=1)
            raise RuntimeError(msg)

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
        "normalisation_mode": "clamp_scale",  # "normal", "fix_gamma", "skip_final_bn2", "clamp_scale", "mean_only", or "centering_only"
        "mlr_type": "fc_mlr",  # "lorentz_mlr" or "fc_mlr"

        # Optimization
        "optimizer": "sgd",
        "learning_rate": 1e-1,
        "weight_decay": 5e-4,
        "momentum": 0.9,
        "batch_size": 128,
        "num_epochs": 200,
        "grad_clip": 1.0,

        # Scheduler
        "scheduler": "cosine",
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
        "debug_interval": 5,       # Inspect model every N epochs (0 to disable)
    }

    # wandb.init() will use sweep config if run by wandb agent,
    # otherwise uses default_config
    wandb.init(
        project="ICML_Hyperbolic",
        config=default_config,
        name="ignore"
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
    manifold = Lorentz(k_value=config.get('curvature', 1.0))
    model = lorentz_resnet18(
        num_classes=100,
        base_dim=config.get('hidden_dim', 64),
        manifold=manifold,
        init_method=config.get('init_method', 'lorentz_kaiming'),
        input_proj_type=config.get('input_proj_type', 'conv_bn_relu'),
        mlr_init=config.get('mlr_init', 'mlr'),
        normalisation_mode=config.get('normalisation_mode', config.get('bn_mode', 'normal')),
        mlr_type=config.get('mlr_type', config.get('classifier_type', 'lorentz_mlr')),
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
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--inspect', type=str, default=None,
    #                     help='Path to checkpoint to inspect (skips training)')
    # args = parser.parse_args()

    # if args.inspect:
    #     inspect_checkpoint(args.inspect)
    # else:
    main()
