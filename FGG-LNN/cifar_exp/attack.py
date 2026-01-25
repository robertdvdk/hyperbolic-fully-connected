# -----------------------------------------------------
# Setup path to import from parent directory (From Source 1)
from pathlib import Path
import sys
import os

parent_dir = Path(__file__).parent
sys.path.insert(0, str(parent_dir.parent))
# -----------------------------------------------------

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

# Imports from your repo
from layers import lorentz_resnet18, Lorentz

# -----------------------------------------------------
# 1. Helper Classes & Functions (Ported from Source 2)
# -----------------------------------------------------

class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# -----------------------------------------------------
# 2. Attack Implementations (Ported from Source 2)
# -----------------------------------------------------

def fgsm_attack(x, x_in, epsilon):
    """Fast Gradient Sign Method"""
    # Note: x.grad must be calculated before calling this
    sign_x_grad = x.grad.sign()
    perturbed_img = x + epsilon * sign_x_grad
    # Clip to ensure valid pixel range (assuming normalized roughly around 0-1 or similar)
    # If your data is normalized, you might want to clip to (min_val, max_val) of the dataset
    return perturbed_img

def pgd_attack(x, x_in, epsilon, alpha=None):
    """Projected Gradient Descent"""
    if alpha is None:
        alpha = epsilon / 4.0
    
    sign_x_grad = x.grad.sign()
    x_adv = x + alpha * sign_x_grad
    
    # Project back to epsilon ball
    eta = torch.clamp(x_adv - x_in, min=-epsilon, max=epsilon)
    perturbed_img = x_in + eta
    
    return perturbed_img

def run_attack_evaluation(model, device, data_loader, attack_name, epsilon, iters=10):
    """Main loop for running adversarial attacks."""
    acc1 = AverageMeter("Acc@1", ":6.2f")
    acc5 = AverageMeter("Acc@5", ":6.2f")
    
    model.eval() # Ensure model is in eval mode (batchnorm, dropout)
    criterion = torch.nn.CrossEntropyLoss()

    print(f"\nRunning {attack_name} (eps={epsilon}, iters={iters})...")

    for i, (x, target) in enumerate(tqdm(data_loader, desc=f"{attack_name}")):
        x, target = x.to(device), target.to(device)
        x_in = x.clone().detach() # Keep original for projection

        # If attack is PGD, we might start from a random point within epsilon ball
        if attack_name == 'pgd':
             x = x + torch.empty_like(x).uniform_(-epsilon, epsilon)
             x = torch.clamp(x, min=x_in.min(), max=x_in.max()) # basic clipping

        # Iterative attack loop
        current_iters = 1 if attack_name == 'fgsm' else iters
        
        # We need a mutable tensor for the loop
        x_adv = x.clone().detach()

        for _ in range(current_iters):
            x_adv.requires_grad = True
            
            # Forward pass
            output = model(x_adv)
            loss = criterion(output, target)
            
            # Backward pass
            model.zero_grad()
            loss.backward()

            # Update inputs
            with torch.no_grad():
                if attack_name == "fgsm":
                    x_adv = fgsm_attack(x_adv, x_in, epsilon)
                elif attack_name == "pgd":
                    x_adv = pgd_attack(x_adv, x_in, epsilon)
                
                # Detach for next iteration
                x_adv = x_adv.detach()

        # Final evaluation on adversarial examples
        with torch.no_grad():
            output_adv = model(x_adv)
            top1, top5 = accuracy(output_adv, target, topk=(1, 5))
            acc1.update(top1.item(), x.shape[0])
            acc5.update(top5.item(), x.shape[0])

    print(f"Result {attack_name} (eps={epsilon}): Top1={acc1.avg:.2f}%, Top5={acc5.avg:.2f}%")
    return acc1.avg

# -----------------------------------------------------
# 3. Model Loading & Config (From Source 1)
# -----------------------------------------------------

class MockWandbConfig:
    """Mock class to load checkpoints with broken wandb.Config objects."""
    def __init__(self): object.__setattr__(self, '_items', {})
    def __setstate__(self, state): 
        for k, v in state.items(): object.__setattr__(self, k, v)
    def __getstate__(self): return self.__dict__
    def __getattr__(self, name):
        if name.startswith('_'): raise AttributeError(name)
        return lambda *args, **kwargs: None
    def get(self, key, default=None):
        return object.__getattribute__(self, '_items').get(key, default)

def load_model_from_checkpoint(checkpoint_path, device, args):
    """Load LorentzResNet model from training checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}")

    # Patch wandb
    import wandb.sdk.wandb_config as wc
    original_config = wc.Config
    wc.Config = MockWandbConfig

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    finally:
        wc.Config = original_config

    # Extract config
    config = checkpoint.get('config', {})
    if hasattr(config, '_items'): config = config._items if config._items else {}
    if not isinstance(config, dict): config = {}

    # Overrides
    hidden_dim = args.hidden_dim if args.hidden_dim is not None else config.get('hidden_dim', 64)
    curvature = args.curvature if args.curvature is not None else config.get('curvature', 1.0)
    init_method = args.init_method if args.init_method is not None else config.get('init_method', 'lorentz_kaiming')
    input_proj_type = config.get('input_proj_type', 'conv_bn_relu')
    mlr_init = config.get('mlr_init', 'mlr')
    normalisation_mode = config.get('normalisation_mode', 'normal')
    mlr_type = config.get('mlr_type', 'fc_mlr')
    use_weight_norm = config.get('use_weight_norm', False)

    # Num classes
    num_classes_map = {'cifar10': 10, 'cifar100': 100, 'imagenet200': 200, 'imagenet1k': 1000}
    num_classes = num_classes_map.get(args.id_name, 100)

    print(
        "Reconstructing model: "
        f"hidden_dim={hidden_dim}, k={curvature}, init={init_method}, classes={num_classes}, "
        f"input_proj_type={input_proj_type}, mlr_init={mlr_init}, normalisation_mode={normalisation_mode}, mlr_type={mlr_type}, use_weight_norm={use_weight_norm}"
    )

    # Create model
    manifold = Lorentz(k_value=curvature)
    model = lorentz_resnet18(
        num_classes=num_classes,
        base_dim=hidden_dim,
        manifold=manifold,
        # activation=nn.ReLU,
        init_method=init_method,
        input_proj_type=input_proj_type,
        mlr_init=mlr_init,
        normalisation_mode=normalisation_mode,
        mlr_type=mlr_type,
        use_weight_norm=use_weight_norm,
    )

    # Load weights
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    
    return model

# -----------------------------------------------------
# 4. Data Loading Logic
# -----------------------------------------------------

def get_dataloader(args):
    """
    Simple dataloader setup. 
    Tries to use torchvision for standard datasets to ensure 
    compatibility with the 'id_name' argument.
    """
    import torchvision
    import torchvision.transforms as transforms
    
    print(f"Preparing data for {args.id_name}...")

    # Standard normalization for CIFAR
    if 'cifar' in args.id_name:
        mean = (0.5074, 0.4867, 0.4411)
        std = (0.267, 0.256, 0.276)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        # Standard Imagenet normalization
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    if args.id_name == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(root=args.data_root, train=False, download=True, transform=transform)
    elif args.id_name == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(root=args.data_root, train=False, download=True, transform=transform)
    elif args.id_name == 'imagenet1k':
        # Assumes standard ImageNet structure
        val_dir = os.path.join(args.data_root, 'val')
        dataset = torchvision.datasets.ImageFolder(val_dir, transform=transform)
    else:
        raise ValueError(f"Dataset {args.id_name} not natively supported in this script yet.")

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    return dataloader

# -----------------------------------------------------
# 5. Main Execution
# -----------------------------------------------------

def get_arguments():
    parser = argparse.ArgumentParser(description='Adversarial Robustness for Lorentz ResNet')

    # Model Args (Source 1)
    parser.add_argument('--checkpoint', required=True, type=str, help="Path to checkpoint.")
    parser.add_argument('--hidden_dim', type=int, default=None)
    parser.add_argument('--curvature', type=float, default=None)
    parser.add_argument('--init_method', type=str, default=None)
    parser.add_argument('--id_name', default='cifar100', type=str, choices=['cifar10', 'cifar100', 'imagenet1k'])
    parser.add_argument('--data_root', default='./data', type=str)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)

    # Adversarial Args (Source 2)
    parser.add_argument('--attack', default='pgd', type=str, choices=['fgsm', 'pgd', 'autoattack', 'all'],
                        help="Type of attack to run.")
    parser.add_argument('--epsilons', type=float, nargs='+', default=[0.0, 0.8/255, 1.6/255, 3.2/255], # [0, 1/255, 2/255, 8/255]
                        help="List of epsilons to test.")
    parser.add_argument('--iters', type=int, default=7, help="Number of iterations for PGD.")

    return parser.parse_args()

def main(args):
    device = args.device
    if 'cuda' in device:
        torch.cuda.set_device(device)
    
    # 1. Load Model
    model = load_model_from_checkpoint(args.checkpoint, device, args)

    # 2. Load Data
    test_loader = get_dataloader(args)

    # 3. Clean Accuracy Check
    print(f"\n{'='*60}")
    print("Evaluating Clean Accuracy")
    print(f"{'='*60}")
    
    acc1_meter = AverageMeter("Acc@1")
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Clean"):
            x, y = x.to(device), y.to(device)
            out = model(x)
            acc1, _ = accuracy(out, y, topk=(1, 5))
            acc1_meter.update(acc1.item(), x.size(0))
    
    print(f"Clean Accuracy: {acc1_meter.avg:.2f}%")

    # 4. Run Attacks
    attacks_to_run = []
    if args.attack == 'all':
        attacks_to_run = ['fgsm', 'pgd', 'autoattack']
    else:
        attacks_to_run = [args.attack]

    for atk in attacks_to_run:
        print(f"\n{'='*60}")
        print(f"Starting {atk.upper()} Attacks")
        print(f"{'='*60}")

        if atk == 'autoattack':
            try:
                from autoattack import AutoAttack
            except ImportError:
                print("Error: 'autoattack' library not found. Please run `pip install autoattack`.")
                continue
            
            # AutoAttack needs normalized data usually, but handles its own loops
            # We construct it once per epsilon usually, or standard eps=8/255
            for eps in args.epsilons:
                if eps == 0: continue
                print(f"Running AutoAttack with eps={eps}")
                adversary = AutoAttack(model, norm='Linf', eps=eps, version='standard', device=device)
                
                # AutoAttack expects full tensors usually, or works batch-wise
                # For simplicity here, we run batch-wise manually if dataset is large, 
                # but standard AA usage is:
                # adversary.run_standard_evaluation(images, labels, bs=batch_size)
                
                # Let's run on a subset or full set depending on time
                # Here we simulate the batch loop for AA
                l = [x for (x, y) in test_loader]
                x_test = torch.cat(l, 0)
                l = [y for (x, y) in test_loader]
                y_test = torch.cat(l, 0)
                
                # Run AA
                adversary.run_standard_evaluation(x_test, y_test, bs=args.batch_size)

        else:
            # FGSM / PGD
            for eps in args.epsilons:
                if eps == 0: continue # Skip 0, we did clean already
                run_attack_evaluation(model, device, test_loader, atk, eps, iters=args.iters)

    print("\nDone.")

if __name__ == '__main__':
    args = get_arguments()
    main(args)