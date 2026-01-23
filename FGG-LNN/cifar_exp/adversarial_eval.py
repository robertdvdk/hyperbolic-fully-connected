# -----------------------------------------------------
# Setup path to import from parent directory
from pathlib import Path
import sys

parent_dir = Path(__file__).parent
sys.path.insert(0, str(parent_dir.parent))
# -----------------------------------------------------

import argparse
from typing import List, Dict, Optional
import json

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import foolbox as fb
from tqdm import tqdm
import numpy as np

from layers import lorentz_resnet18, Lorentz


class MockWandbConfig:
    """Mock class to load checkpoints with broken wandb.Config objects."""

    def __init__(self):
        object.__setattr__(self, '_items', {})

    def __setstate__(self, state):
        for k, v in state.items():
            object.__setattr__(self, k, v)

    def __getstate__(self):
        return self.__dict__

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return lambda *args, **kwargs: None

    def get(self, key, default=None):
        return object.__getattribute__(self, '_items').get(key, default)


def load_model_from_checkpoint(checkpoint_path, device, args):
    """Load LorentzResNet model from training checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}")

    # Patch wandb to handle broken Config objects in old checkpoints
    import wandb.sdk.wandb_config as wc
    original_config = wc.Config
    wc.Config = MockWandbConfig

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    finally:
        wc.Config = original_config

    # Extract config from checkpoint
    config = checkpoint.get('config', {})
    if hasattr(config, '_items'):
        config = config._items if config._items else {}
    if not isinstance(config, dict):
        config = {}

    # Use command-line args to override checkpoint config if provided
    hidden_dim = args.hidden_dim if args.hidden_dim is not None else config.get('hidden_dim', 64)
    curvature = args.curvature if args.curvature is not None else config.get('curvature', 1.0)
    init_method = args.init_method if args.init_method is not None else config.get('init_method', 'lorentz_kaiming')
    do_mlr = config.get('do_mlr', 'angle')

    num_classes = 100  # CIFAR-100

    print(f"  hidden_dim: {hidden_dim}")
    print(f"  curvature: {curvature}")
    print(f"  init_method: {init_method}")
    print(f"  do_mlr: {do_mlr}")

    # Create manifold and model
    manifold = Lorentz(k=curvature)
    model = lorentz_resnet18(
        num_classes=num_classes,
        base_dim=hidden_dim,
        manifold=manifold,
        activation=nn.ReLU,
        init_method=init_method,
        do_mlr=do_mlr,
    )

    # Load weights (handle torch.compile() prefix if present)
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    epoch = checkpoint.get('epoch', 'unknown')
    val_acc = checkpoint.get('val_acc', 'unknown')
    print(f"  Loaded successfully (epoch: {epoch}, val_acc: {val_acc})")

    return model, config


def get_cifar100_testloader(batch_size=200, subset_size=None):
    """Load CIFAR-100 test set with proper normalization."""
    mean = (0.5074, 0.4867, 0.4411)
    std = (0.267, 0.256, 0.276)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    testset = torchvision.datasets.CIFAR100(
        root='./data/cifar',
        train=False,
        download=True,
        transform=transform
    )

    # Use subset for faster evaluation if requested
    if subset_size is not None and subset_size < len(testset):
        indices = np.random.RandomState(42).choice(len(testset), subset_size, replace=False)
        testset = Subset(testset, indices)

    loader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return loader


class HyperbolicModelWrapper(nn.Module):
    """Wrapper for Foolbox compatibility with hyperbolic models.

    Handles the exponential map from Euclidean space to hyperboloid.
    Adversarial perturbations are applied in Euclidean space (before expmap).
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        # Extract manifold from the model's first layer
        self.manifold = self._get_manifold(model)

    def _get_manifold(self, model):
        """Extract the Lorentz manifold from the model."""
        # The manifold should be stored in the model's layers
        for module in model.modules():
            if hasattr(module, 'manifold'):
                return module.manifold
        raise ValueError("Could not find manifold in model")

    def forward(self, x):
        """Forward pass that maps Euclidean input to hyperboloid first."""
        # x is in Euclidean space (B, C, H, W)
        # Map each spatial position to hyperboloid
        # batch_size, channels, height, width = x.shape

        # Rearrange to (B, H, W, C) for expmap
        # x_euclidean = x.permute(0, 2, 3, 1)

        # Map to hyperboloid: (B, H, W, C) -> (B, H, W, C+1)
        # x_hyperbolic = self.manifold.expmap0(x_euclidean)

        # Rearrange back to (B, C+1, H, W)
        # x_hyperbolic = x_hyperbolic.permute(0, 3, 1, 2)

        # Forward through model
        # logits = self.model(x)
        return self.model(x)


def evaluate_clean_accuracy(model, loader, device):
    """Evaluate clean accuracy (no adversarial attack)."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Clean accuracy", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def evaluate_adversarial_robustness(
    model,
    loader,
    device,
    attack,
    epsilon,
    bounds=(0, 1),
    max_batches=None
):
    """
    Evaluate adversarial robustness using Foolbox.

    Args:
        model: PyTorch model (will be wrapped for Foolbox)
        loader: DataLoader for test data
        device: Device to run on
        attack: Foolbox attack instance
        epsilon: Perturbation budget (L-inf norm)
        bounds: Input bounds for the attack
        max_batches: Limit number of batches for faster testing

    Returns:
        robust_accuracy: Accuracy on adversarial examples
    """
    model.eval()

    # Wrap model for Foolbox
    fmodel = fb.PyTorchModel(model, bounds=bounds, device=device)

    correct = 0
    total = 0
    batch_count = 0

    for images, labels in tqdm(loader, desc=f"  ε={epsilon:.3f}", leave=False):
        if max_batches is not None and batch_count >= max_batches:
            break

        images, labels = images.to(device), labels.to(device)

        # Generate adversarial examples
        # For epsilon=0, we just evaluate clean accuracy
        if epsilon == 0:
            with torch.no_grad():
                predictions = fmodel(images).argmax(dim=-1)
        else:
            # Run attack
            _, advs, success = attack(fmodel, images, labels, epsilons=epsilon)

            # Evaluate on adversarial examples
            with torch.no_grad():
                predictions = fmodel(advs).argmax(dim=-1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        batch_count += 1

    robust_accuracy = 100 * correct / total
    return robust_accuracy


def get_attack(attack_name: str):
    """Get Foolbox attack by name."""
    attacks = {
        'fgsm': fb.attacks.FGSM(),
        'pgd': fb.attacks.LinfPGD(steps=20, random_start=True),
        'pgd_50': fb.attacks.LinfPGD(steps=50, random_start=True),
        'deepfool': fb.attacks.LinfDeepFoolAttack(steps=50),
        'cw': fb.attacks.L2CarliniWagnerAttack(steps=1000, binary_search_steps=5),
    }

    if attack_name.lower() not in attacks:
        raise ValueError(f"Unknown attack: {attack_name}. Available: {list(attacks.keys())}")

    return attacks[attack_name.lower()]


def compare_models(
    checkpoint_paths: List[str],
    model_names: List[str],
    attacks: List[str],
    epsilons: List[float],
    device: str,
    args
):
    """
    Compare adversarial robustness of multiple models.

    Returns:
        results: Dict mapping model_name -> attack_name -> epsilon -> accuracy
    """
    # Load test data
    print("\nLoading CIFAR-100 test set...")
    test_loader = get_cifar100_testloader(
        batch_size=args.batch_size,
        subset_size=args.subset_size
    )
    print(f"Loaded {len(test_loader.dataset)} test images\n")

    # Load models
    print("Loading models...")
    models = {}
    for checkpoint_path, model_name in zip(checkpoint_paths, model_names):
        model, config = load_model_from_checkpoint(checkpoint_path, device, args)
        wrapped_model = HyperbolicModelWrapper(model)
        models[model_name] = wrapped_model
        print()

    # Initialize results
    results = {name: {} for name in model_names}

    # Evaluate each model
    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")

        # Clean accuracy
        print("\nClean accuracy (no attack)...")
        clean_acc = evaluate_clean_accuracy(model, test_loader, device)
        print(f"  Clean accuracy: {clean_acc:.2f}%")

        if 'clean' not in results[model_name]:
            results[model_name]['clean'] = {}
        results[model_name]['clean'][0.0] = clean_acc

        # Adversarial robustness for each attack
        for attack_name in attacks:
            print(f"\nAttack: {attack_name.upper()}")
            attack = get_attack(attack_name)
            results[model_name][attack_name] = {}

            for epsilon in epsilons:
                robust_acc = evaluate_adversarial_robustness(
                    model,
                    test_loader,
                    device,
                    attack,
                    epsilon,
                    max_batches=args.max_batches
                )
                results[model_name][attack_name][epsilon] = robust_acc
                print(f"  ε={epsilon:.3f}: {robust_acc:.2f}%")

    return results


def print_comparison_table(results: Dict, attacks: List[str], epsilons: List[float]):
    """Print a formatted comparison table."""
    print(f"\n{'='*80}")
    print("ADVERSARIAL ROBUSTNESS COMPARISON")
    print(f"{'='*80}\n")

    model_names = list(results.keys())

    # Clean accuracy table
    print("Clean Accuracy (no attack):")
    print(f"{'Model':<30} {'Accuracy':>10}")
    print("-" * 42)
    for name in model_names:
        acc = results[name]['clean'][0.0]
        print(f"{name:<30} {acc:>9.2f}%")
    print()

    # Per-attack tables
    for attack_name in attacks:
        print(f"\n{attack_name.upper()} Attack:")

        # Header
        header = f"{'Model':<30}"
        for eps in epsilons:
            header += f" ε={eps:<6.3f}"
        print(header)
        print("-" * (30 + 10 * len(epsilons)))

        # Results for each model
        for name in model_names:
            row = f"{name:<30}"
            for eps in epsilons:
                acc = results[name][attack_name].get(eps, 0.0)
                row += f" {acc:>7.2f}%"
            print(row)
        print()


def save_results(results: Dict, output_path: str):
    """Save results to JSON file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")


def get_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Adversarial Robustness Evaluation for Hyperbolic Networks'
    )

    # Model checkpoints
    parser.add_argument('--checkpoints', nargs='+', required=True,
                        help="Paths to model checkpoints to evaluate")
    parser.add_argument('--names', nargs='+', default=None,
                        help="Names for models (default: checkpoint filenames)")

    # Model config overrides
    parser.add_argument('--hidden_dim', type=int, default=None)
    parser.add_argument('--curvature', type=float, default=None)
    parser.add_argument('--init_method', type=str, default=None)

    # Attack settings
    parser.add_argument('--attacks', nargs='+',
                        default=['fgsm', 'pgd'],
                        choices=['fgsm', 'pgd', 'pgd_50', 'deepfool', 'cw'],
                        help="Attacks to evaluate")
    parser.add_argument('--epsilons', nargs='+', type=float,
                        default=[0.01, 0.03, 0.05, 0.1],
                        help="Epsilon values (L-inf perturbation budgets)")

    # Evaluation settings
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--subset_size', type=int, default=None,
                        help="Use subset of test set for faster evaluation")
    parser.add_argument('--max_batches', type=int, default=None,
                        help="Max batches per attack (for quick testing)")
    parser.add_argument('--device', type=str, default='cuda:0')

    # Output
    parser.add_argument('--output', type=str, default='adversarial_results.json',
                        help="Path to save results JSON")

    return parser.parse_args()


def main():
    args = get_arguments()

    # Set device
    device = args.device
    if 'cuda' in device:
        torch.cuda.set_device(device)
        print(f"Using device: {device}")

    # Default model names to checkpoint filenames if not provided
    if args.names is None:
        args.names = [Path(ckpt).stem for ckpt in args.checkpoints]

    if len(args.names) != len(args.checkpoints):
        raise ValueError("Number of names must match number of checkpoints")

    print(f"\n{'='*80}")
    print("ADVERSARIAL ROBUSTNESS EVALUATION")
    print(f"{'='*80}")
    print(f"\nModels to compare: {len(args.checkpoints)}")
    for name, ckpt in zip(args.names, args.checkpoints):
        print(f"  - {name}: {ckpt}")
    print(f"\nAttacks: {', '.join(args.attacks)}")
    print(f"Epsilons: {args.epsilons}")
    print()

    # Run comparison
    results = compare_models(
        checkpoint_paths=args.checkpoints,
        model_names=args.names,
        attacks=args.attacks,
        epsilons=args.epsilons,
        device=device,
        args=args
    )

    # Print results
    print_comparison_table(results, args.attacks, args.epsilons)

    # Save results
    save_results(results, args.output)

    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
