# -----------------------------------------------------
# Setup path to import from parent directory
from pathlib import Path
import sys

parent_dir = Path(__file__).parent
sys.path.insert(0, str(parent_dir.parent))
# -----------------------------------------------------

import argparse

import torch
import torch.nn as nn
from layers import lorentz_resnet18, Lorentz


class OpenOODWrapper(nn.Module):
    """Wrapper to make LorentzResNet compatible with OpenOOD evaluator."""

    def __init__(self, resnet_model):
        super().__init__()
        self.model = resnet_model

    def forward(self, x, return_feature=False, return_feature_list=False):
        """Forward pass returning logits (features not supported for hyperbolic models)."""
        logits = self.model(x)
        return logits


def get_arguments():
    """Parse command-line options for OOD evaluation."""
    parser = argparse.ArgumentParser(description='OpenOOD Evaluation for Lorentz ResNet')

    # Required: checkpoint to evaluate
    parser.add_argument('--checkpoint', required=True, type=str,
                        help="Path to trained model checkpoint.")

    # Model configuration (can override checkpoint config)
    parser.add_argument('--hidden_dim', type=int, default=None,
                        help="Base dimension for ResNet (overrides checkpoint config if set).")
    parser.add_argument('--curvature', type=float, default=None,
                        help="Manifold curvature (overrides checkpoint config if set).")
    parser.add_argument('--init_method', type=str, default=None,
                        help="Initialization method (overrides checkpoint config if set).")

    # OpenOOD settings
    parser.add_argument('--id_name', default='cifar100', type=str,
                        choices=['cifar10', 'cifar100', 'imagenet200', 'imagenet1k'],
                        help="ID dataset name for OpenOOD.")
    parser.add_argument('--data_root', default='./data', type=str,
                        help="Root directory for OpenOOD datasets.")
    parser.add_argument('--postprocessor', default='msp', type=str,
                        help="OOD detection method (e.g., msp, ebo, odin, vim, knn, react, ash).")
    parser.add_argument('--batch_size', default=200, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument('--device', default='cuda:0', type=str,
                        help="Device to use for evaluation.")

    # Multiple postprocessors
    parser.add_argument('--test_all', action='store_true',
                        help="Test multiple postprocessors.")

    return parser.parse_args()


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

    # Extract config from checkpoint (handle broken wandb.Config objects)
    config = checkpoint.get('config', {})
    if hasattr(config, '_items'):
        config = config._items if config._items else {}
    if not isinstance(config, dict):
        config = {}

    # Use command-line args to override checkpoint config if provided
    # Default to values that match main.py defaults
    hidden_dim = args.hidden_dim if args.hidden_dim is not None else config.get('hidden_dim', 64)
    curvature = args.curvature if args.curvature is not None else config.get('curvature', 1.0)
    init_method = args.init_method if args.init_method is not None else config.get('init_method', 'lorentz_kaiming')
    input_proj_type = config.get('input_proj_type', 'conv_bn_relu')
    mlr_init = config.get('mlr_init', 'mlr')

    # Determine num_classes from id_name
    num_classes_map = {
        'cifar10': 10,
        'cifar100': 100,
        'imagenet200': 200,
        'imagenet1k': 1000,
    }
    num_classes = num_classes_map.get(args.id_name, 100)

    print("Reconstructing model with configuration:")
    print(f"  hidden_dim: {hidden_dim}")
    print(f"  curvature: {curvature}")
    print(f"  init_method: {init_method}")
    print(f"  input_proj_type: {input_proj_type}")
    print(f"  mlr_init: {mlr_init}")
    print(f"  num_classes: {num_classes}")

    # Create manifold and model
    manifold = Lorentz(k=curvature)
    model = lorentz_resnet18(
        num_classes=num_classes,
        base_dim=hidden_dim,
        manifold=manifold,
        init_method=init_method,
        input_proj_type=input_proj_type,
        mlr_init=mlr_init,
    )

    # Load weights (handle torch.compile() prefix if present)
    state_dict = checkpoint['model_state_dict']

    # Strip _orig_mod. prefix if present (from torch.compile())
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    epoch = checkpoint.get('epoch', 'unknown')
    val_acc = checkpoint.get('val_acc', 'unknown')
    print(f"Model loaded successfully (epoch: {epoch}, val_acc: {val_acc})")

    return model, config


def main(args):
    from openood.evaluation_api import Evaluator

    device = args.device
    if 'cuda' in device:
        torch.cuda.set_device(device)

    # Load trained model
    model, config = load_model_from_checkpoint(args.checkpoint, device, args)

    # Wrap for OpenOOD
    wrapped_model = OpenOODWrapper(model)
    wrapped_model.eval()

    print(f"\n{'='*60}")
    print("Starting OpenOOD Evaluation")
    print(f"{'='*60}")

    if args.test_all:
        # Test multiple postprocessors
        postprocessors = ['msp', 'energy', 'odin']
        print(f"Testing postprocessors: {postprocessors}\n")

        results = {}
        for method in postprocessors:
            print(f"\n{'-'*60}")
            print(f"Testing: {method.upper()}")
            print(f"{'-'*60}")

            try:
                evaluator = Evaluator(
                    wrapped_model,
                    id_name=args.id_name,
                    data_root=args.data_root,
                    postprocessor_name=method,
                    batch_size=args.batch_size,
                )

                # Evaluate OOD detection
                ood_metrics = evaluator.eval_ood(fsood=False, progress=True)
                results[method] = ood_metrics

                print(f"\nResults for {method.upper()}:")
                print(ood_metrics)

            except Exception as e:
                print(f"Error with {method}: {e}")
                import traceback
                traceback.print_exc()
                results[method] = None

        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY - All Methods")
        print(f"{'='*60}")
        for method, metrics in results.items():
            if metrics is not None:
                print(f"\n{method.upper()}:")
                print(metrics)

    else:
        # Test single postprocessor
        print(f"Testing postprocessor: {args.postprocessor}\n")

        evaluator = Evaluator(
            wrapped_model,
            id_name=args.id_name,
            data_root=args.data_root,
            postprocessor_name=args.postprocessor,
            batch_size=args.batch_size,
        )

        # Evaluate ID accuracy
        print("Evaluating ID accuracy...")
        id_acc = evaluator.eval_acc(data_name='id')
        print(f"ID Accuracy: {id_acc:.2f}%")

        # Evaluate OOD detection
        print("\nEvaluating OOD detection...")
        ood_metrics = evaluator.eval_ood(fsood=False, progress=True)

        print(f"\n{'='*60}")
        print(f"Results for {args.postprocessor.upper()}")
        print(f"{'='*60}")
        print(ood_metrics)

    print(f"\n{'='*60}")
    print("Evaluation Complete")
    print(f"{'='*60}")


if __name__ == '__main__':
    args = get_arguments()

    # Check OpenOOD is installed
    try:
        from openood.evaluation_api import Evaluator
    except ImportError:
        print("ERROR: OpenOOD not installed!")
        print("Please run: pip install git+https://github.com/Jingkang50/OpenOOD")
        print("            pip install libmr")
        exit(1)

    main(args)