#!/usr/bin/env python3
"""
Sweep runner for hyperparameter optimization.

Usage:
    # Create a new sweep and get the sweep ID
    python run_sweep.py --create

    # Run agents for an existing sweep
    python run_sweep.py --sweep_id <SWEEP_ID> --count 10

    # Create and immediately run agents
    python run_sweep.py --create --count 10
"""

import argparse
import yaml
from pathlib import Path

import wandb

from main import train


def load_sweep_config(config_path: str = "sweep_config.yaml") -> dict:
    """Load sweep configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_sweep(project: str, config_path: str = "sweep_config.yaml") -> str:
    """Create a new W&B sweep and return the sweep ID."""
    config = load_sweep_config(config_path)
    sweep_id = wandb.sweep(config, project=project)
    print(f"Created sweep with ID: {sweep_id}")
    print(f"View at: https://wandb.ai/{wandb.api.default_entity}/{project}/sweeps/{sweep_id}")
    return sweep_id


def run_agent(sweep_id: str, project: str, count: int = None):
    """Run a W&B sweep agent."""

    def train_wrapper():
        """Wrapper that initializes wandb and calls train."""
        wandb.init()
        try:
            train(wandb.config)
        except Exception as e:
            print(f"Run failed with error: {e}")
            raise
        finally:
            wandb.finish()

    wandb.agent(
        sweep_id,
        function=train_wrapper,
        project=project,
        count=count
    )


def main():
    parser = argparse.ArgumentParser(description="Run W&B hyperparameter sweep")
    parser.add_argument(
        "--create",
        action="store_true",
        help="Create a new sweep"
    )
    parser.add_argument(
        "--sweep_id",
        type=str,
        default=None,
        help="Existing sweep ID to join"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of runs for this agent (None = run indefinitely)"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="ICML_Hyperbolic_Sweep",
        help="W&B project name"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="sweep_config.yaml",
        help="Path to sweep config YAML"
    )

    args = parser.parse_args()

    # Change to script directory to ensure relative paths work
    script_dir = Path(__file__).parent
    import os
    os.chdir(script_dir)

    if args.create:
        sweep_id = create_sweep(args.project, args.config)
        if args.count is not None and args.count > 0:
            print(f"\nStarting agent with {args.count} runs...")
            run_agent(sweep_id, args.project, args.count)
    elif args.sweep_id:
        print(f"Joining sweep {args.sweep_id}...")
        run_agent(args.sweep_id, args.project, args.count)
    else:
        print("Error: Must specify either --create or --sweep_id")
        parser.print_help()
        exit(1)


if __name__ == "__main__":
    main()
