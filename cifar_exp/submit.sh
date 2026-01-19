#!/bin/bash -l
#SBATCH --job-name=train_500epochs
#SBATCH --output=./%x_%j.out
#SBATCH --time=06:00:00
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --account=a132

set -x

ulimit -c 0

conda activate
uv run main.py