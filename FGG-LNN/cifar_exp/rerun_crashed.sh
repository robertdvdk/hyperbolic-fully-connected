#!/bin/bash -l
#SBATCH --job-name=cifar_rerun_crashed
#SBATCH --output=./%x_%j.out
#SBATCH --time=05:00:00
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=128
#SBATCH --account=a132

set -x
ulimit -c 0
conda activate

echo "Rerunning 8 crashed sweep configs"
echo "GPUs available: $(nvidia-smi -L)"

GROUP="sweep-rkmny5zg-rerun"

# Batch 1: 4 configs in parallel (one per GPU)
CUDA_VISIBLE_DEVICES=0 uv run python main.py --learning_rate 0.1 --weight_decay 0.0001 --scheduler cosine --normalisation_mode clamp_scale --group $GROUP &
CUDA_VISIBLE_DEVICES=1 uv run python main.py --learning_rate 0.1 --weight_decay 0.0001 --scheduler steplr --normalisation_mode clamp_scale --group $GROUP &
CUDA_VISIBLE_DEVICES=2 uv run python main.py --learning_rate 0.1 --weight_decay 0.0001 --scheduler cosine --normalisation_mode centering_only --group $GROUP &
CUDA_VISIBLE_DEVICES=3 uv run python main.py --learning_rate 0.1 --weight_decay 0.0001 --scheduler steplr --normalisation_mode fix_gamma --group $GROUP &

wait
echo "Batch 1 complete"

# Batch 2: remaining 4 configs
CUDA_VISIBLE_DEVICES=0 uv run python main.py --learning_rate 0.1 --weight_decay 0.0005 --scheduler steplr --normalisation_mode fix_gamma --group $GROUP &
CUDA_VISIBLE_DEVICES=1 uv run python main.py --learning_rate 0.1 --weight_decay 0.0005 --scheduler steplr --normalisation_mode clamp_scale --group $GROUP &
CUDA_VISIBLE_DEVICES=2 uv run python main.py --learning_rate 0.1 --weight_decay 0.001 --scheduler steplr --normalisation_mode fix_gamma --group $GROUP &
CUDA_VISIBLE_DEVICES=3 uv run python main.py --learning_rate 0.1 --weight_decay 0.001 --scheduler cosine --normalisation_mode clamp_scale --group $GROUP &

wait
echo "Batch 2 complete"

echo "All 8 configs finished"