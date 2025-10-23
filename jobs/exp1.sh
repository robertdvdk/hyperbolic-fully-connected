#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=Exp1LayerRuntimes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=07:00:00
#SBATCH --output=outputs/slurm_output_%A.out

module purge
module load 2024
module load Anaconda3/2024.06-1
source activate hypernn

cd $HOME/hyperbolic_fully_connected

echo "GPU status:"
nvidia-smi

python -c "import torch; print(torch.__version__); print(torch.version.cuda)"

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

echo "Running experiment 1: Layer Runtimes"
python -u exp1/layer_runtimes.py
