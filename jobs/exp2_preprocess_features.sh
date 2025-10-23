#!/bin/bash

#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=Exp1LayerRuntimes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=05:00:00
#SBATCH --output=outputs/slurm_output_%A.out

module purge
module load 2024
module load Anaconda3/2024.06-1
source activate hypernn

cd $HOME/hyperbolic_fully_connected

# Load PATH_SCRATCH_SHARE (and other env vars) from project .env if present
if [ -f "$HOME/hyperbolic_fully_connected/.env" ]; then
    # shellcheck disable=SC1090
    source "$HOME/hyperbolic_fully_connected/.env"
fi

echo "GPU status:"
nvidia-smi

# for dataset in aircraft flowers dogs-custom cub; do
for dataset in aircraft flowers dogs-custom; do
    echo "Processing dataset: $dataset"
    python exp2/preprocess_features.py \
        --dataset $dataset \
        --data_path ${PATH_SCRATCH_SHARE}/data \
        --features_path ${PATH_SCRATCH_SHARE}/features \
        --num_epochs 10 
done