#!/bin/bash

#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=HPSearch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=05:00:00
#SBATCH --output=outputs/slurm_search_%A_%a.out
#SBATCH --array=0-8  # Total jobs = 3 datasets * 6 models = 18 (indices 0-17)

# --- Setup Environment ---
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

# --- Define Search Space ---
# The order here is important and must not be changed without updating the array size
datasets=("aircraft" "flowers" "dogs-custom")
models=("lorentz_forward" "lorentz_angle" "lorentz_dist")

# --- Calculate Task-Specific Parameters from Array Index ---
# This uses bash arithmetic to assign a unique dataset/model pair to each task
num_models=${#models[@]}
model_idx=$((SLURM_ARRAY_TASK_ID % num_models))
dataset_idx=$((SLURM_ARRAY_TASK_ID / num_models))

dataset=${datasets[$dataset_idx]}
model=${models[$model_idx]}

# --- Log Job Information ---
echo "--- Starting Job Array Task ${SLURM_ARRAY_TASK_ID} ---"
echo "Submission Time: $(date)"
echo "Host: $(hostname)"
echo "Dataset: ${dataset}"
echo "Model: ${model}"
echo "GPU status:"
nvidia-smi

# --- Run Hyperparameter Search ---
# The python script will now run with the specific combination for this task
python exp2/hyperparam_search.py \
    --features_path "${PATH_SCRATCH_SHARE}/features" \
    --dataset "${dataset}" \
    --model "${model}" \
    --n_trials 250 \
    --epochs_per_trial 70 \
    --wandb_project "FineGrained-HPSearch"

echo "--- Task ${SLURM_ARRAY_TASK_ID} Complete ---"
