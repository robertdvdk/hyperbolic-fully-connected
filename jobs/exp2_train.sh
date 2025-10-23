#!/bin/bash

#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=TrainArray
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=05:00:00
#SBATCH --output=outputs/slurm_output_%A_%a.out
#SBATCH --array=1-18

module purge
module load 2024
module load Anaconda3/2024.06-1
source activate hypernn

cd $HOME/hyperbolic_fully_connected
source "$HOME/hyperbolic_fully_connected/.env"

# Define shared path and number of epochs
NUM_EPOCHS=120

# Path to the file with hyperparameter combinations
HYPERPARAM_FILE="exp2/hyperparameters.txt"

# Extract the line of parameters for the current array task ID
PARAMS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $HYPERPARAM_FILE)

echo "---"
echo "Starting Slurm Job ID: $SLURM_JOB_ID, Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "GPU status:"
nvidia-smi
echo "---"
echo "Running with parameters from file: $PARAMS"
echo "Appending additional arguments: --features_path ${PATH_SCRATCH_SHARE}/features --epochs $NUM_EPOCHS"
echo "---"

# Execute the training script with parameters from the file AND the additional arguments
python exp2/train_from_features.py \
    $PARAMS \
    --features_path ${PATH_SCRATCH_SHARE}/features \
    --epochs $NUM_EPOCHS

echo "Task $SLURM_ARRAY_TASK_ID finished."