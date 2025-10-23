#!/bin/bash

#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=TrainMultiRunArray  # Changed job name for clarity
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=05:00:00 # This should be enough time for 10 runs
#SBATCH --output=outputs/slurm_output_%A_%a.out
#SBATCH --array=1-18 # This should match the number of lines in your hyperparameters.txt

module purge
module load 2024
module load Anaconda3/2024.06-1
source activate hypernn

cd $HOME/hyperbolic_fully_connected
source "$HOME/hyperbolic_fully_connected/.env"


# --- Script Arguments ---
# These arguments are fixed for all runs controlled by this script.
FEATURES_PATH="${PATH_SCRATCH_SHARE}/features"
NUM_EPOCHS=120
NUM_RUNS=10 # The number of runs with different seeds for each hyperparameter set
INITIAL_SEED=1 # The starting seed, will be incremented for each run
WANDB_PROJECT="classification-head-experiments-multi-run"

# Path to the file with hyperparameter combinations
HYPERPARAM_FILE="exp2/hyperparameters_same.txt"

# Extract the line of parameters for the current array task ID
# These are: --dataset --model --lr --curvature --batch_size --weight_decay
PARAMS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $HYPERPARAM_FILE)

echo "---"
echo "Starting Slurm Job ID: $SLURM_JOB_ID, Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "This task will perform $NUM_RUNS runs."
echo "GPU status:"
nvidia-smi
echo "---"
echo "Running with parameters from file: $PARAMS"
echo "Appending additional arguments..."
echo "---"

# Execute the multi-run training script with parameters from the file AND the additional arguments
# Note the change to the python script name and the new arguments
python exp2/train_multi_run.py \
    $PARAMS \
    --features_path $FEATURES_PATH \
    --epochs $NUM_EPOCHS \
    --num_runs $NUM_RUNS \
    --seed $INITIAL_SEED \
    --wandb_project $WANDB_PROJECT

echo "Task $SLURM_ARRAY_TASK_ID finished."