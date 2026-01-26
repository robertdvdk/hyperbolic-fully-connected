#!/bin/bash -l
#SBATCH --job-name=cifar_finalruns
#SBATCH --output=./%x_%j.out
#SBATCH --time=03:00:00
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=128
#SBATCH --account=a132

set -x

ulimit -c 0

conda activate

echo "Starting 4 wandb agents for sweep: $SWEEP_ID"
echo "GPUs available: $(nvidia-smi -L)"

# Launch 4 agents, one per GPU
for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i uv run wandb agent "robert-vdklis/hyperbolic-fully-connected-FGG-LNN_cifar_exp/r9nemj61" &
    echo "Started agent $i on GPU $i (PID: $!)"
done

# Wait for all background jobs to complete
wait

echo "All agents finished"
