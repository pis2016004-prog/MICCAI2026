#!/bin/bash
#SBATCH --job-name=MED92  # Job name
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks-per-node=1          # Number of tasks per node (usually 1 for multi-GPU)
#SBATCH --cpus-per-task=24           # Number of CPU cores per task
#SBATCH --mem=32G                    # Memory per node
#SBATCH --gres=gpu:volta:1           # Number of GPUs per node
#SBATCH --time=2-00:00:00            # Max job runtime (2 days)
#SBATCH --partition=gpu              # Partition type
#SBATCH --output=slurm-%j.log        # Output log file

export MASTER_PORT=12355
export WORLD_SIZE=24
echo "WORLD_SIZE="$WORLD_SIZE

# Get the master node address
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

# Activate Conda Environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate mamba-attn

# Ensure GPUs are visible
nvidia-smi

# 🚀 Run the training script using `torchrun`
sh training_scripts/run_training_acdc.sh
#python test.py  --test_path ./data/Synapse/test_vol_h5  
