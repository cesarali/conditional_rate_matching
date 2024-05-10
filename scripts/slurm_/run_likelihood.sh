#!/bin/bash
#SBATCH --partition=gpu              # Partition (job queue)
#SBATCH --requeue                    # Return job to the queue if preempted
#SBATCH --job-name=NLL               # Assign a short name to your job
#SBATCH --nodes=1                    # Number of nodes you require
#SBATCH --ntasks=1                   # Total number of tasks across all nodes
#SBATCH --cpus-per-task=1            # Cores per task
#SBATCH --mem=16G                    # Real memory required
#SBATCH --time=3:00:00               # Total run time limit
#SBATCH --gres=gpu:1                                            # Number of GPUs
#SBATCH --exclude=gpu009,gpu[005-006],cuda[001-008],volta[001-003]     # exclude specific GPUs
#SBATCH --output=./log/%x_%N_jobid_%j.out                       # STDOUT output file
#SBATCH --error=./log/%x_%N_jobid_%j.err                        # STDERR output file

source ~/.bashrc
conda activate conditional_rate_matching
cd ../../experiments/nist/loglikelihood/

echo "Initial GPU Status:"
nvidia-smi

python3 loglikelihood.py --fwd_process "$1" --bck_process "$2" --timesteps $3