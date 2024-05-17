#!/bin/bash
#SBATCH --partition=gpu              # Partition (job queue)
#SBATCH --requeue                    # Return job to the queue if preempted
#SBATCH --job-name=analysis          # Assign a short name to your job
#SBATCH --nodes=1                    # Number of nodes you require
#SBATCH --ntasks=1                   # Total number of tasks across all nodes
#SBATCH --cpus-per-task=1            # Cores per task
#SBATCH --mem=10G                    # Real memory required
#SBATCH --time=0:20:00               # Total run time limit
#SBATCH --gres=gpu:1                                            # Number of GPUs
#SBATCH --exclude=gpu[005-006],cuda[001-008],volta[001-003]     # exclude specific GPUs
#SBATCH --output=./log/%x_%N_jobid_%j.out                       # STDOUT output file
#SBATCH --error=./log/%x_%N_jobid_%j.err                        # STDERR output file

echo "Job started on $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node List: $SLURM_JOB_NODELIST"
echo "Job Name: $SLURM_JOB_NAME"

source ~/.bashrc
conda activate conditional_rate_matching
cd ../../../experiments/nist

echo "Initial GPU Status:"
nvidia-smi

python3 analysis_nist.py --experiment "$1" --overwrite "$2" --timepsilon $3

echo "Job finished on $(date)"
