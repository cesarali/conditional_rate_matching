#!/bin/bash
#SBATCH --partition=gpu              # Partition (job queue)
#SBATCH --requeue                    # Return job to the queue if preempted
#SBATCH --job-name=cifar10             # Assign a short name to your job
#SBATCH --nodes=1                    # Number of nodes you require
#SBATCH --ntasks=1                   # Total number of tasks across all nodes
#SBATCH --cpus-per-task=1            # Cores per task
#SBATCH --mem=10G                    # Real memory required
#SBATCH --time=72:00:00              # Total run time limit
#SBATCH --gres=gpu:1             
#SBATCH --nodelist=gpu015,gpu016,gpu[019-026]  # Request specific nodes
#SBATCH --output=./log/%x_%N_jobid_%j.out                       # STDOUT output file
#SBATCH --error=./log/%x_%N_jobid_%j.err                        # STDERR output file

echo "Job started on $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node List: $SLURM_JOB_NODELIST"
echo "Job Name: $SLURM_JOB_NAME"

source ~/.bashrc
conda activate conditional_rate_matching
cd ../../../src/conditional_rate_matching/configs/experiments_configs/crm

echo "Initial GPU Status:"
nvidia-smi

python3 crm_experiments_Cifar.py 

echo "Job finished on $(date)"
