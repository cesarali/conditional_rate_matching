#!/bin/bash
#SBATCH --partition=gpu              # Partition (job queue)
#SBATCH --requeue                    # Return job to the queue if preempted
#SBATCH --job-name=graphs_gnn           # Assign a short name to your job
#SBATCH --nodes=1                    # Number of nodes you require
#SBATCH --ntasks=1                   # Total number of tasks across all nodes
#SBATCH --cpus-per-task=1            # Cores per task
#SBATCH --mem=16G                    # Real memory required
#SBATCH --time=24:00:00              # Total run time limit
#SBATCH --gres=gpu:1                 # Request one GPU
#SBATCH --nodelist=gpu015,gpu016,gpu[019-026]  # Request specific nodes
#SBATCH --output=./log/%x_%N_jobid_%j.out  # STDOUT output file
#SBATCH --error=./log/%x_%N_jobid_%j.err   # STDERR output file

echo "Job started on $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node List: $SLURM_JOB_NODELIST"
echo "Job Name: $SLURM_JOB_NAME"

source ~/.bashrc
conda activate conditional_rate_matching
cd ../../experiments/graph

echo "Initial GPU Status:"
nvidia-smi

python3 crm_graph_single_run.py --source "noise" \
                                --target "enzymes" \
                                --model "gnn" \
                                --gamma 1.0 \
                                --lr $1 \
                                --dim $2 \
                                --temb $3 \
                                --epochs 1000 \
                                --timepsilon 0.05

echo "Job finished on $(date)"
