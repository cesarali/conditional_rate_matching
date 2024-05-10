#!/bin/bash
#SBATCH --partition=gpu              # Partition (job queue)
#SBATCH --requeue                    # Return job to the queue if preempted
#SBATCH --job-name=mnist             # Assign a short name to your job
#SBATCH --nodes=1                    # Number of nodes you require
#SBATCH --ntasks=1                   # Total number of tasks across all nodes
#SBATCH --cpus-per-task=1            # Cores per task
#SBATCH --mem=10G                    # Real memory required
#SBATCH --time=6:00:00              # Total run time limit
#SBATCH --gres=gpu:1                                            # Number of GPUs
#SBATCH --exclude=gpu009,gpu[005-006],cuda[001-008],volta[001-003]     # exclude specific GPUs
#SBATCH --output=./log/%x_%N_jobid_%j.out                       # STDOUT output file
#SBATCH --error=./log/%x_%N_jobid_%j.err                        # STDERR output file

source ~/.bashrc
conda activate conditional_rate_matching
cd ../../experiments/nist

echo "Initial GPU Status:"
nvidia-smi

python3 crm_nist_single_run_1.py --source "$1" \
                                 --target "$2" \
                                 --model "$3" \
                                 --coupling "$4" \
                                 --thermostat "$5" \
                                 --gamma $6 \
                                 --epochs $7 \
                                 --timesteps $8
