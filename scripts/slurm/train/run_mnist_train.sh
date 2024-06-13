#!/bin/bash
#SBATCH --partition=gpu              # Partition (job queue)
#SBATCH --requeue                    # Return job to the queue if preempted
#SBATCH --job-name=mnist             # Assign a short name to your job
#SBATCH --nodes=1                    # Number of nodes you require
#SBATCH --ntasks=1                   # Total number of tasks across all nodes
#SBATCH --cpus-per-task=1            # Cores per task
#SBATCH --mem=16G                    # Real memory required
#SBATCH --time=12:00:00               # Total run time limit
#SBATCH --gres=gpu:1                                    
#SBATCH --exclude=gpu009,gpu[005-006],cuda[001-008],volta[001-003],gpuk[001-006]      # exclude specific GPUs
#SBATCH --output=/scratch/df630/log/mnist/train/%x_%N_jobid_%j.out                                # STDOUT output file
#SBATCH --error=/scratch/df630/log/mnist/train/%x_%N_jobid_%j.err                                 # STDERR output file
echo "Job started on $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node List: $SLURM_JOB_NODELIST"
echo "Job Name: $SLURM_JOB_NAME"
mmlsquota --block-size=auto scratch cache
source ~/.bashrc
conda activate conditional_rate_matching
cd ../../../experiments/nist
echo "Initial GPU Status:"
nvidia-smi
python3 crm_nist_single_run.py --source "$1" \
                               --target "$2" \
                               --model "$3" \
                               --gamma $4 \
                               --results_dir "amarel" \
                               --coupling "$5" \
                               --thermostat "$6" \
                               --epochs $7 \
                               --timesteps $8 \
                               --id "$SLURM_JOB_ID"
echo "Job finished on $(date)"
