#!/bin/bash

# Directory containing the folders
DIRECTORY="/scratch/df630/conditional_rate_matching/results/crm/images"
#DIRECTORY="/home/df630/conditional_rate_matching/results/crm/images"

# Loop over each folder in the directory
for folder in "$DIRECTORY"/*; do
    if [ -d "$folder" ]; then  
        folder_name=$(basename "$folder") 
        if [ -f "$folder/run/best_model.tr" ]; then
            sbatch run_mnist_analysis.sh "$folder_name" "TRUE" 0.0
            echo "Submitted job for $folder_name"
        else
            echo "best_model.tr not found in $folder_name, skipping job submission."
        fi
    fi
done
