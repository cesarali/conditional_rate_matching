#!/bin/bash

# Directory containing the folders
DIRECTORY="/scratch/df630/conditional_rate_matching/results/crm/images"
#DIRECTORY="/home/df630/conditional_rate_matching/results/crm/images"

# Loop over each folder in the directory
for folder in "$DIRECTORY"/*; do
    if [ -d "$folder" ]; then  
        folder_name=$(basename "$folder") 
        if [ -f "$folder/best_model.tr" ]; then
            echo "best model is here!--------"
            rm "$folder/model_checkpoint_50.tr"
            rm "$folder/model_checkpoint_100.tr"
            rm "$folder/sample_gen_x0.dat"
        else
            echo "nothing to delete"
            rm -r "$folder"
        
        fi
    fi
done