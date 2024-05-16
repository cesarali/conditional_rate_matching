#!/bin/bash

lrs=(0.01 0.001 0.0001)
dims=(64 128 256)
tembs=(16 32 64 128)
#
echo "Submitting jobs"

for lr in "${lrs[@]}"
do
  for dim in "${dims[@]}"
  do
    for temb in "${tembs[@]}"
    do
      sbatch run_graph.sh $lr $dim $temb 
    done
  done
done