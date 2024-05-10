#!/bin/bash

# Variables
sources=("emnist" "fashion" "noise")   
targets=("mnist") 
models=("unet")
couplings=("OTlog" "OTl2")
thermostats=("Constant")
gammas=(1.0 0.001 0.005 0.01 0.05 0.1 0.25 0.5 0.75 1.25 1.5 1.75 2.0 2.5 3.0 3.5 4.0 4.5 5.0)
epochs=100
timesteps=100
#
for source in "${sources[@]}"
do
  for target in "${targets[@]}"
  do
    for model in "${models[@]}"
    do
      for coupling in "${couplings[@]}"
      do
        for thermostat in "${thermostats[@]}"
        do
          for gamma in "${gammas[@]}"
          do
            # Submit the job
            sbatch run.sh $source $target $model $coupling $thermostat $gamma $epochs $timesteps
          done
        done
      done
    done
  done
done