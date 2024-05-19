#!/bin/bash
sources=("fashion" "noise" "emnist")   
targets=("mnist") 
models=("unet")
couplings=("uniform")
thermostats=("Constant")
gammas=(0.001 0.005)
epochs=500
timesteps=500
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
            sbatch run_mnist_train.sh $source $target $model $gamma $coupling $thermostat $epochs $timesteps
          done
        done
      done
    done
  done
done