#!/bin/bash
sources=("emnist" "fashion" "noise")   
targets=("mnist") 
models=("unet")
couplings=("OTlog" "OTl2" "uniform")
thermostats=("Constant")
gammas=(0.001 0.005 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.25 1.5 1.75 2.0 2.5 3.0 3.5 4.0 4.5 5.0)
epochs=100
timesteps=250
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