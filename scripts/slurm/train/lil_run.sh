#!/bin/bash
sleep 5h
sources=("noise" "fashion" "emnist")   
targets=("mnist") 
models=("unet")
couplings=("uniform" "OTlog" "OTl2")
thermostats=("Constant")
gammas=(0.001 0.005 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.25 1.5 1.75 2.0 2.5 3.0 3.5 4.0)
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
            sbatch run_mnist_train.sh $source $target $model $gamma $coupling $thermostat $epochs $timesteps
          done
        done
      done
    done
  done
done