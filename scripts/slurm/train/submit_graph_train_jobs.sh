#!/bin/bash

lrs=(0.001 0.0005)
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
      sbatch run_graph_train.sh $lr $dim $temb 
    done
  done
done

#!/bin/bash
sources=("noise")   
targets=("enzymes") 
models=("gnn")
gammas=(1.0)
lrs=(0.001 0.0005)
dims=(64 128 256)
couplings=("OTlog" "OTl2" "uniform")
thermostats=("Constant")
epochs=1000
#
for source in "${sources[@]}"
do
  for target in "${targets[@]}"
  do
    for model in "${models[@]}"
    do
      for gamma in "${gammas[@]}"
      do
        for lr in "${lrs[@]}"
        do
          for dim in "${dims[@]}"
          do
            for coupling in "${couplings[@]}"
            do
              for thermostat in "${thermostats[@]}"
              do
                sbatch run_graph_train.sh $source $target $model $gamma $lr $dim $coupling $thermostat $epochs
              done
            done
          done
        done
      done
    done
  done
done
