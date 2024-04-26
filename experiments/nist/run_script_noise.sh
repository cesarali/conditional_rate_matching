# !/bin/bash

# cuda = sys.argv[1]
# experiment = sys.argv[2]
# thermostat = sys.argv[3]
# gamma = sys.argv[4]
# max = sys.argv[5]
# ensemble = sys.argv[6]
# dataset0 = experiment.split('_')[0]

# Run the NIST experiments

##############
# Constant
##############


python crm_nist_single_run_1.py  --source "noise" --target "mnist" --model "unet"  --thermostat "Constant" --gamma 0.1 --epochs 100 --timesteps 200
# python crm_nist_single_run_1.py --source "noise" --target "mnist" --model "unet"  --thermostat "Constant" --gamma 3.0 --epochs 100 --timesteps 200
# python crm_nist_single_run_1.py --source "noise" --target "mnist" --model "unet"  --thermostat "Constant" --gamma 4.0 --epochs 100 --timesteps 200
