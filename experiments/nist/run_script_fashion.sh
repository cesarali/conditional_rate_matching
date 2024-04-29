# !/bin/bash

###################
# FASHION -> DIGITS
###################

python crm_nist_single_run_1.py  --device "cuda:1" --source "fashion" --target "mnist" --model "unet" --coupling "uniform" --thermostat "Constant" --gamma 0.75 --epochs 100 --timesteps 100
