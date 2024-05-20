# !/bin/bash

###################
# FASHION -> DIGITS
###################

python crm_nist_single_run.py  --device "cuda:0" --source "fashion" --target "mnist" --model "unet" --coupling "OTlog" --thermostat "Constant" --gamma 0.75 --epochs 100 --timesteps 100
