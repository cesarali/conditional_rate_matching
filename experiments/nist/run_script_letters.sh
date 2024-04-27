# !/bin/bash

###################
# LETTERS -> DIGITS
###################

python crm_nist_single_run_1.py  --source "emnist" --target "mnist" --model "unet" --coupling "OTlog" --thermostat "Constant" --gamma 0.5 --epochs 100 --timesteps 200
python crm_nist_single_run_1.py  --source "emnist" --target "mnist" --model "unet" --coupling "OTl2" --thermostat "Constant" --gamma 0.5 --epochs 100 --timesteps 200
python crm_nist_single_run_1.py  --source "emnist" --target "mnist" --model "unet" --coupling "OTlog" --thermostat "Constant" --gamma 0.1 --epochs 100 --timesteps 200
python crm_nist_single_run_1.py  --source "emnist" --target "mnist" --model "unet" --coupling "OTl2" --thermostat "Constant" --gamma 0.1 --epochs 100 --timesteps 200
python crm_nist_single_run_1.py  --source "emnist" --target "mnist" --model "unet" --coupling "OTlog" --thermostat "Constant" --gamma 1.0--epochs 100 --timesteps 200
python crm_nist_single_run_1.py  --source "emnist" --target "mnist" --model "unet" --coupling "OTl2" --thermostat "Constant" --gamma 1.0 --epochs 100 --timesteps 200
