# !/bin/bash

# cuda = sys.argv[1]
# experiment = sys.argv[2]
# thermostat = sys.argv[3]
# gamma = sys.argv[4]
# max = sys.argv[5]
# ensemble = sys.argv[6]
# dataset0 = experiment.split('_')[0]

# Run the NIST experiments


# python crm_nist_single_run_1.py 1 "emnist_to_mnist_unet_att" 145 "Swish" "Constant" 2.0 0.0
# python crm_nist_single_run_1.py 1 "emnist_to_mnist_unet_att" 145 "Swish" "Constant" 1.5 0.0
# # python crm_nist_single_run_1.py 1 "emnist_to_mnist_unet_att" 145 "Swish" "Constant" 1.0 0.0
# python crm_nist_single_run_1.py 1 "emnist_to_mnist_unet_att" 145 "Swish" "Constant" 0.75 0.0
# python crm_nist_single_run_1.py 1 "emnist_to_mnist_unet_att" 145 "Swish" "Constant" 0.5 0.0
# python crm_nist_single_run_1.py 1 "emnist_to_mnist_unet_att" 145 "Swish" "Constant" 0.25 0.0
# python crm_nist_single_run_1.py 1 "emnist_to_mnist_unet_att" 145 "Swish" "Constant" 0.1 0.0
# python crm_nist_single_run_1.py 1 "emnist_to_mnist_unet_att" 145 "Swish" "Constant" 0.05 0.0
# python crm_nist_single_run_1.py 1 "emnist_to_mnist_unet_att" 145 "Swish" "Constant" 0.01 0.0

python crm_nist_single_run_1.py --cuda 0 --source "emnist" --target "mnist" --model "unet"  --thermostat "Constant" --gamma 2.5 --epochs 100 --timesteps 200
python crm_nist_single_run_1.py --cuda 0 --source "emnist" --target "mnist" --model "unet"  --thermostat "Constant" --gamma 3.0 --epochs 100 --timesteps 200
python crm_nist_single_run_1.py --cuda 0 --source "emnist" --target "mnist" --model "unet"  --thermostat "Constant" --gamma 3.5 --epochs 100 --timesteps 200
python crm_nist_single_run_1.py --cuda 0 --source "emnist" --target "mnist" --model "unet"  --thermostat "Constant" --gamma 4.0 --epochs 100 --timesteps 200
