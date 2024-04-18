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

# python crm_nist_single_run_1.py 1 "noise_to_mnist_unet_att" 145 "Swish" "Constant" 0.01 0.0
# python crm_nist_single_run_1.py 1 "noise_to_mnist_unet_att" 145 "Swish" "Constant" 0.05 0.0
# python crm_nist_single_run_1.py 1 "noise_to_mnist_unet_att" 145 "Swish" "Constant" 0.1 0.0
# python crm_nist_single_run_1.py 1 "noise_to_mnist_unet_att" 145 "Swish" "Constant" 0.25 0.0
# python crm_nist_single_run_1.py 1 "noise_to_mnist_unet_att" 145 "Swish" "Constant" 0.5 0.0
# python crm_nist_single_run_1.py 1 "noise_to_mnist_unet_att" 145 "Swish" "Constant" 0.75 0.0
# python crm_nist_single_run_1.py 1 "noise_to_mnist_unet_att" 145 "Swish" "Constant" 1.0 0.0
# python crm_nist_single_run_1.py 1 "noise_to_mnist_unet_att" 145 "Swish" "Constant" 1.5 0.0
# python crm_nist_single_run_1.py 1 "noise_to_mnist_unet_att" 145 "Swish" "Constant" 2.0 0.0

##############
# Plateau
##############

# python crm_nist_single_run_1.py 2 "noise_to_mnist_unet_att_shift_0.4" 145 "Swish" "Plateau" 1.0 15.0
# python crm_nist_single_run_1.py 2 "noise_to_mnist_unet_att_shift_0.4" 145 "Swish" "Plateau" 0.5 15.0

##############
# Polynomial
##############

# python crm_nist_single_run_1.py 1 "noise_to_mnist_unet_att" 145 "Swish" "Polynomial" 1.0 1.0
# python crm_nist_single_run_1.py 1 "noise_to_mnist_unet_att" 145 "Swish" "Polynomial" 0.5 1.0

# python crm_nist_single_run_1.py 1 "noise_to_mnist_unet_att" 145 "Swish" "Polynomial" 1.0 2.0
# python crm_nist_single_run_1.py 1 "noise_to_mnist_unet_att" 145 "Swish" "Polynomial" 0.5 2.0

# python crm_nist_single_run_1.py 1 "noise_to_mnist_unet_att" 145 "Swish" "Polynomial" 1.0 4.0
# python crm_nist_single_run_1.py 1 "noise_to_mnist_unet_att" 145 "Swish" "Polynomial" 0.5 4.0

#############
# Exponential
#############

python crm_nist_single_run_1.py 2 "noise_to_mnist_unet_att" 145 "Swish" "Exponential" 10 1.0
python crm_nist_single_run_1.py 2 "noise_to_mnist_unet_att" 145 "Swish" "Exponential" 10 0.5
python crm_nist_single_run_1.py 2 "noise_to_mnist_unet_att" 145 "Swish" "Exponential" 10 0.1

python crm_nist_single_run_1.py 2 "noise_to_mnist_unet_att" 145 "Swish" "Exponential" 20 1.0
python crm_nist_single_run_1.py 2 "noise_to_mnist_unet_att" 145 "Swish" "Exponential" 20 0.5
python crm_nist_single_run_1.py 2 "noise_to_mnist_unet_att" 145 "Swish" "Exponential" 20 0.1
