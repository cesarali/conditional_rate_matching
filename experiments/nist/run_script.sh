# !/bin/bash

# cuda = sys.argv[1]
# experiment = sys.argv[2]
# thermostat = sys.argv[3]
# gamma = sys.argv[4]
# max = sys.argv[5]
# ensemble = sys.argv[6]
# dataset0 = experiment.split('_')[0]

# Run the NIST experiments


python crm_nist_single_run_1.py 2 "fashion_to_mnist_unet_att" 145 "Swish" "Exponential" 5.0 0.75
python crm_nist_single_run_1.py 2 "fashion_to_mnist_unet_att" 145 "Swish" "Exponential" 20.0 0.75
python crm_nist_single_run_1.py 2 "noise_to_mnist_unet_att" 145 "Swish" "Exponential" 5.0 0.75
python crm_nist_single_run_1.py 2 "noise_to_mnist_unet_att" 145 "Swish" "Exponential" 20.0 0.75


