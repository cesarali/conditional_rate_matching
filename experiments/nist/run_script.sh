# !/bin/bash

# cuda = sys.argv[1]
# experiment = sys.argv[2]
# thermostat = sys.argv[3]
# gamma = sys.argv[4]
# max = sys.argv[5]
# ensemble = sys.argv[6]
# dataset0 = experiment.split('_')[0]

# Run the NIST experiments


python crm_nist_single_run_1.py 1 "emnist_to_mnist_unet_att" 145 "Swish" "Exponential" 2.0 0.0
python crm_nist_single_run_1.py 1 "emnist_to_mnist_unet_att" 145 "Swish" "Constant" 1.5 0.0


