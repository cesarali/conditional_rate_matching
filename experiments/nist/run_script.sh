# !/bin/bash

# cuda = sys.argv[1]
# experiment = sys.argv[2]
# thermostat = sys.argv[3]
# gamma = sys.argv[4]
# max = sys.argv[5]
# ensemble = sys.argv[6]
# dataset0 = experiment.split('_')[0]

# Run the NIST experiments

python crm_nist_single_run.py 0 "emnist_to_mnist_unet" "Periodic" 6.0 0.05
python crm_nist_single_run.py 0 "emnist_to_mnist_unet" "Periodic" 4.0 0.05
python crm_nist_single_run.py 0 "emnist_to_mnist_unet" "Periodic" 2.0 0.05
python crm_nist_single_run.py 0 "emnist_to_mnist_unet" "Periodic" 6.0 0.5
python crm_nist_single_run.py 0 "emnist_to_mnist_unet" "Periodic" 4.0 0.5
python crm_nist_single_run.py 0 "emnist_to_mnist_unet" "Periodic" 2.0 0.5



