# !/bin/bash

# cuda = sys.argv[1]
# experiment = sys.argv[2]
# thermostat = sys.argv[3]
# gamma = sys.argv[4]
# max = sys.argv[5]
# ensemble = sys.argv[6]
# dataset0 = experiment.split('_')[0]

# Run the NIST experiments

# python crm_nist_single_run.py 0 "emnist_to_mnist_unet_OT" "Periodic" 6.0 0.05
# python crm_nist_single_run.py 0 "emnist_to_mnist_unet_OT" "Periodic" 4.0 0.05
# python crm_nist_single_run.py 0 "emnist_to_mnist_unet_OT" "Periodic" 2.0 0.05
# python crm_nist_single_run.py 0 "emnist_to_mnist_unet_OT" "Periodic" 6.0 0.5
# python crm_nist_single_run.py 0 "emnist_to_mnist_unet_OT" "Periodic" 4.0 0.5
# python crm_nist_single_run.py 0 "emnist_to_mnist_unet_OT" "Periodic" 2.0 0.5

python crm_nist_single_run.py 0 "fashion_to_mnist_unet_OT" "Constant" 1.25 0.0
python crm_nist_single_run.py 0 "fashion_to_mnist_unet_OT" "Constant" 1.0 0.0
python crm_nist_single_run.py 0 "fashion_to_mnist_unet_OT" "Constant" 0.75 0.0
python crm_nist_single_run.py 0 "fashion_to_mnist_unet_OT" "Constant" 0.5 0.0
python crm_nist_single_run.py 0 "fashion_to_mnist_unet_OT" "Constant" 0.25 0.0
python crm_nist_single_run.py 0 "fashion_to_mnist_unet_OT" "Constant" 0.1 0.0
python crm_nist_single_run.py 0 "fashion_to_mnist_unet_OT" "Constant" 0.05 0.0
python crm_nist_single_run.py 0 "fashion_to_mnist_unet_OT" "Constant" 0.01 0.0