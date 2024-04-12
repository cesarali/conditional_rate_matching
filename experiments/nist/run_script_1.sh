# !/bin/bash
# cuda = sys.argv[1]
# experiment = sys.argv[2]
# gamma = sys.argv[3]
# ensemble = sys.argv[4]

# Run the NIST experiments


python crm_nist_single_run.py 1 "emnist_to_mnist_unet" "Constant" 1.25
python crm_nist_single_run.py 1 "emnist_to_mnist_unet" "Constant" 1.0
python crm_nist_single_run.py 1 "emnist_to_mnist_unet" "Constant" 0.75
python crm_nist_single_run.py 1 "emnist_to_mnist_unet" "Constant" 0.5
python crm_nist_single_run.py 1 "emnist_to_mnist_unet" "Constant" 0.25
python crm_nist_single_run.py 1 "emnist_to_mnist_unet" "Constant" 0.05
python crm_nist_single_run.py 1 "emnist_to_mnist_unet" "Constant" 0.01