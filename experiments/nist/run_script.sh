# !/bin/bash

# cuda = sys.argv[1]
# experiment = sys.argv[2]
# thermostat = sys.argv[3]
# gamma = sys.argv[4]
# max = sys.argv[5]
# ensemble = sys.argv[6]
# dataset0 = experiment.split('_')[0]

# Run the NIST experiments

python crm_nist_single_run_1.py 1 "noise_to_mnist_unet_optimized" "Constant" 1.25 0.0
python crm_nist_single_run_1.py 1 "noise_to_mnist_unet_optimized" "Constant" 1.0 0.0
python crm_nist_single_run_1.py 1 "noise_to_mnist_unet_optimized" "Constant" 0.75 0.0
python crm_nist_single_run_1.py 1 "noise_to_mnist_unet_optimized" "Constant" 0.5 0.0
python crm_nist_single_run_1.py 1 "noise_to_mnist_unet_optimized" "Constant" 0.25 0.0
python crm_nist_single_run_1.py 1 "noise_to_mnist_unet_optimized" "Constant" 0.1 0.0
python crm_nist_single_run_1.py 1 "noise_to_mnist_unet_optimized" "Constant" 0.05 0.0
python crm_nist_single_run_1.py 1 "noise_to_mnist_unet_optimized" "Constant" 0.01 0.0

python crm_nist_single_run.py 1 "noise_to_mnist_unet" "Constant" 3.0 0.0
python crm_nist_single_run.py 1 "noise_to_mnist_unet" "Constant" 2.0 0.0
python crm_nist_single_run.py 1 "noise_to_mnist_unet" "Constant" 1.5 0.0
# python crm_nist_single_run.py 1 "noise_to_mnist_unet" "Constant" 1.25 0.0
# python crm_nist_single_run.py 1 "noise_to_mnist_unet" "Constant" 1.0 0.0
# python crm_nist_single_run.py 1 "noise_to_mnist_unet" "Constant" 0.75 0.0
# python crm_nist_single_run.py 1 "noise_to_mnist_unet" "Constant" 0.5 0.0
# python crm_nist_single_run.py 1 "noise_to_mnist_unet" "Constant" 0.25 0.0
# python crm_nist_single_run.py 1 "noise_to_mnist_unet" "Constant" 0.1 0.0
# python crm_nist_single_run.py 1 "noise_to_mnist_unet" "Constant" 0.05 0.0
# python crm_nist_single_run.py 1 "noise_to_mnist_unet" "Constant" 0.01 0.0
python crm_nist_single_run.py 1 "noise_to_mnist_unet" "Constant" 0.005 0.0

python crm_nist_single_run.py 1 "fashion_to_mnist_unet" "Constant" 3.0 0.0
python crm_nist_single_run.py 1 "fashion_to_mnist_unet" "Constant" 2.0 0.0
python crm_nist_single_run.py 1 "fashion_to_mnist_unet" "Constant" 1.5 0.0
# python crm_nist_single_run.py 1 "fashion_to_mnist_unet_OT" "Constant" 1.25 0.0
# python crm_nist_single_run.py 1 "fashion_to_mnist_unet_OT" "Constant" 1.0 0.0
# python crm_nist_single_run.py 1 "fashion_to_mnist_unet_OT" "Constant" 0.75 0.0
# python crm_nist_single_run.py 1 "fashion_to_mnist_unet_OT" "Constant" 0.5 0.0
# python crm_nist_single_run.py 1 "fashion_to_mnist_unet_OT" "Constant" 0.25 0.0
# python crm_nist_single_run.py 1 "fashion_to_mnist_unet_OT" "Constant" 0.1 0.0
# python crm_nist_single_run.py 1 "fashion_to_mnist_unet_OT" "Constant" 0.05 0.0
# python crm_nist_single_run.py 1 "fashion_to_mnist_unet_OT" "Constant" 0.01 0.0
python crm_nist_single_run.py 1 "fashion_to_mnist_unet" "Constant" 0.005 0.0

python crm_nist_single_run.py 1 "fashion_to_mnist_unet_OT" "Constant" 3.0 0.0
python crm_nist_single_run.py 1 "fashion_to_mnist_unet_OT" "Constant" 2.0 0.0
python crm_nist_single_run.py 1 "fashion_to_mnist_unet_OT" "Constant" 1.5 0.0
# python crm_nist_single_run.py 1 "fashion_to_mnist_unet_OT" "Constant" 1.25 0.0
# python crm_nist_single_run.py 1 "fashion_to_mnist_unet_OT" "Constant" 1.0 0.0
# python crm_nist_single_run.py 1 "fashion_to_mnist_unet_OT" "Constant" 0.75 0.0
# python crm_nist_single_run.py 1 "fashion_to_mnist_unet_OT" "Constant" 0.5 0.0
# python crm_nist_single_run.py 1 "fashion_to_mnist_unet_OT" "Constant" 0.25 0.0
# python crm_nist_single_run.py 1 "fashion_to_mnist_unet_OT" "Constant" 0.1 0.0
# python crm_nist_single_run.py 1 "fashion_to_mnist_unet_OT" "Constant" 0.05 0.0
# python crm_nist_single_run.py 1 "fashion_to_mnist_unet_OT" "Constant" 0.01 0.0
python crm_nist_single_run.py 1 "fashion_to_mnist_unet_OT" "Constant" 0.005 0.0


