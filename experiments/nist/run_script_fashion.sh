# !/bin/bash
# cuda = sys.argv[1]
# experiment = sys.argv[2]
# gamma = sys.argv[3]
# ensemble = sys.argv[4]

# Run the NIST experiments


##############
# Polynomial
##############

python crm_nist_single_run_1.py 0 "fashion_to_mnist_unet_att" 145 "Swish" "Polynomial" 1.0 1.0
python crm_nist_single_run_1.py 0 "fashion_to_mnist_unet_att" 145 "Swish" "Polynomial" 0.5 1.0

python crm_nist_single_run_1.py 0 "fashion_to_mnist_unet_att" 145 "Swish" "Polynomial" 1.0 2.0
python crm_nist_single_run_1.py 0 "fashion_to_mnist_unet_att" 145 "Swish" "Polynomial" 0.5 2.0

python crm_nist_single_run_1.py 0 "fashion_to_mnist_unet_att" 145 "Swish" "Polynomial" 1.0 4.0
python crm_nist_single_run_1.py 0 "fashion_to_mnist_unet_att" 145 "Swish" "Polynomial" 0.5 4.0

#############
# Exponential
#############

# python crm_nist_single_run_1.py 3 "fashion_to_mnist_unet_att" 145 "Swish" "Exponential" 10 1.0
# python crm_nist_single_run_1.py 3 "fashion_to_mnist_unet_att" 145 "Swish" "Exponential" 10 0.5
# python crm_nist_single_run_1.py 3 "fashion_to_mnist_unet_att" 145 "Swish" "Exponential" 10 0.1

# python crm_nist_single_run_1.py 3 "fashion_to_mnist_unet_att" 145 "Swish" "Exponential" 20 1.0
# python crm_nist_single_run_1.py 3 "fashion_to_mnist_unet_att" 145 "Swish" "Exponential" 20 0.5
# python crm_nist_single_run_1.py 3 "fashion_to_mnist_unet_att" 145 "Swish" "Exponential" 20 0.1