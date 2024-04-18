# !/bin/bash
# cuda = sys.argv[1]
# experiment = sys.argv[2]
# gamma = sys.argv[3]
# ensemble = sys.argv[4]

# Run the NIST experiments

python crm_distort_nist_single.py 1 "corrupted_to_mnist_unet_att" 145 "Swish" "Constant" 0.05 0.0
python crm_distort_nist_single.py 1 "corrupted_to_mnist_unet_att" 145 "Swish" "Constant" 0.5 0.0
