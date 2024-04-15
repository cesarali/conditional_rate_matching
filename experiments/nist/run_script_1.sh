# !/bin/bash
# cuda = sys.argv[1]
# experiment = sys.argv[2]
# gamma = sys.argv[3]
# ensemble = sys.argv[4]

# Run the NIST experiments

python crm_nist_single_run_1.py 1 "fashion_to_mnist_unet_att" 145 "Swish" "Constant" 2.0 0.0
python crm_nist_single_run_1.py 1 "fashion_to_mnist_unet_att" 145 "Swish" "Constant" 1.5 0.0
python crm_nist_single_run_1.py 1 "fashion_to_mnist_unet_att" 145 "Swish" "Constant" 1.0 0.0
python crm_nist_single_run_1.py 1 "fashion_to_mnist_unet_att" 145 "Swish" "Constant" 0.75 0.0
python crm_nist_single_run_1.py 1 "fashion_to_mnist_unet_att" 145 "Swish" "Constant" 0.5 0.0
python crm_nist_single_run_1.py 1 "fashion_to_mnist_unet_att" 145 "Swish" "Constant" 0.25 0.0
python crm_nist_single_run_1.py 1 "fashion_to_mnist_unet_att" 145 "Swish" "Constant" 0.1 0.0
python crm_nist_single_run_1.py 1 "fashion_to_mnist_unet_att" 145 "Swish" "Constant" 0.05 0.0
python crm_nist_single_run_1.py 1 "fashion_to_mnist_unet_att" 145 "Swish" "Constant" 0.01 0.0

python crm_nist_single_run_1.py 1 "fashion_to_mnist_unet_att_OT" 145 "Swish" "Constant" 2.0 0.0
python crm_nist_single_run_1.py 1 "fashion_to_mnist_unet_att_OT" 145 "Swish" "Constant" 1.5 0.0
python crm_nist_single_run_1.py 1 "fashion_to_mnist_unet_att_OT" 145 "Swish" "Constant" 1.0 0.0
python crm_nist_single_run_1.py 1 "fashion_to_mnist_unet_att_OT" 145 "Swish" "Constant" 0.75 0.0
python crm_nist_single_run_1.py 1 "fashion_to_mnist_unet_att_OT" 145 "Swish" "Constant" 0.5 0.0
python crm_nist_single_run_1.py 1 "fashion_to_mnist_unet_att_OT" 145 "Swish" "Constant" 0.25 0.0
python crm_nist_single_run_1.py 1 "fashion_to_mnist_unet_att_OT" 145 "Swish" "Constant" 0.1 0.0
python crm_nist_single_run_1.py 1 "fashion_to_mnist_unet_att_OT" 145 "Swish" "Constant" 0.05 0.0
python crm_nist_single_run_1.py 1 "fashion_to_mnist_unet_att_OT" 145 "Swish" "Constant" 0.01 0.0