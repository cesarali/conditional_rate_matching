# !/bin/bash
# cuda = sys.argv[1]
# experiment = sys.argv[2]
# gamma = sys.argv[3]
# ensemble = sys.argv[4]

# Run the NIST experiments

python crm_nist_single_run.py 0 "emnist_to_mnist_unet_cfm_expthermostat_0.3max" 10.0 0
python crm_nist_single_run.py 0 "emnist_to_mnist_unet_cfm_expthermostat_0.3max_OT" 10.0 0




