# !/bin/bash
# cuda = sys.argv[1]
# experiment = sys.argv[2]
# gamma = sys.argv[3]
# ensemble = sys.argv[4]

# Run the NIST experiments
python crm_nist_single_run.py 1 "emnist_to_mnist_unet_cfm_expthermostat_OT" 10.0 0
python crm_nist_single_run.py 1 "emnist_to_mnist_unet_cfm_expthermostat_OT" 2.0 0
python crm_nist_single_run.py 1 "emnist_to_mnist_unet_cfm_expthermostat_OT" 4.0 0
python crm_nist_single_run.py 1 "emnist_to_mnist_unet_cfm_expthermostat_OT" 6.0 0
python crm_nist_single_run.py 1 "emnist_to_mnist_unet_cfm_expthermostat_OT" 8.0 0
python crm_nist_single_run.py 1 "emnist_to_mnist_unet_cfm_expthermostat_OT" 20.0 0
python crm_nist_single_run.py 1 "emnist_to_mnist_unet_cfm_expthermostat_OT" 50.0 0