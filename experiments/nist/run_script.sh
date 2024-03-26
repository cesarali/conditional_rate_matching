# !/bin/bash
# cuda = sys.argv[1]
# experiment = sys.argv[2]
# gamma = sys.argv[3]
# ensemble = sys.argv[4]

# Run the NIST experiments

python crm_nist_single_run.py 0 "emnist_to_mnist_unet_cfm_expthermostat" 10.0 0
python crm_nist_single_run.py 0 "emnist_to_mnist_unet_cfm_expthermostat" 2.0 0
python crm_nist_single_run.py 0 "emnist_to_mnist_unet_cfm_expthermostat" 4.0 0
python crm_nist_single_run.py 0 "emnist_to_mnist_unet_cfm_expthermostat" 6.0 0
python crm_nist_single_run.py 0 "emnist_to_mnist_unet_cfm_expthermostat" 8.0 0
python crm_nist_single_run.py 0 "emnist_to_mnist_unet_cfm_expthermostat" 20.0 0
python crm_nist_single_run.py 0 "emnist_to_mnist_unet_cfm_expthermostat" 50.0 0



