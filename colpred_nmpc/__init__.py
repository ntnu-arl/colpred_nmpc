import os

COLPREDMPC_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
COLPREDMPC_DATA_DIR = os.path.join(COLPREDMPC_ROOT_DIR, '../collision_data')
COLPREDMPC_LOG_DIR = os.path.join(COLPREDMPC_ROOT_DIR, 'logs')
COLPREDMPC_WEIGHT_DIR = os.path.join(COLPREDMPC_ROOT_DIR, 'logs')
COLPREDMPC_CONFIG_DIR = os.path.join(COLPREDMPC_ROOT_DIR, 'config')

# print('PACKAGE ROOT DIR:', COLPREDMPC_ROOT_DIR)
