import numpy as np
import yaml
import os
from collision_predictor_mpc import COLPREDMPC_CONFIG_DIR, COLPREDMPC_WEIGHT_DIR
import casadi as cs

class Config:
    """Read all yaml config file for a given run and store into single object."""
    def __init__(self, config_file):
        ## open mission config file
        with open(config_file, 'r') as f:
            self.mission = yaml.load(f, Loader=yaml.FullLoader)

        ## camera config
        self.camera_config_file = os.path.join(COLPREDMPC_CONFIG_DIR, self.mission['camera_config'])
        with open(self.camera_config_file, 'r') as f:
            self.camera = yaml.load(f, Loader=yaml.FullLoader)
        self.B_p_C = self.camera['extrinsics']['position']
        # self.B_R_C = euler2rot(self.camera['extrinsics']['orientation'])  # unused because identity
        self.depth_max = self.camera['depth_max']
        self.hfov = self.camera['hfov']
        self.vfov = self.camera['vfov']

        ## model config
        self.model_config_file = os.path.join(COLPREDMPC_CONFIG_DIR, self.mission['drone_config'])
        with open(self.model_config_file, 'r') as f:
            self.model = yaml.load(f, Loader=yaml.FullLoader)
        self.limits = self.model['limits']
        self.weight_file = os.path.join(COLPREDMPC_WEIGHT_DIR, self.mission['weight_file'])
        self.size_latent = self.mission['size_latent']
