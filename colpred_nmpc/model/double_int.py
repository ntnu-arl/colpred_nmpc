import numpy as np
import casadi as cs
from casadi import vertcat, horzcat, diagcat, transpose, diag, inv, pinv, trace, norm_2  # arrays
from casadi import cos, sin, tan, tanh, acos, asin, atan, atan2, log10, exp, dot, sqrt, fabs, cross, fmin  # maths
from collision_predictor_mpc.model import Model
from collision_predictor_mpc.utils import *


P_FLAG = 0
P_P0 = 1
P_ETA0 = 4
P_LATENT = 7

class Quad(Model):
    def __init__(self, config, size_latent=0):
        super().__init__('quad')

        ## model constants
        self.nx = 6
        self.nu = 3
        self.ny = 9
        self.nyN = 6
        self.nbu = 3
        self.np = 1+6+size_latent  # flag + x0[0:6] + size of latent space for depth images

        ## symbols
        self.gen_symbols()

        amax = 10
        p = self.x[0:3]
        v = self.x[3:6]
        a = self.u * amax

        ## dynamics
        self.f_expl = cs.vertcat(
            v,
            a
        )
        self.f_impl = self.dx - self.f_expl

        ## cost
        self.W = np.array([
            0, 0, 10,  # p
            5, 5, 10,  # v
            1, 1, 1  # a
        ])
        self.WN = self.W[:self.nyN]

        self.y = cs.vertcat(p, v, a)
        self.yN = cs.vertcat(p, v)

        ## input bounds
        self.lbu = np.array([-1,-1,-1])
        self.ubu = np.array([1,1,1])
