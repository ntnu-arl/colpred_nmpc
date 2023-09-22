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
        self.nx = 6  # x, y, z, qw, qz, vx (body)
        self.nu = 3  # ax, vz, wz (body)
        self.ny = 8
        self.nyN = 5
        self.nbx = 1
        self.nbxN = 1
        self.nbu = self.nu
        self.np = 1+6+size_latent  # flag + x0[0:6] + size of latent space for depth images

        ## symbols
        self.gen_symbols()

        p = self.x[0:3]
        q = cs.vertcat(self.x[3], 0, 0, self.x[4]) ; q /= norm_2(q)
        yaw = quat2euler(q)[2]
        vx = self.x[-1]

        ## dynamics
        ax = self.u[0] * config.limits['ax']
        vz = self.u[1] * config.limits['vz']
        wz = self.u[2] * config.limits['wz']
        dq = hamilton_prod(q, cs.vertcat(0, 0, 0, wz)) / 2

        self.f_expl = cs.vertcat(
            vx * cs.cos(yaw),
            vx * cs.sin(yaw),
            vz,
            dq[0],
            dq[3],
            ax,
        )
        self.f_impl = self.dx - self.f_expl

        ## cost
        self.W = np.array([
            0, 0, 2,  # p
            5, 5,  # vx vy (world)
            3, 1, 1,  # u
        ])
        self.WN = self.W[:self.nyN]

        self.y = cs.vertcat(p, vx * cs.cos(yaw), vx * cs.sin(yaw), self.u)
        self.yN = cs.vertcat(p, vx * cs.cos(yaw), vx * cs.sin(yaw))

        ## state/input bounds
        self.idxbx = np.array([5])
        self.lbx = np.array([-0.01])
        self.ubx = np.array([config.limits['vx']])
        self.idxbxN = self.idxbx
        self.lbxN = self.lbx
        self.ubxN = self.ubx
        self.lbu = np.array([-1, -1, -1])
        self.ubu = np.array([1, 1, 1])
