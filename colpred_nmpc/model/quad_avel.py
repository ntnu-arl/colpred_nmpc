import numpy as np
import casadi as cs
from casadi import vertcat, horzcat, diagcat, transpose, diag, inv, pinv, trace, norm_2  # arrays
from casadi import cos, sin, tan, tanh, acos, asin, atan, atan2, log10, exp, dot, sqrt, fabs, cross, fmin  # maths
from collision_predictor_mpc.model import Model
from collision_predictor_mpc.utils import *


P_FLAG = 0
P_P0 = 1
P_ETA0 = 4
P_VREF = 7
P_LATENT = 10

class Quad(Model):
    def __init__(self, config, size_latent=0):
        super().__init__('quad')

        ## model constants
        self.nx = 9
        self.nu = 4
        self.ny = 11  # z, att, vel, u
        self.nyN = 7
        self.nbu = 4
        self.np = 1+6+3+size_latent  # flag + x0[0:6] + v_ref + size of latent space for depth images

        ## parameters
        g = cs.vertcat(0,0,-9.81)
        self.gamma_max = 100
        self.avel_max = 10

        ## symbols
        self.gen_symbols()

        p = self.x[0:3]
        eta = self.x[3:6]
        v = self.x[6:9]
        gamma = self.u[0] * self.gamma_max
        w = self.u[1:4] * self.avel_max

        ## dynamics
        W_R_B = euler2rot(eta)
        T = deuler_avel_map(eta)

        self.f_expl = vertcat(
            v,
            T @ w,
            self.phy.g + gamma * W_R_B @ cs.vertcat(0,0,1)
        )
        self.f_impl = self.dx - self.f_expl

        ## cost
        self.W = np.array([
            10,  # pz
            2, 2, 0,  # roll pitch yaw
            5, 5, 10,  # vx vy vz
            0.1, 2, 2, 8,  # T, wx wy wz
        ])
        self.WN = self.W[:self.nyN]

        self.y = vertcat(self.x[2:], self.u)
        self.yN = self.y[:self.nyN]

        ## input bounds
        self.lbu = np.array([0,-1,-1,-1])
        self.ubu = np.array([1,1,1,1])


    def add_frustrum_constraint(self, safe_ball, depth_max, hfov, vfov):
        """Creates a set of constraints to contrain the motion to occur inside the camera fov."""
        def identity(x): return x
        def id_flag(x): return x[0]*x[1]
        def args_x(x,u,p):
            Bo_x = euler2rot(p[P_ETA0:P_ETA0+3]).T @ (x[:3] - p[P_P0:P_P0+3])
            return Bo_x[0]
        def args_bearing_h(x,u,p):
            Bo_x = euler2rot(p[P_ETA0:P_ETA0+3]).T @ (x[:3] + cs.vertcat(0.1,0,0) - p[P_P0:P_P0+3])
            return p[0], cs.atan2(Bo_x[1], Bo_x[0])
        def args_bearing_v(x,u,p):
            Bo_x = euler2rot(p[P_ETA0:P_ETA0+3]).T @ (x[:3] + cs.vertcat(0.1,0,0) - p[P_P0:P_P0+3])
            return p[0], cs.atan2(Bo_x[2], Bo_x[0])
        self.add_constraint_path(identity, args_x, [-safe_ball, depth_max])


    def add_yaw_align_cost(self, weight):
        """Create a cost to align the forward angle (ie, camera principal axis) with the reference velocity."""
        def forward_angle(args):
            yaw = args[0]
            vref = args[1]
            normv = cs.norm_2(vref)
            vnorm = cs.if_else(normv < 1e-1, cs.vertcat(0,0,0), vref/normv)
            W_forward = euler2rot([0,0,yaw]) @ cs.vertcat(1,0,0)
            return cs.if_else(normv < 1e-1, 0, (1 - cs.dot(W_forward, vnorm)))
        def args_forward_angle(x,u,p):
            return x[5], p[P_VREF:P_VREF+3]
        self.add_cost(forward_angle, args_forward_angle, weight)
