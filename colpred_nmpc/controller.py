import numpy as np
import casadi as cs
import torch
from collision_predictor_mpc.ocp import Ocp
from collision_predictor_mpc.model import unicycle as model_quad
from collision_predictor_mpc.colpred.depth_state_check import Colpred
from collision_predictor_mpc.utils import euler2rot, quat2rot


class NMPC:
    """Wrapper around the NMPC controller with depth image-based collision prediction.
    """
    def __init__(self, config):
        self.config = config
        torch.no_grad()
        colpred_bounds = [-0.1, 0.5]  # col score bounds (and lower numerical margin)

        ## model
        self.model = model_quad.Quad(config, config.size_latent)

        ## colpred
        self.colpred = None
        if self.config.mission['enable_colpred']:
            ## colpred
            self.colpred = Colpred(size_latent=config.size_latent)
            self.colpred.load_weights(filename=config.weight_file)
            self.colpred.encoder.device = 'cuda'
            self.colpred.linear.device = 'cpu'

            ## argument functions for colpred cost and constraints
            def cp_args(x, u, p):
                W_R_Co = euler2rot(p[model_quad.P_ETA0:model_quad.P_ETA0+3])
                W_p_Co = p[model_quad.P_P0:model_quad.P_P0+3]
                W_p_B = x[:3]
                # W_R_B = euler2rot([0,0,x[3]])
                # W_R_B = quat2rot(x[3:7])
                W_R_B = quat2rot([x[3], 0, 0, x[4]])
                Co_p_C = W_R_Co.T @ (W_R_B @ config.B_p_C + W_p_B - W_p_Co)
                return p[model_quad.P_FLAG], cs.vertcat(Co_p_C[0]/config.depth_max, Co_p_C[1]/config.depth_max/cs.tan(config.hfov), Co_p_C[2]/config.depth_max/cs.tan(config.vfov)), p[model_quad.P_LATENT:]
                # Bo_x = euler2rot(p[model_quad.P_ETA0:model_quad.P_ETA0+3]).T @ (x[:3] - p[model_quad.P_P0:model_quad.P_P0+3])
                # return p[model_quad.P_FLAG], cs.vertcat(Bo_x[0]/config.depth_max, Bo_x[1]/config.depth_max/cs.tan(config.hfov), Bo_x[2]/config.depth_max/cs.tan(config.vfov)), p[model_quad.P_LATENT:]
            def cp_args_noflag(x, u, p):
                return cp_args(x, u, cs.vertcat(1, p[1:]))

            ## add model cost and constraints
            self.model.add_eval(self.colpred.infer_sigm, cp_args_noflag)
            if self.config.mission['colpred_cost']:
                self.model.add_cost(self.colpred.infer_sigm, cp_args, 200)
            if self.config.mission['colpred_constraint']:
                self.model.add_constraint_path(self.colpred.infer_exp, cp_args, colpred_bounds)
                self.model.add_constraint_term(self.colpred.infer_exp, cp_args, colpred_bounds)

        ## generate model and OCP
        self.model.gen_acados_model()
        self.T = self.config.mission['T']
        self.N = self.config.mission['N']
        self.ocp = Ocp(self.model, self.T, self.N)

        ## init variables
        self.x0 = None
        self.p = np.zeros(self.model.np)
        self.y = np.zeros((self.model.ny, self.N))
        self.yN = np.zeros(self.model.nyN)


    ## parameter setters
    def set_colpred_flag(self, flag):
        """Set flag to enable/disable colpred cost and constraints."""
        self.p[model_quad.P_FLAG] = flag


    def set_new_image(self, depth_img, pos, att):
        """Process new image into latent space and save current state."""
        if self.colpred is not None:
            W_p_Bo = pos
            W_R_Bo = euler2rot(att)
            self.p[model_quad.P_P0:model_quad.P_P0+3] = np.array((W_R_Bo @ self.config.B_p_C + W_p_Bo)).flatten()
            self.p[model_quad.P_ETA0:model_quad.P_ETA0+3] = att
            depth_tensor = torch.tensor(depth_img, device=self.colpred.encoder.device).unsqueeze(0).unsqueeze(0)
            self.p[model_quad.P_LATENT:] = self.colpred.encoder.encode(depth_tensor).detach().cpu().numpy()


    ## control iteration
    def set_x0(self, x0):
        """Current state feedback."""
        self.x0 = x0


    def solve(self):
        """Solve the NLP."""
        try:
            self.ocp.solve(self.x0, self.y, self.yN, self.p)
            fail = False
        except:
            fail = True
        return fail


    def get_u(self):
        """Returns last computed system inputs."""
        u = self.ocp.get_u()
        return np.array([u[0]*self.config.limits['ax'], u[1]*self.config.limits['vz'], u[2]*self.config.limits['wz']])


    def eval(self, k):
        """Return value of model evaluate vector for shooting node k."""
        if self.model.eval_vec.shape[0] == 0:
            return [0]
        else:
            return np.array(self.model.eval(self.ocp.solver.get(k, 'x'), self.ocp.get_u(), self.p)).flatten()


    ## reference setters
    def set_xy_ref(self, xy):
        """Set (x,y) position reference reference."""
        self.y[0,:] = xy[0]
        self.y[1,:] = xy[1]
        self.yN = self.y[:self.model.nyN,-1]


    def set_z_ref(self, z):
        """Set altitude reference."""
        self.y[2,:] = z
        self.yN = self.y[:self.model.nyN,-1]


    def set_world_vel_ref(self, v):
        """Set world frame velocity reference vector."""
        self.y[3,:] = v[0]
        self.y[4,:] = v[1]
        self.yN = self.y[:self.model.nyN,-1]


    def set_waypoint(self, p, speed):
        """Set (x,y) waypoint reference and nominal speed."""
        v = p - self.x0[:2]
        norm_v = np.linalg.norm(v)
        if norm_v > speed:
            v = v/norm_v*speed
        if norm_v < 0.1:
            v = [0, 0]
        self.y[3,:] = v[0]
        self.y[4,:] = v[1]
        self.yN = self.y[:self.model.nyN,-1]


    def set_xy_weights(self, w):
        """Set (x,y) positions weights."""
        for i in range(self.N):
            W = self.model.W
            W[:2] = w
            W[4] = w
            for j in range(self.N):
                self.ocp.solver.cost_set(j, 'W', np.diag(W))
            self.ocp.solver.cost_set(self.N, 'W', np.diag(W[:self.model.nyN]))
