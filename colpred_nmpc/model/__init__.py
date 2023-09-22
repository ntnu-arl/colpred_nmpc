import numpy as np
from acados_template import AcadosModel
import casadi as cs
import yaml


class Model:
    """Wrapper around the AcadosModel class, that contains all the necessary extra info for NMPC."""
    def __init__(self, name):
        self.name = name
        self.nx = 0  # number of states
        self.nu = 0  # number of outputs
        self.nz = 0  # number of algebraic states
        self.np = 0  # number of parameters
        self.ny = 0  # number of outputs
        self.nyN = 0  # number of terminal outputs
        self.nbx = 0  # number of bounds on states
        self.nbxN = 0  # number of terminal bounds on states
        self.nbu = 0  # number of bounds on inputs
        self.nh = 0  # number of general nonlinear constraints
        self.nhN = 0  # number of terminal general nonlinear constraints

        self.p_val = np.array([])

        self.idxbx = np.array([])
        self.lbx = np.array([])
        self.ubx = np.array([])

        self.idxbxN = np.array([])
        self.lbxN = np.array([])
        self.ubxN = np.array([])

        self.lbu = np.array([])
        self.ubu = np.array([])

        self.lh = np.array([])
        self.uh = np.array([])
        self.lh = np.array([])
        self.uh = np.array([])
        self.lhN = np.array([])
        self.uhN = np.array([])

        self.lsbx = np.array([])
        self.usbx = np.array([])

        self.eval_vec = np.array([])

        self.model_ac = None
        self.gen_symbols()



    def gen_symbols(self):
        """Convenience function to generate all symbolic vectors. Should be called after all the model sizes are set."""
        self.x = cs.MX.sym('x', self.nx, 1)
        self.dx = cs.MX.sym('dx', self.nx, 1)
        self.u = cs.MX.sym('u', self.nu, 1)
        self.z = cs.MX.sym('z', self.nz, 1)
        self.p = cs.MX.sym('p', self.np, 1)
        self.p_val = np.zeros(self.np)
        self.h = cs.MX.sym('h', self.nh, 1)
        self.hN = cs.MX.sym('hN', self.nhN, 1)
        self.eval = cs.Function('eval', [self.x, self.u, self.p], [self.eval_vec])


    def gen_acados_model(self):
        """Generate the Acados model and fill all fields from self. Should be called after the model is fully defined."""
        self.model_ac = AcadosModel()
        self.model_ac.name = self.name
        self.model_ac.f_impl_expr = self.f_impl
        self.model_ac.f_expl_expr = self.f_expl
        self.model_ac.x = self.x
        self.model_ac.xdot = self.dx
        self.model_ac.u = self.u
        self.model_ac.z = self.z
        self.model_ac.p = self.p
        self.model_ac.con_h_expr = self.h
        self.model_ac.con_h_expr_e = self.hN
        self.model_ac.cost_y_expr = self.y
        self.model_ac.cost_y_expr_e = self.yN


    def add_eval(self, function, args):
        """Add an evaluation variable to be returned by the eval() function.
        function    -- returns the value to be evaluated
        args        -- takes as input (x, u, p) and returns the subset that is required as input for function
        """
        self.eval_vec = cs.vertcat(self.eval_vec, function(args(self.x, self.u, self.p)))
        self.eval = cs.Function('eval', [self.x, self.u, self.p], [self.eval_vec])


    def add_cost(self, function, args, weight):
        """Add a term to the running cost function."""
        self.W = np.concatenate([self.W[:self.nyN], [weight], self.W[self.nyN:]])
        self.y = cs.vertcat(self.y[:self.nyN], function(args(self.x, self.u, self.p)), self.y[self.nyN:])
        self.ny += 1


    def add_constraint_path(self, function, args, bounds):
        """Add a path constraint."""
        self.nh += 1
        # self.nsh += 1
        if (self.nh == 1):
            self.lh = np.array([bounds[0]])
            self.uh = np.array([bounds[1]])
        else:
            self.lh = np.concatenate((self.lh, [bounds[0]]))
            self.uh = np.concatenate((self.uh, [bounds[1]]))
        self.h = cs.vertcat(self.h, function(args(self.x, self.u, self.p)))


    def add_constraint_term(self, function, args, bounds):
        """Add a terminal constraint."""
        self.nhN += 1
        if (self.nhN == 1):
            self.lhN = np.array([bounds[0]])
            self.uhN = np.array([bounds[1]])
        else:
            self.lhN = np.concatenate((self.lhN, [bounds[0]]))
            self.uhN = np.concatenate((self.uhN, [bounds[1]]))
        self.hN = cs.vertcat(self.hN, function(args(self.x, self.u, self.p)))
