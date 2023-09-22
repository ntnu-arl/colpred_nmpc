import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver

class Ocp:
    def __init__(self, model, T, N):
        self.model = model
        self.T = T
        self.N = N

        ## ocp size
        self.ocp_ac = AcadosOcp()
        self.ocp_ac.model = model.model_ac
        self.ocp_ac.dims.nx = model.nx
        self.ocp_ac.dims.nu = model.nu
        self.ocp_ac.dims.ny = model.ny
        self.ocp_ac.dims.ny_e = model.nyN
        self.ocp_ac.dims.nbx = model.nbx
        self.ocp_ac.dims.nbx_e = model.nbxN
        self.ocp_ac.dims.nbu = model.nbu
        self.ocp_ac.dims.nh = model.nh
        self.ocp_ac.dims.nh_e = model.nhN
        self.ocp_ac.dims.N = N
        self.ocp_ac.parameter_values = model.p_val

        ## cost
        self.ocp_ac.cost.W = np.diag(model.W)
        self.ocp_ac.cost.W_e = np.diag(model.WN)

        self.ocp_ac.cost.cost_type = 'NONLINEAR_LS'
        self.ocp_ac.cost.cost_type_e = 'NONLINEAR_LS'

        self.ocp_ac.cost.yref = np.zeros(model.ny)
        self.ocp_ac.cost.yref_e = np.zeros(model.nyN)

        ## constraints
        self.ocp_ac.constraints.x0 = np.zeros(model.nx)
        self.ocp_ac.constraints.lbx = model.lbx
        self.ocp_ac.constraints.ubx = model.ubx
        self.ocp_ac.constraints.idxbx = model.idxbx
        self.ocp_ac.constraints.lbx_e = model.lbxN
        self.ocp_ac.constraints.ubx_e = model.ubxN
        self.ocp_ac.constraints.idxbx_e = model.idxbxN
        self.ocp_ac.constraints.lbu = model.lbu
        self.ocp_ac.constraints.ubu = model.ubu
        self.ocp_ac.constraints.idxbu = np.linspace(0,model.nbu-1,model.nbu)
        self.ocp_ac.constraints.lh = model.lh
        self.ocp_ac.constraints.uh = model.uh
        self.ocp_ac.constraints.lh_e = model.lhN
        self.ocp_ac.constraints.uh_e = model.uhN

        ## solver parameters
        ## see https://docs.acados.org/python_interface/index.html#acados_template.acados_ocp.AcadosOcpOptions
        self.ocp_ac.solver_options.print_level = 0
        self.ocp_ac.solver_options.integrator_type = 'ERK'  # ERK, IRK, GNSF, DISCRETE, LIFTED_IRK
        self.ocp_ac.solver_options.tf = T
        ## NLP solver parameters
        self.ocp_ac.solver_options.nlp_solver_type = 'SQP_RTI'
        # self.ocp_ac.solver_options.nlp_solver_type = 'SQP'
        self.ocp_ac.solver_options.rti_phase = 0  # 0: both; 1: preparation; 2: feedback
        ## QP solver parameters
        self.ocp_ac.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        # self.ocp_ac.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        # self.ocp_ac.solver_options.qp_solver_cond_N = N  # number of shooting nodes with partial condensing (N: no condensing)
        self.ocp_ac.solver_options.hpipm_mode = 'ROBUST'
        self.ocp_ac.solver_options.qp_solver_iter_max = 100
        self.ocp_ac.solver_options.qp_solver_warm_start = 1  # 0: cold; 1: primal warm; 2: primal+dual warm
        # self.ocp_ac.solver_options.warm_start_first_qp = 1
        ## Hessian approx
        self.ocp_ac.solver_options.hessian_approx = 'GAUSS_NEWTON'
        # self.ocp_ac.solver_options.hessian_approx = 'EXACT'
        # self.ocp_ac.solver_options.exact_hess_dyn = 0
        # self.ocp_ac.solver_options.exact_hess_cost = 0
        self.ocp_ac.solver_options.levenberg_marquardt = 20.

        ## create mpc solver
        self.solver = AcadosOcpSolver(self.ocp_ac)
        self.solver.options_set('warm_start_first_qp', True)

        ## init return variables
        self.u = np.zeros(model.nu)
        self.t = 0


    def init_x(self, x0):
        for j in range(self.N+1):
            self.solver.set(j, 'x', x0)


    def solve(self, x0, y, yN, p):
        self.ocp_ac.constraints.x0 = x0
        self.solver.set(0, 'x', x0)
        for j in range(self.N):
            self.solver.cost_set(j, 'yref', y[:,j])
            self.solver.set(j, 'p', p)
        self.solver.cost_set(self.N, 'yref', yN)
        self.solver.set(self.N, 'p', p)
        self.u = self.solver.solve_for_x0(x0)
        self.t = self.solver.get_stats('time_tot')


    def get_u(self):
        return self.u


    def get_t(self):
        return float(self.t)
