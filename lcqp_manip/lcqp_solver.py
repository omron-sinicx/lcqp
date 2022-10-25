import casadi
import numpy as np
import lcqpow
from .lcqp import LCQP


class LCQPSolver:
    """ The LCQP solver. """
    def __init__(self, lcqp: LCQP):
        """ Construct a solver. 

            Args: 
                lcqp: The LCQP problem.
        """
        self.lcqp = lcqp
        self.opt_vars = lcqp.opt_var
        self.dimq = lcqp.dimq
        self.dimv = lcqp.dimv
        self.dimf = lcqp.dimf
        self.dimdf = lcqp.dimdf

        # linear complementarity constraints
        cp = self.lcqp.cp
        cf = self.lcqp.cf
        Sp = casadi.jacobian(cp, self.opt_vars)
        lbSp = - cp 
        Sf = casadi.jacobian(cf, self.opt_vars)
        lbSf = - cf 

        # variable bounds and force balance constraints
        c = self.lcqp.c
        lbc = self.lcqp.lbc
        ubc = self.lcqp.ubc
        A = casadi.jacobian(c, self.opt_vars)
        lbA = lbc - c 
        ubA = ubc - c

        # LCQP dimensions 
        self.nV = self.opt_vars.shape[0]
        self.nC = c.shape[0]
        self.nComp = cp.shape[0]

        # numerical evaluation
        vars = casadi.vertcat(lcqp.q, lcqp.f, self.opt_vars) 
        self.Sp   = casadi.Function('Sp', [vars], [Sp])
        self.lbSp = casadi.Function('lbSp', [vars], [lbSp])
        self.Sf   = casadi.Function('Sf', [vars], [Sf])
        self.lbSf = casadi.Function('lbSf', [vars], [lbSf])
        self.A   = casadi.Function('A', [vars], [A])
        self.lbA = casadi.Function('lbA', [vars], [lbA])
        self.ubA = casadi.Function('ubA', [vars], [ubA])

        self.options = lcqpow.Options()
        self.options.setPrintLevel(lcqpow.PrintLevel.NONE)
        # self.options.setQPSolver(lcqpow.QPSolver.QPOASES_DENSE)
        self.options.setQPSolver(lcqpow.QPSolver.QPOASES_SPARSE)
        # self.options.setQPSolver(lcqpow.QPSolver.OSQP_SPARSE)
        # self.options.setQPSolver(lcqpow.QPSolver.OSQP_DENSE)

        self.options.setMaxRho(1.0e06)
        self.options.setStationarityTolerance(1.0e-03)
        self.primal_solution = None
        self.f_val = np.zeros(self.dimf)

    def solve(self, q_val: np.ndarray, f_val=None):
        """ Solves the LCQP problem. 

            Args: 
                q_val: Current configuration.
                f_val: Current contact forces.
        """
        if f_val is not None:
            self.f_val = f_val
        params = np.concatenate([q_val, self.f_val, np.zeros(self.opt_vars.shape[0])])
        H, g = self.lcqp.cost.get_qp_cost(q_val, self.f_val)
        Sp   = np.array(self.Sp(params))
        lbSp = np.array(self.lbSp(params))
        Sf   = np.array(self.Sf(params))
        lbSf = np.array(self.lbSf(params))
        A    = np.array(self.A(params))
        lbA  = np.array(self.lbA(params))
        ubA  = np.array(self.ubA(params))
        lcqp = lcqpow.LCQProblem(nV=self.nV, nC=self.nC, nComp=self.nComp)
        lcqp.setOptions(self.options)
        ret_val = lcqp.loadLCQP(H=H, g=g.T, S1=Sp.T, S2=Sf.T, lbS1=lbSp, lbS2=lbSf, A=A.T, lbA=lbA, ubA=ubA)
        if ret_val != lcqpow.ReturnValue.SUCCESSFUL_RETURN:
            print("Failed to load LCQP.")
            return False
        ret_val = lcqp.runSolver()
        if ret_val != lcqpow.ReturnValue.SUCCESSFUL_RETURN:
            print("Failed to solve LCQP.")
            return False
        self.primal_solution = lcqp.getPrimalSolution().copy()
        df = self.primal_solution[self.dimv:self.dimv+self.dimdf]
        for i in range(len(self.lcqp.contacts)):
            self.f_val[3*i:3*i+3] = self.f_val[3*i:3*i+3] + df[5*i:5*i+3]
        return True

    def set_print_level(self, print_level: lcqpow.PrintLevel):
        """ Sets the print level of the inner LCQP solver. 

            Args: 
                print_level: The print level.
        """
        self.options.setPrintLevel(print_level)

    def get_solution(self):
        """ Gets the primal solution of the LCQP problem. 
        """
        return self.primal_solution
