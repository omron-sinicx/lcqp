#
# Copyright (c) 2022 OMRON SINIC X Corporation
#

import math
import numpy as np
from .lcqp import LCQP
from .lcqp_solver import LCQPSolver


class PlanarSimulation:
    def __init__(self, lcqp: LCQP, dt: float, tsim: float):
        self.objs = lcqp.objects
        self.contacts = lcqp.contacts
        self.dimq = 0
        for obj in self.objs:
            self.dimq += obj.dimq
        self.dt = dt
        self.tsim = tsim

    def run(self, lcqp_solver: LCQPSolver, q0: np.ndarray, verbose=True, debug=False):
        q = q0.copy()
        if verbose:
            print('q: ', q)
        sim_steps = math.floor(self.tsim/self.dt)
        q_data = [q0]
        v_data = []
        f_data = [lcqp_solver.f_val]
        if debug:
            lcqp_solver.lcqp.update_q_val(q)
            for e in lcqp_solver.lcqp.contacts:
                e.print()
            for e in lcqp_solver.lcqp.position_limits:
                e.print()
        success = True 
        for _ in range(sim_steps):
            success = lcqp_solver.solve(q)
            if not success:
                break
            s = lcqp_solver.get_solution()
            v = s[:self.dimq]
            alpha = 1.0
            q = (q + alpha * v).copy()
            if verbose:
                print('q: ', q)
            q_data.append(q.copy())
            v_data.append(v.copy())
            f_data.append(lcqp_solver.f_val.copy())
            if debug:
                lcqp_solver.lcqp.update_q_val(q)
                for e in lcqp_solver.lcqp.contacts:
                    e.print()
                for e in lcqp_solver.lcqp.position_limits:
                    e.print()
        return success, q_data, v_data, f_data