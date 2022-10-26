#
# Copyright (c) 2022 OMRON SINIC X Corporation
#

import numpy as np
import casadi

class TwoObjectsDistanceCost:
    """ Cost on the distance between two objects. """
    def __init__(self, objects, contacts, object1, object2, dist_ref=np.zeros(3), dist_weight=1.0):
        """ Construct a cost. 

            Args: 
                objects: List of the objects.
                contacts: List of the contacts.
                object1: The first object.
                object2: The second object.
                dist_ref: Reference distance.
                dist_weight: Weight of the cost.
        """
        self.dimq = 0
        for e in objects:
            self.dimq += e.dimq
        self.objects = objects
        self.object1 = object1
        self.object2 = object2
        assert dist_ref.size == 3
        self.dist_ref = dist_ref
        if isinstance(dist_weight, np.ndarray):
            assert(dist_weight.size == 3) 
            self.dist_weight = dist_weight 
        else:
            self.dist_weight = dist_weight * np.ones(3)
        self.dimdf = 5 * len(contacts)
        self.cost = None
        self.q = None
        self.f = None
        self.v = None
        self.df = None
        self.lcqp_var = None 
        self.var = None 
        self.H = None
        self.g = None
        self.gq = np.zeros(self.dimq)
        self.gv = np.zeros(self.dimq)
        self.gf = np.zeros(self.dimdf)
        self.H = None
        self.g = None

    def set_opt_vars(self, q, f, v, df):
        """ Sets the optimization variables of the LCQP. """
        self.q = q
        self.f = f
        self.v = v
        self.df = df
        self.lcqp_var = casadi.vertcat(v, df)
        self.var = casadi.vertcat(q, f, v, df)
        qdiff = self.object1.q[0:3] + self.object1.dq[0:3] - self.object2.q[0:3] - self.object2.dq[0:3] - self.dist_ref
        qdiff[0:2] = self.object1.R.T @ qdiff[0:2]
        self.cost = 0.5 * qdiff.T @ casadi.diag(self.dist_weight) @ qdiff
        g = casadi.jacobian(self.cost, self.lcqp_var)
        H = casadi.jacobian(g, self.lcqp_var)
        self.g = casadi.Function('config_cost_g', [self.var], [g]) 
        self.H = casadi.Function('config_cost_H', [self.var], [H]) 

    def get_qp_cost(self, q_val, f_val):
        """ Gets the QP cost function, i.e., the Hessian and linear term. """
        val = np.concatenate([q_val, f_val, np.zeros(self.dimq+self.dimdf)])
        g = np.array(self.g(val))
        H = np.array(self.H(val)) 
        return H, g