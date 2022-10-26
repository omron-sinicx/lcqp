#
# Copyright (c) 2022 OMRON SINIC X Corporation
#

import numpy as np

class ConfigurationCost:
    """ Configuration space cost. """
    def __init__(self, objects, contacts, q_ref=None, q_weight=1.0, v_weight=1.0, f_weight=0.001, slack_penalty=1.0e08):
        """ Construct a cost. 

            Args: 
                objects: List of the objects.
                contacts: List of the contacts.
                q_ref: Reference configuration.
                q_weight: Weight on the configuration.
                v_weight: Weight on the generalized velocity.
                f_weight: Weight on the contact forces.
                slack_penalty: Penalty parameter of the slack variables.
        """
        self.dimq = 0
        for e in objects:
            self.dimq += e.dimq
        self.objects = objects
        self.contacts = contacts
        if q_ref is not None:
            self.q_ref = q_ref
        else:
            self.q_ref = np.zeros(self.dimq)
        if isinstance(q_weight, np.ndarray):
            self.q_weight = q_weight 
        else:
            self.q_weight = q_weight * np.ones(self.dimq)
        if isinstance(v_weight, np.ndarray):
            self.v_weight = v_weight 
        else:
            self.v_weight = v_weight * np.ones(self.dimq)
        self.dimdf = 5 * len(contacts)
        if isinstance(f_weight, np.ndarray):
            self.f_weight = f_weight 
        else:
            self.f_weight = f_weight * np.ones(self.dimdf)
        for i in range(len(self.contacts)):
            self.f_weight[i*5+4] = slack_penalty
        self.gq = np.zeros(self.dimq)
        self.gv = np.zeros(self.dimq)
        self.gf = np.zeros(self.dimdf)
        self.H = 0.5 * np.diag(np.concatenate([self.q_weight+self.v_weight, self.f_weight]))
        self.g = None

    def set_opt_vars(self, q, f, v, df):
        """ Sets the optimization variables of the LCQP. """
        pass

    def get_qp_cost(self, q_val, f_val):
        """ Gets the QP cost function, i.e., the Hessian and linear term. """
        self.gv = np.diag(self.q_weight) @ (q_val-self.q_ref) 
        self.g = np.concatenate([self.gv, self.gf])
        return self.H, self.g