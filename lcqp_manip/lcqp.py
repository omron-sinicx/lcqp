#
# Copyright (c) 2022 OMRON SINIC X Corporation
#

import casadi
import numpy as np
from .box import Box
from .position_limit_constraint import PositionLimitConstraint
from .force_direction_constraint import ForceDirectionConstraint
from .configuration_cost import ConfigurationCost
from .two_objects_distance_cost import TwoObjectsDistanceCost
from .cost_collection import CostCollection


class LCQP:
    """ The LCQP problem. """
    def __init__(self, objects: list, contacts: list):
        """ Construct a LCQP. 

            Args: 
                objects: Objects of this LCQP problem.
                contacts: Contacts between the objects.
        """
        self.objects = objects
        self.contacts = contacts
        self.position_limits = []
        self.force_direction_limits = []

        # collect variables
        self.q = None
        self.dq = None
        for e in objects:
            if e.q is not None:
                if self.q is not None:
                    self.q = casadi.vertcat(self.q, e.q)
                    self.dq = casadi.vertcat(self.dq, e.dq)
                else:
                    self.q = e.q
                    self.dq = e.dq
        self.f = None
        self.df = None
        for e in contacts:
            if self.f is not None:
                self.f = casadi.vertcat(self.f, e.f)
                self.df = casadi.vertcat(self.df, e.df)
            else:
                self.f = e.f
                self.df = e.df
        self.opt_var = casadi.vertcat(self.dq, self.df)
        self.dimq = self.q.shape[0]
        self.dimv = self.dq.shape[0]
        self.dimf = self.f.shape[0]
        self.dimdf = self.df.shape[0]

        # collect variable bounds
        self.c = None
        self.lbc = None
        self.ubc = None
        for e in objects:
            if e.dq is not None:
                c, lbc, ubc = e.get_opt_var_bounds()
                if self.c is not None:
                    self.c = casadi.vertcat(self.c, c) 
                    self.lbc = casadi.vertcat(self.lbc, lbc)
                    self.ubc = casadi.vertcat(self.ubc, ubc)
                else:
                    self.c = c
                    self.lbc = lbc
                    self.ubc = ubc
        for e in contacts:
            c, lbc, ubc = e.get_opt_var_bounds()
            self.c = casadi.vertcat(self.c, lbc) 
            self.lbc = casadi.vertcat(self.lbc, lbc)
            self.ubc = casadi.vertcat(self.ubc, ubc)

        # collect complementarity constraints
        self.cp = None
        self.cf = None
        for e in contacts:
            cp, cf = e.get_contact_complementarity_constraint()
            if self.cp is None:
                self.cp = cp
                self.cf = cf
            else:
                self.cp = casadi.vertcat(self.cp, cp) 
                self.cf = casadi.vertcat(self.cf, cf) 

        self.cost = CostCollection(objects, contacts)
        self.config_cost = None
        self.dist_cost = None


    def set_force_balance(self, obj: Box):
        f_balance = obj.get_force_moment_balance_constraint(self.contacts)
        self.c = casadi.vertcat(self.c, f_balance) 
        self.lbc = casadi.vertcat(self.lbc, casadi.SX.zeros(f_balance.shape[0])) 
        self.ubc = casadi.vertcat(self.ubc, casadi.SX.zeros(f_balance.shape[0])) 


    def set_position_limit(self, point, surface, limit_to_surface_rot: float=0, margin: float=0., inv_dir: bool=False):
        position_limit_constraint = PositionLimitConstraint(point, surface, limit_to_surface_rot, margin, inv_dir)
        c, lbc, ubc = position_limit_constraint.get_constraints()
        self.c = casadi.vertcat(self.c, c) 
        self.lbc = casadi.vertcat(self.lbc, lbc)
        self.ubc = casadi.vertcat(self.ubc, ubc)
        self.position_limits.append(position_limit_constraint)


    def set_foce_direction_limit(self, contact, margin: float=0., inv_dir: bool=False):
        foce_direction_limit_constraint = ForceDirectionConstraint(contact, margin, inv_dir)
        c, lbc, ubc = foce_direction_limit_constraint.get_constraints()
        self.c = casadi.vertcat(self.c, c) 
        self.lbc = casadi.vertcat(self.lbc, lbc)
        self.ubc = casadi.vertcat(self.ubc, ubc)
        self.force_direction_limits.append(foce_direction_limit_constraint)


    def set_config_cost(self, q_ref, q_weight, v_weight, f_weight, slack_penalty):
        self.config_cost = ConfigurationCost(self.objects, self.contacts, q_ref=q_ref,
                                             q_weight=q_weight, v_weight=v_weight, f_weight=f_weight,
                                             slack_penalty=slack_penalty) 
        self.config_cost.set_opt_vars(self.q, self.f, self.dq, self.df)
        self.cost.append(self.config_cost)


    def set_two_object_distance_cost(self, obj1, obj2, dist_ref, dist_weight):
        self.dist_cost = TwoObjectsDistanceCost(self.objects, self.contacts, obj1, obj2,
                                                dist_ref, dist_weight)
        self.dist_cost.set_opt_vars(self.q, self.f, self.dq, self.df)
        self.cost.append(self.dist_cost)


    def update_q_val(self, q: np.ndarray):
        """ Updates the internal q value (configuration). 
        """
        dimq_begin = 0 
        for e in self.objects:
            if e.dimq > 0:
                e.update_q_val(q[dimq_begin:dimq_begin+e.dimq])
                dimq_begin = dimq_begin + e.dimq
            else:
                pass


    def update_dq_val(self, dq: np.ndarray):
        """ Updates the internal dq value (generalized velocity). 
        """
        dimq_begin = 0 
        for e in self.objects:
            if e.dimq > 0:
                e.update_dq_val(dq[dimq_begin:dimq_begin+e.dimq])
                dimq_begin = dimq_begin + e.dimq
            else:
                pass