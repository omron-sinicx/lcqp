#
# Copyright (c) 2022 OMRON SINIC X Corporation
#

import casadi
import numpy as np

class PositionLimitConstraint:
    """ Position limit constraint. """
    def __init__(self, point, surface, rotation_angle: float=0, margin: float=0, inv_dir: bool=False):
        """ Construct a position limit constraint. 

            Args: 
                point: A point whose position is constrained.
                surface: Position constraint surface.
                rotation_angle: Rotation angle of the surface.
                margin: Margin of the constraint.
                inv_dir: Flaf if the directin if inverted or not.
        """
        self.point = point
        self.surface = surface
        self.rotation_angle = rotation_angle
        self.margin = margin
        self.inv_dir = inv_dir

        if point.q is not None and surface.q is not None:
            assert point.dq is not None
            assert surface.dq is not None
            self.q = casadi.vertcat(point.q, surface.q)
            self.dq = casadi.vertcat(point.dq, surface.dq)
            self.q_val = np.concatenate([point.q_val, surface.q_val])
        elif point.q is not None:
            assert(point.dq is not None)
            self.q = point.q
            self.dq = point.dq
            self.q_val = point.q_val
        elif surface.q is not None:
            assert(surface.dq is not None)
            self.q = surface.q
            self.dq = surface.dq
            self.q_val = surface.q_val
        else:
            return NotImplementedError()
        sth = casadi.sin(rotation_angle)
        cth = casadi.cos(rotation_angle)
        self.R_local = casadi.SX([[ cth, -sth],
                                  [ sth,  cth]])

    def get_constraints(self):
        """ Gets the constraints. 
        """
        p_local = self.R_local.T @ self.surface.R.T @ (self.point.p - self.surface.p - self.margin)
        if self.inv_dir:
            p_local = - p_local
        # x_local = p_local[0]
        y_local = p_local[1]
        dp_local = casadi.jacobian(p_local, self.q) @ self.dq
        # dx_local = dp_local[0]
        dy_local = dp_local[1]
        cp = y_local + dy_local 
        lb = self.surface.lb
        return cp, lb, casadi.SX([casadi.inf for i in range(cp.size()[0])])

    def update_q_val(self):
        """ Updates the internal q value (configuration). 
        """
        self.surface.update_q_val()
        self.point.update_q_val()
        if self.point.q_val is not None and self.surface.q_val is not None:
            self.q_val = np.concatenate([self.point.q_val, self.surface.q_val])
        elif self.point.q_val is not None:
            self.q_val = self.point.q_val
        elif self.surface.q_val is not None:
            self.q_val = self.surface.q_val
        else:
            return NotImplementedError()