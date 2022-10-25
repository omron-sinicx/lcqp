import casadi
import numpy as np

class ForceDirectionConstraint:
    """ Force direction constraint. """
    def __init__(self, contact, margin: float=0, inv_dir: bool=False):
        """ Construct a force direction constraint. 

            Args: 
                contact: Contact of whose contact force is constrained.
                margin: Margin of the constraint.
                inv_dir: Flaf if the directin if inverted or not.
        """
        self.contact = contact
        self.contact_point = contact.contact_point
        self.contact_surface = contact.contact_surface
        self.margin = margin
        self.inv_dir = inv_dir
        if self.contact_point.q is not None and self.contact_surface.q is not None:
            self.q = casadi.vertcat(self.contact_point.q, self.contact_surface.q)
            self.dq = casadi.vertcat(self.contact_point.dq, self.contact_surface.dq)
            self.q_val = np.concatenate([self.contact_point.q_val, self.contact_surface.q_val])
        elif self.contact_point.q is not None:
            self.q = self.contact_point.q
            self.dq = self.contact_point.dq
            self.q_val = self.contact_point.q_val
        elif self.contact_surface.q is not None:
            self.q = self.contact_surface.q
            self.dq = self.contact_surface.dq
            self.q_val = self.contact_surface.q_val
        else:
            return NotImplementedError()

    def get_constraints(self):
        """ Gets the constraints. 
        """
        if self.contact_surface.q is not None:
            R = np.array(casadi.Function('R', [self.contact_surface.q], [self.contact_surface.R])(self.contact_surface.q_val)) 
        else:
            R = np.zeros([2,2])
            R[0,0] = self.contact_surface.R[0,0]
            R[0,1] = self.contact_surface.R[0,1]
            R[1,0] = self.contact_surface.R[1,0]
            R[1,1] = self.contact_surface.R[1,1]
        f = self.contact.f
        f_local = casadi.vertcat(f[1]-f[2], f[0])
        f_world = R @ f_local
        if self.contact_surface.q is not None:
            R = np.array(casadi.Function('R', [self.contact_point.q], [self.contact_point.R])(self.contact_point.q_val)) 
        else:
            R = np.zeros([2,2])
            R[0,0] = self.contact_point.R[0,0]
            R[0,1] = self.contact_point.R[0,1]
            R[1,0] = self.contact_point.R[1,0]
            R[1,1] = self.contact_point.R[1,1]
        f_local_cylinder = R.T @ f_world
        return f_local_cylinder[1], 0, casadi.inf

    def update_q_val(self):
        """ Updates the internal q value (configuration). 
        """
        self.contact_surface.update_q_val()
        self.contact_point.update_q_val()
        if self.contact_point.q_val is not None and self.contact_surface.q_val is not None:
            self.q_val = np.concatenate([self.contact_point.q_val, self.contact_surface.q_val])
        elif self.contact_point.q_val is not None:
            self.q_val = self.contact_point.q_val
        elif self.contact_surface.q_val is not None:
            self.q_val = self.contact_surface.q_val
        else:
            return NotImplementedError()