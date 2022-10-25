import casadi
import numpy as np

class RelaxedContact:
    """ A set of the relaxed complementarity constraints that expresses a contact 
        candidate between a contact point and a contact surface. """
    def __init__(self, contact_point, contact_surface, contact_name: str, 
                 mu: float=0.6, 
                 fmax=casadi.SX([casadi.inf, casadi.inf, casadi.inf]), 
                 inv_force_dir: bool=False):
        """ Construct a relaxed contact. 

            Args: 
                contact_point: A contact point candidate.
                contact_surface: A contact surface candidate.
                contact_name: Name of this contact candidate.
                mu: Friction coefficient.
                fmax: Maximum planar spatial contact force.
                inv_force_dir: Flaf to invert the direction of the contact force.
        """
        assert mu > 0
        assert fmax.shape[0] == 3
        self.contact_point = contact_point 
        self.contact_surface = contact_surface
        self.name = contact_name
        self.mu = mu
        self.fmax = fmax
        # NOTE: the contact force is expressed in the local coordinate of the contact surface
        self.fn = casadi.SX.sym(self.name + '_fn') # vertical contact force
        self.ft_plus = casadi.SX.sym(self.name + '_ft_plus') # tangential contact force (positive direction)
        self.ft_minus = casadi.SX.sym(self.name + '_ft_minus') # tangential contact force (negative direction)
        self.f = casadi.vertcat(self.fn, self.ft_plus, self.ft_minus) 
        self.dfn = casadi.SX.sym(self.name + '_dfn') # vertical contact force
        self.dft_plus = casadi.SX.sym(self.name + '_dft_plus') # tangential contact force (positive direction)
        self.dft_minus = casadi.SX.sym(self.name + '_dft_minus') # tangential contact force (negative direction)
        self.nu = casadi.SX.sym(self.name + '_nu') # absolute tangential contact velocity 
        self.slack = casadi.SX.sym(self.name + '_slack') # slack variable of complementarity constraints
        self.df = casadi.vertcat(self.dfn, self.dft_plus, self.dft_minus, self.nu, self.slack) 
        # for animation
        self.f_arrow = None
        self.inv_force_dir = inv_force_dir

        if contact_point.q is not None and contact_surface.q is not None:
            assert contact_point.dq is not None
            assert contact_surface.dq is not None
            self.q = casadi.vertcat(contact_point.q, contact_surface.q)
            self.dq = casadi.vertcat(contact_point.dq, contact_surface.dq)
            self.q_val = np.concatenate([contact_point.q_val, contact_surface.q_val])
            self.dq_val = np.concatenate([contact_point.dq_val, contact_surface.dq_val])
        elif contact_point.q is not None:
            assert contact_point.dq is not None
            self.q = contact_point.q
            self.dq = contact_point.dq
            self.q_val = contact_point.q_val
            self.dq_val = contact_point.dq_val
        elif contact_surface.q is not None:
            assert contact_surface.dq is not None
            self.q = contact_surface.q
            self.dq = contact_surface.dq
            self.q_val = contact_surface.q_val
            self.dq_val = contact_surface.dq_val
        else:
            return NotImplementedError()
        self.R = np.zeros([2, 2])

    def get_opt_vars(self):
        """ Gets the optimization variables of the LCQP. """
        return self.df

    def get_opt_var_bounds(self):
        """ Gets the bounds of the optimization variables of the LCQP. """
        fn = self.fn + self.dfn
        ft_plus = self.ft_plus + self.dft_plus
        ft_minus = self.ft_minus + self.dft_minus
        c = casadi.vertcat(fn, ft_plus, ft_minus)
        lb = - self.fmax
        ub = self.fmax
        return c, lb, ub

    def get_contact_complementarity_constraint(self):
        """ Gets the contact complementarity constraints. 
        """
        # distance and contact velocity expressed in the local coordinate of the contact surface
        p_local = self.contact_surface.R.T @ (self.contact_point.p - self.contact_surface.p)
        x_local = p_local[0]
        y_local = p_local[1] + self.contact_point.offset - self.contact_surface.offset
        if hasattr(self.contact_point, 'radius'):
            y_local = y_local - self.contact_point.radius 
        dp_local = casadi.jacobian(p_local, self.q) @ self.dq
        dx_local = dp_local[0]
        if hasattr(self.contact_point, 'radius'):
            dx_local = dx_local + self.contact_point.radius * self.dq[2]
        dy_local = dp_local[1]
        fn = self.fn + self.dfn
        ft_plus = self.ft_plus + self.dft_plus
        ft_minus = self.ft_minus + self.dft_minus
        # cp = casadi.vertcat(y_local+dy_local-self.contact_surface.lb+self.slack, self.nu, self.nu+dx_local+self.slack, self.nu-dx_local+self.slack)
        cp = casadi.vertcat(y_local+dy_local-self.contact_surface.lb+self.slack, self.nu+self.slack, self.nu+dx_local+self.slack, self.nu-dx_local+self.slack)
        # cf = casadi.vertcat(fn, self.mu*fn-ft_plus-ft_minus, ft_plus, ft_minus)
        # cf = casadi.vertcat(fn, self.mu*fn-ft_plus-ft_minus+self.slack, ft_plus, ft_minus)
        cf = casadi.vertcat(fn+self.slack, self.mu*fn-ft_plus-ft_minus+self.slack, ft_plus+self.slack, ft_minus+self.slack)
        return cp, cf

    def update_q_val(self):
        """ Updates the internal q value (configuration). 
        """
        self.contact_point.update_q_val()
        self.contact_surface.update_q_val()
        if self.contact_point.q_val is not None and self.contact_surface.q_val is not None:
            self.q_val = np.concatenate([self.contact_point.q_val, self.contact_surface.q_val])
        elif self.contact_point.q_val is not None:
            self.q_val = self.contact_point.q_val
        elif self.contact_surface.q_val is not None:
            self.q_val = self.contact_surface.q_val
        else:
            return NotImplementedError()

    def update_dq_val(self):
        """ Updates the internal dq value (generalized velocity).
        """
        self.contact_point.update_dq_val()
        self.contact_surface.update_dq_val()
        if self.contact_point.dq_val is not None and self.contact_surface.dq_val is not None:
            self.dq_val = np.concatenate([self.contact_point.dq_val, self.contact_surface.dq_val])
        elif self.contact_point.dq_val is not None:
            self.dq_val = self.contact_point.dq_val
        elif self.contact_surface.dq_val is not None:
            self.dq_val = self.contact_surface.dq_val
        else:
            return NotImplementedError()

    def get_f_world(self, f):
        """ Gets the contact forces expressed in the world frame from the spatial contact force.

            Args: 
                f: Spatial contact force. Size must be 3.
        """
        self.update_q_val()
        if self.contact_point.q is not None:
            p = np.array(casadi.Function('p', [self.contact_point.q], [self.contact_point.p])(self.contact_point.q_val)) 
        else:
            p = self.contact_point.p
        if self.contact_surface.q is not None:
            R = np.array(casadi.Function('R', [self.contact_surface.q], [self.contact_surface.R])(self.contact_surface.q_val)) 
        else:
            R = np.zeros([2,2])
            R[0,0] = self.contact_surface.R[0,0]
            R[0,1] = self.contact_surface.R[0,1]
            R[1,0] = self.contact_surface.R[1,0]
            R[1,1] = self.contact_surface.R[1,1]
        f_local = np.array([f[1]-f[2], f[0]])
        f_world = R @ f_local
        return f_world

    def get_contact_dist(self):
        """ Gets the distance between the contact point and contact surface.
        """
        self.update_q_val()
        p_local = self.contact_surface.R.T @ (self.contact_point.p - self.contact_surface.p)
        x_local = p_local[0]
        y_local = p_local[1] + self.contact_point.offset - self.contact_surface.offset
        d = casadi.Function('y_local', [self.q], [y_local])(self.q_val).__float__()
        return d

    def get_contact_vel(self):
        """ Gets the velocity between the contact point and contact surface.
        """
        self.update_q_val()
        self.update_dq_val()
        p_local = self.contact_surface.R.T @ (self.contact_point.p - self.contact_surface.p)
        dp_local = casadi.jacobian(p_local, self.q) @ self.dq
        dx_local = dp_local[0]
        if hasattr(self.contact_point, 'radius'):
            dx_local = dx_local + self.contact_point.radius * self.dq[2]
        v = casadi.Function('dx_local', [self.q, self.dq], [dx_local])(self.q_val, self.dq_val).__float__()
        return v

    def init_anim(self, ax, color='green', linewidth: float=2.5):
        """ Initializes the animation. 

            Args: 
                ax: Axis object of the matplotlib.
                color: Color of contact forces.
                linewidth: Line width.
        """
        # self.f_arrow = ax.arrow([], [], [], [], color='green')
        self.f_arrow, = ax.plot([], [], color=color, linewidth=linewidth, alpha=0.5)

    def update_anim(self, f, scaling: float=1):
        """ Updates the animation. 

            Args: 
                f: Spatial contact force.
                f_scaling: Scaling factor of the contact forces.
        """
        f_world = self.get_f_world(f)
        if self.inv_force_dir:
            f_world = -f_world
        if self.contact_point.q is not None:
            p = np.array(casadi.Function('p', [self.contact_point.q], [self.contact_point.p])(self.contact_point.q_val)) 
        else:
            p = self.contact_point.p
        self.f_arrow.set_data((p[0], p[0]+0.1*scaling*f_world[0]), (p[1], p[1]+0.1*scaling*f_world[1]))