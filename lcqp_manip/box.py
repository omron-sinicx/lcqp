import casadi
import numpy as np

class Box:
    """ 2D model of a box. """
    def __init__(self, h: float, w: float, m: float, g: float=9.81, 
                 vmax=casadi.SX([casadi.inf, casadi.inf, casadi.inf]), 
                 qref=casadi.SX.zeros(3)):
        """ Construct a box. 

            Args: 
                h: Height of this box.
                w: Width of this box.
                m: Weight of this box.
                g: Gravity acceleration.
                vmax: Maximum planar spatial velocity of this box. Size must be 3.
                qref: Reference configuration of this box. Size must be 3.
        """
        assert h > 0.
        assert w > 0.
        assert m > 0.
        assert vmax.shape[0] == 3
        assert qref.shape[0] == 3
        self.h = h
        self.w = w
        self.m = m
        self.g = g
        self.vmax = vmax
        self.qref = qref
        self.dimq = 3
        # configuration
        self.x  = casadi.SX.sym('x_box')
        self.y  = casadi.SX.sym('y_box')
        self.th = casadi.SX.sym('th_box')
        self.q  = casadi.vertcat(self.x, self.y, self.th)
        # velocity
        self.dx  = casadi.SX.sym('dx_box')
        self.dy  = casadi.SX.sym('dy_box')
        self.dth = casadi.SX.sym('dth_box')
        self.dq  = casadi.vertcat(self.dx, self.dy, self.dth)
        # rotation matirx and its derivative
        sin_th = casadi.sin(self.th)
        cos_th = casadi.cos(self.th)
        self.R = casadi.SX(2, 2)
        self.R[0,0] = cos_th
        self.R[0,1] = -sin_th
        self.R[1,0] = sin_th
        self.R[1,1] = cos_th
        # corner positions in the local coordiante 
        self.TR_corner_local = casadi.vertcat( 0.5*w,  0.5*h) # top-right corner
        self.BR_corner_local = casadi.vertcat( 0.5*w, -0.5*h) # bottom-right corner
        self.TL_corner_local = casadi.vertcat(-0.5*w,  0.5*h) # top-left corner
        self.BL_corner_local = casadi.vertcat(-0.5*w, -0.5*h) # bottom-left corner
        # corner Jacobians in the local coordiante 
        self.J_TR_corner_local = casadi.jacobian(self.TR_corner_local, self.q)
        self.J_BR_corner_local = casadi.jacobian(self.BR_corner_local, self.q)
        self.J_TL_corner_local = casadi.jacobian(self.TL_corner_local, self.q)
        self.J_BL_corner_local = casadi.jacobian(self.BL_corner_local, self.q)
        # corner positions in the world coordiante 
        self.center    = casadi.vertcat(self.x, self.y)
        self.TR_corner = self.center + self.R @ self.TR_corner_local
        self.BR_corner = self.center + self.R @ self.BR_corner_local
        self.TL_corner = self.center + self.R @ self.TL_corner_local
        self.BL_corner = self.center + self.R @ self.BL_corner_local
        # corner Jacobians in the world coordiante 
        self.J_center    = casadi.jacobian(self.center, self.q)
        self.J_TR_corner = casadi.jacobian(self.TR_corner, self.q)
        self.J_BR_corner = casadi.jacobian(self.BR_corner, self.q)
        self.J_TL_corner = casadi.jacobian(self.TL_corner, self.q)
        self.J_BL_corner = casadi.jacobian(self.BL_corner, self.q)
        # for animation
        self.q_val = np.zeros(3)
        self.dq_val = np.zeros(3)
        self.T_line = None
        self.B_line = None
        self.R_line = None
        self.L_line = None
        self.f_grav_arrow = None

    def get_opt_vars(self):
        """ Gets the optimization variables of the LCQP. """
        return self.dq

    def get_opt_var_bounds(self):
        """ Gets the bounds of the optimization variables of the LCQP. """
        c = self.dq
        lb = - self.vmax
        ub = self.vmax
        return c, lb, ub

    def get_force_moment_balance_constraint(self, contacts: list):
        """ Gets a force balance equality constraint. 

            Args: 
                contacts: List of contacts around this box.
        """
        f = casadi.vertcat(0., -self.m*self.g)
        Mz = 0.
        for e in contacts:
            fn = e.fn + e.dfn
            ft_plus = e.ft_plus + e.dft_plus
            ft_minus = e.ft_minus + e.dft_minus
            fxy_local = casadi.vertcat(ft_plus-ft_minus, fn)
            fxy_world = e.contact_surface.R @ fxy_local
            if e.inv_force_dir:
                f = f - fxy_world
            else:
                f = f + fxy_world
            com_to_point_world = e.contact_point.p - self.center
            Mz_e = com_to_point_world[0] * fxy_world[1] + com_to_point_world[1] * fxy_world[0]
            if e.inv_force_dir:
                Mz = Mz - Mz_e
            else:
                Mz = Mz + Mz_e
        return casadi.vertcat(f, Mz) 

    def update_q_val(self, q: np.ndarray):
        """ Updates the internal q value (configuration). 

            Args: 
                q: Configuration. Size must be 3.
        """
        assert q.shape[0] == 3
        self.q_val = q.copy()

    def update_dq_val(self, dq: np.ndarray):
        """ Updates the internal dq value (generalized velocity). 

            Args: 
                dq: Generalized velocity. Size must be 3.
        """
        assert dq.shape[0] == 3
        self.dq_val = dq.copy()

    def init_anim(self, ax, color='orange', linewidth: float=1.5):
        """ Initializes the animation. 

            Args: 
                ax: Axis object of the matplotlib.
                color: Color of box.
                linewidth: Line width.
        """
        self.T_line, = ax.plot([], [], color=color, linewidth=linewidth)
        self.B_line, = ax.plot([], [], color=color, linewidth=linewidth)
        self.R_line, = ax.plot([], [], color=color, linewidth=linewidth)
        self.L_line, = ax.plot([], [], color=color, linewidth=linewidth)
        self.f_grav_arrow, = ax.plot([], [], color='red', linewidth=2.5, alpha=0.5)

    def update_anim(self, q: np.ndarray, f_scaling: float=1):
        """ Updates the animation. 

            Args: 
                q: Configuration of this box.
                f_scaling: A scaling factor of forces.
        """
        self.update_q_val(q)
        TR_corner = np.array(casadi.Function('TR_corner', [self.q], [self.TR_corner])(q)) 
        BR_corner = np.array(casadi.Function('BR_corner', [self.q], [self.BR_corner])(q)) 
        TL_corner = np.array(casadi.Function('TL_corner', [self.q], [self.TL_corner])(q)) 
        BL_corner = np.array(casadi.Function('BL_corner', [self.q], [self.BL_corner])(q)) 
        self.T_line.set_data([TL_corner[0], TR_corner[0]], [TL_corner[1], TR_corner[1]])
        self.B_line.set_data([BR_corner[0], BL_corner[0]], [BR_corner[1], BL_corner[1]])
        self.R_line.set_data([TR_corner[0], BR_corner[0]], [TR_corner[1], BR_corner[1]])
        self.L_line.set_data([BL_corner[0], TL_corner[0]], [BL_corner[1], TL_corner[1]])
        center = np.array(casadi.Function('center', [self.q], [self.center])(q)) 
        self.f_grav_arrow.set_data((center[0], center[0]), (center[1], center[1]-f_scaling*0.1*self.m*self.g))


class BoxTopLeftCorner:
    """ Top left corner of a box. """
    def __init__(self, box: Box):
        """ Construct the corner. 

            Args: 
                box: Box of interest.
        """
        self.p = box.TL_corner
        self.q = box.q
        self.dq = box.dq
        self.box = box
        self.q_val = self.box.q_val.copy()
        self.dq_val = self.box.dq_val.copy()
        self.offset = 0

    def update_q_val(self):
        """ Updates the internal q value (configuration) through internal box object. 
        """
        self.q_val = self.box.q_val.copy()

    def update_anim(self):
        """ Updates the animation. 
        """
        self.update_q_val()

    def update_dq_val(self):
        """ Updates the internal dq value (generalized velocity) through internal box object. 
        """
        self.dq_val = self.box.dq_val


class BoxBottomLeftCorner:
    """ Bottom left corner of a box. """
    def __init__(self, box: Box):
        """ Construct the corner. 

            Args: 
                box: Box of interest.
        """
        self.p = box.BL_corner
        self.q = box.q
        self.dq = box.dq
        self.box = box
        self.q_val = self.box.q_val.copy()
        self.dq_val = self.box.dq_val.copy()
        self.offset = 0

    def update_q_val(self):
        """ Updates the internal q value (configuration) through internal box object. 
        """
        self.q_val = self.box.q_val.copy()

    def update_anim(self):
        """ Updates the animation. 
        """
        self.update_q_val()

    def update_dq_val(self):
        """ Updates the internal dq value (generalized velocity) through internal box object. 
        """
        self.dq_val = self.box.dq_val


class BoxTopRightCorner:
    """ Top right corner of a box. """
    def __init__(self, box: Box):
        """ Construct the corner. 

            Args: 
                box: Box of interest.
        """
        self.p = box.TR_corner
        self.q = box.q
        self.dq = box.dq
        self.box = box
        self.q_val = self.box.q_val.copy()
        self.dq_val = self.box.dq_val.copy()
        self.offset = 0

    def update_q_val(self):
        """ Updates the internal q value (configuration) through internal box object. 
        """
        self.q_val = self.box.q_val.copy()

    def update_anim(self):
        """ Updates the animation. 
        """
        self.update_q_val()

    def update_dq_val(self):
        """ Updates the internal dq value (generalized velocity) through internal box object. 
        """
        self.dq_val = self.box.dq_val


class BoxBottomRightCorner:
    """ Bottom right corner of a box. """
    def __init__(self, box: Box):
        """ Construct the corner. 

            Args: 
                box: Box of interest.
        """
        self.p = box.BR_corner
        self.q = box.q
        self.dq = box.dq
        self.box = box
        self.q_val = self.box.q_val.copy()
        self.dq_val = self.box.dq_val.copy()
        self.offset = 0

    def update_q_val(self):
        """ Updates the internal q value (configuration) through internal box object. 
        """
        self.q_val = self.box.q_val.copy()

    def update_anim(self):
        """ Updates the animation. 
        """
        self.update_q_val()

    def update_dq_val(self):
        """ Updates the internal dq value (generalized velocity) through internal box object. 
        """
        self.dq_val = self.box.dq_val


class BoxTopSurface:
    """ Top surface of a box. """
    def __init__(self, box: Box):
        """ Construct the surface. 

            Args: 
                box: Box of interest.
        """
        # rotation of the frame
        self.R_local = casadi.SX([[ 1.0, 0.0],
                                  [ 0.0, 1.0]])
        self.R = box.R @ self.R_local  
        # translation of the origin of the frame expressed in the world frame
        self.p_local = casadi.vertcat(0, 0.5*box.h)
        self.p = box.center + box.R @ self.p_local 
        self.lb = 0.
        self.q = box.q
        self.dq = box.dq
        self.box = box
        self.q_val = box.q_val
        self.dq_val = box.dq_val
        self.offset = 0

    def update_q_val(self):
        """ Updates the internal q value (configuration) through internal box object. 
        """
        self.q_val = self.box.q_val

    def update_anim(self):
        """ Updates the animation. 
        """
        self.update_q_val()

    def update_dq_val(self):
        """ Updates the internal dq value (generalized velocity) through internal box object. 
        """
        self.dq_val = self.box.dq_val


class BoxBottomSurface:
    """ Bottom surface of a box. """
    def __init__(self, box: Box):
        """ Construct the surface. 

            Args: 
                box: Box of interest.
        """
        # rotation of the frame
        self.R_local = casadi.SX([[-1.0,  0.0],
                                  [ 0.0, -1.0]])
        self.R = box.R @ self.R_local  
        # translation of the origin of the frame expressed in the world frame
        self.p_local = casadi.vertcat(0, -0.5*box.h)
        self.p = box.center + box.R @ self.p_local 
        self.lb = 0.
        self.q = box.q
        self.dq = box.dq
        self.box = box
        self.q_val = box.q_val
        self.dq_val = box.dq_val
        self.offset = 0

    def update_q_val(self):
        """ Updates the internal q value (configuration) through internal box object. 
        """
        self.q_val = self.box.q_val

    def update_anim(self):
        """ Updates the animation. 
        """
        self.update_q_val()

    def update_dq_val(self):
        """ Updates the internal dq value (generalized velocity) through internal box object. 
        """
        self.dq_val = self.box.dq_val


class BoxLeftSurface:
    """ Left surface of a box. """
    def __init__(self, box: Box):
        """ Construct the surface. 

            Args: 
                box: Box of interest.
        """
        # rotation of the frame
        self.R_local = casadi.SX([[ 0.0, -1.0],
                                  [ 1.0,  0.0]])
        self.R = box.R @ self.R_local  
        # translation of the origin of the frame expressed in the world frame
        self.p_local = casadi.vertcat(-0.5*box.w, 0)
        self.p = box.center + box.R @ self.p_local 
        self.lb = 0.
        self.q = box.q
        self.dq = box.dq
        self.box = box
        self.q_val = box.q_val
        self.dq_val = box.dq_val
        self.offset = 0

    def update_q_val(self):
        """ Updates the internal q value (configuration) through internal box object. 
        """
        self.q_val = self.box.q_val

    def update_anim(self):
        """ Updates the animation. 
        """
        self.update_q_val()

    def update_dq_val(self):
        """ Updates the internal dq value (generalized velocity) through internal box object. 
        """
        self.dq_val = self.box.dq_val


class BoxRightSurface:
    """ Right surface of a box. """
    def __init__(self, box: Box):
        """ Construct the surface. 

            Args: 
                box: Box of interest.
        """
        # rotations of the surface expressed in the box's local and world coordinates
        self.R_local = casadi.SX([[ 0.0,  1.0],
                                  [-1.0,  0.0]])
        self.R = box.R @ self.R_local  
        # translations of the origin of the surface expressed in the box's local and world coordinates
        self.p_local = casadi.vertcat(0.5*box.w, 0)
        self.p = box.center + box.R @ self.p_local 

        self.lb = 0.
        self.q = box.q
        self.dq = box.dq
        self.box = box
        self.q_val = box.q_val
        self.dq_val = box.dq_val
        self.offset = 0

    def update_q_val(self):
        """ Updates the internal q value (configuration) through internal box object. 
        """
        self.q_val = self.box.q_val

    def update_anim(self):
        """ Updates the animation. 
        """
        self.update_q_val()

    def update_dq_val(self):
        """ Updates the internal dq value (generalized velocity) through internal box object. 
        """
        self.dq_val = self.box.dq_val
