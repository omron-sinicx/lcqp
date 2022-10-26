#
# Copyright (c) 2022 OMRON SINIC X Corporation
#

import casadi
import numpy as np

class Gripper:
    """ 2D model of a gripper. """
    def __init__(self, h: float, rmax: float=0.115, rmin: float=0.01,
                 vmax=casadi.SX([casadi.inf, casadi.inf, casadi.inf, casadi.inf]), 
                 qref=casadi.SX.zeros(4)):
        """ Construct a gripper. 

            Args: 
                h: Height of this gripper.
                rmax: Maximum distance between two fingers.
                rmin: Minimum distance between two fingers.
                vmax: Maximum generalized velocity of this gripper. Size must be 4.
                qref: Reference configuration of this gripper. Size must be 4.
        """
        assert h > 0.
        assert rmax > 0.
        assert vmax.shape[0] == 4
        assert qref.shape[0] == 4
        self.h = h
        self.rmax = rmax
        self.rmin = rmin
        self.vmax = vmax
        self.qref = qref
        self.dimq = 4
        # configuration
        self.x  = casadi.SX.sym('x_gripper')
        self.y  = casadi.SX.sym('y_gripper')
        self.th = casadi.SX.sym('th_gripper')
        self.r  = casadi.SX.sym('r_gripper')
        self.q  = casadi.vertcat(self.x, self.y, self.th, self.r)
        # velocity
        self.dx  = casadi.SX.sym('dx_gripper')
        self.dy  = casadi.SX.sym('dy_gripper')
        self.dth = casadi.SX.sym('dth_gripper')
        self.dr  = casadi.SX.sym('dr_gripper')
        self.dq  = casadi.vertcat(self.dx, self.dy, self.dth, self.dr)
        # rotation matirx and its derivative
        sin_th = casadi.sin(self.th)
        cos_th = casadi.cos(self.th)
        self.R = casadi.SX(2, 2)
        self.R[0,0] = cos_th
        self.R[0,1] = -sin_th
        self.R[1,0] = sin_th
        self.R[1,1] = cos_th
        # corner positions in the local coordiante 
        self.TR_corner_local = casadi.vertcat( 0.5*self.rmax,  0) # top-right corner
        self.BR_corner_local = casadi.vertcat( 0.5*self.r, -h) # bottom-right corner
        self.TL_corner_local = casadi.vertcat(-0.5*self.rmax,  0) # top-left corner
        self.BL_corner_local = casadi.vertcat(-0.5*self.r, -h) # bottom-left corner
        self.R_joint_local = casadi.vertcat( 0.5*self.r,  0) # top-right joint 
        self.L_joint_local = casadi.vertcat(-0.5*self.r,  0) # top-left joint
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
        self.L_joint = self.center + self.R @ self.L_joint_local
        self.R_joint = self.center + self.R @ self.R_joint_local
        # corner Jacobians in the world coordiante 
        self.J_center    = casadi.jacobian(self.center, self.q)
        self.J_TR_corner = casadi.jacobian(self.TR_corner, self.q)
        self.J_BR_corner = casadi.jacobian(self.BR_corner, self.q)
        self.J_TL_corner = casadi.jacobian(self.TL_corner, self.q)
        self.J_BL_corner = casadi.jacobian(self.BL_corner, self.q)
        # for animation
        self.q_val = np.zeros(4)
        self.dq_val = np.zeros(4)
        self.T_line = None
        self.T_line_U = None
        self.T_line_UL = None
        self.T_line_UR = None
        self.R_line = None
        self.L_line = None

    def get_opt_vars(self):
        """ Gets the optimization variables of the LCQP. """
        return self.dq

    def get_opt_var_bounds(self):
        """ Gets the bounds of the optimization variables of the LCQP. """
        c  = casadi.vertcat(self.r, self.dq)
        lb = casadi.vertcat(self.rmin, -self.vmax)
        ub = casadi.vertcat(self.rmax, self.vmax)
        return c, lb, ub

    def update_q_val(self, q: np.ndarray):
        """ Updates the internal q value (configuration). 

            Args: 
                q: Configuration. Size must be 4.
        """
        assert q.shape[0] == 4
        self.q_val = q.copy()

    def update_dq_val(self, dq: np.ndarray):
        """ Updates the internal dq value (generalized velocity). 

            Args: 
                dq: Generalized velocity. Size must be 3.
        """
        assert dq.shape[0] == 4
        self.dq_val = dq.copy()

    def init_anim(self, ax, color='blue', linewidth: float=1.5):
        """ Initializes the animation. 

            Args: 
                ax: Axis object of the matplotlib.
                color: Color of gripper.
                linewidth: Line width.
        """
        self.T_line, = ax.plot([], [], color=color, linewidth=linewidth)
        self.R_line, = ax.plot([], [], color=color, linewidth=linewidth)
        self.L_line, = ax.plot([], [], color=color, linewidth=linewidth)

    def update_anim(self, q: np.ndarray):
        """ Updates the animation. 

            Args: 
                q: Configuration of this gripper.
        """
        self.update_q_val(q)
        TR_corner = np.array(casadi.Function('TR_corner', [self.q], [self.TR_corner])(q)) 
        BR_corner = np.array(casadi.Function('BR_corner', [self.q], [self.BR_corner])(q)) 
        TL_corner = np.array(casadi.Function('TL_corner', [self.q], [self.TL_corner])(q)) 
        BL_corner = np.array(casadi.Function('BL_corner', [self.q], [self.BL_corner])(q)) 
        R_joint = np.array(casadi.Function('R_joint', [self.q], [self.R_joint])(q)) 
        L_joint = np.array(casadi.Function('L_joint', [self.q], [self.L_joint])(q))
        self.T_line.set_data((TL_corner[0], TR_corner[0]), (TL_corner[1], TR_corner[1]))
        self.R_line.set_data((R_joint[0], BR_corner[0]), (R_joint[1], BR_corner[1]))
        self.L_line.set_data((BL_corner[0], L_joint[0]), (BL_corner[1], L_joint[1]))


class GripperTopLeftCorner:
    """ Top left corner of a gripper. """
    def __init__(self, gripper: Gripper, offset: float=0):
        """ Construct the corner. 

            Args: 
                gripper: Gripper of interest.
                offset: Offset of this corner.
        """
        self.p = gripper.TL_corner
        self.q = gripper.q
        self.dq = gripper.dq
        self.gripper = gripper
        self.q_val = gripper.q_val
        self.dq_val = gripper.dq_val
        self.offset = offset

    def update_q_val(self):
        """ Updates the internal q value (configuration) through internal gripper object. 
        """
        self.q_val = self.gripper.q_val

    def update_anim(self):
        """ Updates the animation. 
        """
        self.update_q_val()

    def update_dq_val(self):
        """ Updates the internal dq value (generalized velocity) through internal gripper object. 
        """
        self.dq_val = self.gripper.dq_val

class GripperBottomLeftCorner:
    """ Bottom left corner of a gripper. """
    def __init__(self, gripper: Gripper, offset: float=0):
        """ Construct the corner. 

            Args: 
                gripper: Gripper of interest.
                offset: Offset of this corner.
        """
        self.p = gripper.BL_corner
        self.q = gripper.q
        self.dq = gripper.dq
        self.gripper = gripper
        self.q_val = gripper.q_val
        self.dq_val = gripper.dq_val
        self.offset = offset

    def update_q_val(self):
        """ Updates the internal q value (configuration) through internal gripper object. 
        """
        self.q_val = self.gripper.q_val

    def update_anim(self):
        """ Updates the animation. 
        """
        self.update_q_val()

    def update_dq_val(self):
        """ Updates the internal dq value (generalized velocity) through internal gripper object. 
        """
        self.dq_val = self.gripper.dq_val

class GripperTopRightCorner:
    """ Top right corner of a gripper. """
    def __init__(self, gripper: Gripper, offset: float=0):
        """ Construct the corner. 

            Args: 
                gripper: Gripper of interest.
                offset: Offset of this corner.
        """
        self.p = gripper.TR_corner
        self.q = gripper.q
        self.dq = gripper.dq
        self.gripper = gripper
        self.q_val = gripper.q_val
        self.dq_val = gripper.dq_val
        self.offset = offset

    def update_q_val(self):
        """ Updates the internal q value (configuration) through internal gripper object. 
        """
        self.q_val = self.gripper.q_val

    def update_anim(self):
        """ Updates the animation. 
        """
        self.update_q_val()

    def update_dq_val(self):
        """ Updates the internal dq value (generalized velocity) through internal gripper object. 
        """
        self.dq_val = self.gripper.dq_val


class GripperBottomRightCorner:
    """ Bottom right corner of a gripper. """
    def __init__(self, gripper: Gripper, offset: float=0):
        """ Construct the corner. 

            Args: 
                gripper: Gripper of interest.
                offset: Offset of this corner.
        """
        self.p = gripper.BR_corner
        self.q = gripper.q
        self.dq = gripper.dq
        self.gripper = gripper
        self.q_val = gripper.q_val
        self.dq_val = gripper.dq_val
        self.offset = offset

    def update_q_val(self):
        """ Updates the internal q value (configuration) through internal gripper object. 
        """
        self.q_val = self.gripper.q_val

    def update_anim(self):
        """ Updates the animation. 
        """
        self.update_q_val()

    def update_dq_val(self):
        """ Updates the internal dq value (generalized velocity) through internal gripper object. 
        """
        self.dq_val = self.gripper.dq_val


class GripperTopSurface:
    """ Top surface of a gripper. """
    def __init__(self, gripper: Gripper, offset: float=0):
        """ Construct the corner. 

            Args: 
                gripper: Gripper of interest.
                offset: Offset of this surface.
        """
        # rotation of the frame
        self.R_local = casadi.SX([[-1.0, 0.0],
                                  [ 0.0, -1.0]])
        self.R = gripper.R @ self.R_local 
        # translation of the origin of the frame expressed in the world frame
        self.p_local = casadi.SX.zeros(2)
        self.p = gripper.center + gripper.R @ self.p_local 
        self.lb = 0.
        self.q = gripper.q
        self.dq = gripper.dq
        self.gripper = gripper
        self.q_val = gripper.q_val
        self.dq_val = gripper.dq_val
        self.offset = offset

    def update_q_val(self):
        """ Updates the internal q value (configuration) through internal gripper object. 
        """
        self.q_val = self.gripper.q_val

    def update_anim(self):
        """ Updates the animation. 
        """
        self.update_q_val()

    def update_dq_val(self):
        """ Updates the internal dq value (generalized velocity) through internal gripper object. 
        """
        self.dq_val = self.gripper.dq_val

class GripperLeftSurface:
    """ Left surface of a gripper. """
    def __init__(self, gripper: Gripper, offset: float=0):
        """ Construct the corner. 

            Args: 
                gripper: Gripper of interest.
                offset: Offset of this surface.
        """
        # rotation of the frame
        self.R_local = casadi.SX([[ 0.0,  1.0],
                                  [-1.0,  0.0]])
        self.R = gripper.R @ self.R_local 
        # translation of the origin of the frame expressed in the world frame
        self.p_local = casadi.vertcat(-0.5*gripper.r, casadi.SX.zeros(1))
        self.p = gripper.center + gripper.R @ self.p_local 
        self.lb = 0.
        self.q = gripper.q
        self.dq = gripper.dq
        self.gripper = gripper
        self.q_val = gripper.q_val
        self.dq_val = gripper.dq_val
        self.offset = offset

    def update_q_val(self):
        """ Updates the internal q value (configuration) through internal gripper object. 
        """
        self.q_val = self.gripper.q_val

    def update_dq_val(self):
        """ Updates the internal dq value (generalized velocity) through internal gripper object. 
        """
        self.dq_val = self.gripper.dq_val


class GripperRightSurface:
    """ Right surface of a gripper. """
    def __init__(self, gripper: Gripper, offset: float=0):
        """ Construct the corner. 

            Args: 
                gripper: Gripper of interest.
                offset: Offset of this surface.
        """
        # rotation of the frame
        self.R_local = casadi.SX([[ 0.0, -1.0],
                                  [ 1.0,  0.0]])
        # self.R_local = casadi.SX([[ 0.0,  1.0],
        #                           [-1.0,  0.0]])
        self.R = gripper.R @ self.R_local 
        # translation of the origin of the frame expressed in the world frame
        self.p_local = casadi.vertcat(0.5*gripper.r, casadi.SX.zeros(1))
        self.p = gripper.center + gripper.R @ self.p_local 
        self.lb = 0.
        self.q = gripper.q
        self.dq = gripper.dq
        self.gripper = gripper
        self.q_val = gripper.q_val
        self.dq_val = gripper.dq_val
        self.offset = offset

    def update_q_val(self):
        """ Updates the internal q value (configuration) through internal gripper object. 
        """
        self.q_val = self.gripper.q_val

    def update_dq_val(self):
        """ Updates the internal dq value (generalized velocity) through internal gripper object. 
        """
        self.dq_val = self.gripper.dq_val
