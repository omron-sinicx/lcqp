#
# Copyright (c) 2022 OMRON SINIC X Corporation
#

import casadi

class Wall:
    """ A wall. """
    def __init__(self, w0: float=0.):
        """ Construct a wall. 

            Args: 
                w0: Position of the wall.
        """
        self.R = casadi.SX([[ 0.0, 1.0],
                            [-1.0, 0.0]])
        self.p = casadi.SX([w0, 0.]) # translation of the origin of the frame expressed in the world frame
        self.w0 = w0
        self.lb = 0.
        self.q = None
        self.dq = None
        self.q_val = None
        self.dq_val = None
        self.dimq = 0
        self.offset = 0
        self.line = None
        self.length = 1000.0

    def init_anim(self, ax, color='black', linewidth: float=0.5):
        """ Initializes the animation. 

            Args: 
                ax: Axis object of the matplotlib.
                color: Color of the ground.
                linewidth: Line width.
        """
        self.line, = ax.plot([], [], color=color, linewidth=linewidth)

    def update_q_val(self):
        """ Updates the internal q value (configuration).
        """
        pass

    def update_dq_val(self):
        """ Updates the internal dq value (generalized velocity).
        """
        pass

    def update_anim(self, q=None):
        """ Updates the animation. 
        """
        self.line.set_data((self.w0, self.w0), (-self.length, self.length))