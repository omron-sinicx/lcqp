#
# Copyright (c) 2022 OMRON SINIC X Corporation
#

import numpy as np 
from scipy.spatial.transform import Rotation


class Projection:
    def __init__(self, z_offset, axis_pos, axis_rot, rot_proj=0, rot_mat_3D_base=np.eye(3, 3)):
        self.z_offset = z_offset
        self.axis_pos = axis_pos
        self.axis_rot = axis_rot
        self.rot_mat_3D_base = rot_mat_3D_base
        self.pos_3D = np.zeros(3)
        self.rot_proj = np.array([[np.cos(rot_proj), -np.sin(rot_proj)], 
                                  [np.sin(rot_proj),  np.cos(rot_proj)]])
        self.rot_mat_3D = np.zeros([3, 3])

    def projection_3D_to_2D(self, pos_3D, orn_3D):
        self.pos_3D = np.array(pos_3D)
        if self.axis_pos == 'x':
            pos_2D = np.array([pos_3D[1], pos_3D[2]-self.z_offset]) 
        elif self.axis_pos == 'y':
            pos_2D = np.array([pos_3D[0], pos_3D[2]-self.z_offset]) 
        elif self.axis_pos == 'z':
            pos_2D = np.array([pos_3D[0], pos_3D[1]]) 
        else:
            return NotImplementedError()
        rot_mat_3D = Rotation.from_quat(np.array(orn_3D)).as_matrix()
        self.rot_mat_3D = self.rot_mat_3D_base.T @ rot_mat_3D
        if self.axis_rot == 'x':
            rot_mat_2D = np.array([[self.rot_mat_3D[1, 1], self.rot_mat_3D[1, 2]],
                                   [self.rot_mat_3D[2, 1], self.rot_mat_3D[2, 2]]])
        elif self.axis_rot == 'y':
            rot_mat_2D = np.array([[self.rot_mat_3D[0, 0], self.rot_mat_3D[0, 2]],
                                   [self.rot_mat_3D[2, 0], self.rot_mat_3D[2, 2]]])
        elif self.axis_rot == 'z':
            rot_mat_2D = np.array([[self.rot_mat_3D[0, 0], self.rot_mat_3D[0, 1]],
                                   [self.rot_mat_3D[1, 0], self.rot_mat_3D[1, 1]]])
        else:
            return NotImplementedError()
        rot_mat_2D = self.rot_proj.T @ rot_mat_2D
        rot_2D = np.arctan2(rot_mat_2D[1, 0], rot_mat_2D[0, 0])
        return pos_2D, rot_2D

    def projection_2D_to_3D(self, pos_2D, orn_2D, reset_nonprojected_direction=False):
        if self.axis_pos == 'x':
            pos_3D = np.array([self.pos_3D[0], pos_2D[0], pos_2D[1]+self.z_offset])
        elif self.axis_pos == 'y':
            pos_3D = np.array([pos_2D[0], self.pos_3D[1], pos_2D[1]+self.z_offset])
        elif self.axis_pos == 'z':
            pos_3D = np.array([pos_2D[0], pos_2D[1], self.pos_3D[2]])
        else:
            return NotImplementedError()
        rot_mat_2D = np.array([[np.cos(orn_2D), -np.sin(orn_2D)],
                               [np.sin(orn_2D),  np.cos(orn_2D)]])
        rot_mat_2D = rot_mat_2D @ self.rot_proj 
        if self.axis_rot == 'x':
            if reset_nonprojected_direction:
                rot_mat_3D = np.array([[                     1,                    0,                      0],
                                       [                     0,     rot_mat_2D[0, 0],       rot_mat_2D[0, 1]],
                                       [                     0,     rot_mat_2D[1, 0],       rot_mat_2D[1, 1]]])
            else: 
                rot_mat_3D = np.array([[ self.rot_mat_3D[0, 0], self.rot_mat_3D[0, 1], self.rot_mat_3D[0, 2]],
                                       [ self.rot_mat_3D[1, 0],      rot_mat_2D[0, 0],      rot_mat_2D[0, 1]],
                                       [ self.rot_mat_3D[2, 0],      rot_mat_2D[1, 0],      rot_mat_2D[1, 1]]])
        elif self.axis_rot == 'y':
            if reset_nonprojected_direction:
                rot_mat_3D = np.array([[      rot_mat_2D[0, 0],                     0,      rot_mat_2D[0, 1]],
                                       [                     0,                     1,                     0],
                                       [      rot_mat_2D[1, 0],                     0,      rot_mat_2D[1, 1]]])
            else:
                rot_mat_3D = np.array([[      rot_mat_2D[0, 0], self.rot_mat_3D[0, 1],      rot_mat_2D[0, 1]],
                                       [ self.rot_mat_3D[0, 1], self.rot_mat_3D[1, 1], self.rot_mat_3D[1, 2]],
                                       [      rot_mat_2D[1, 0], self.rot_mat_3D[2, 1],      rot_mat_2D[1, 1]]])
        elif self.axis_rot == 'z':
            if reset_nonprojected_direction:
                rot_mat_3D = np.array([[      rot_mat_2D[0, 0],      rot_mat_2D[0, 1],                     0],
                                       [      rot_mat_2D[1, 0],      rot_mat_2D[1, 1],                     0],
                                       [                     0,                     0,                     1]])
            else:
                rot_mat_3D = np.array([[      rot_mat_2D[0, 0],      rot_mat_2D[0, 1], self.rot_mat_3D[0, 2]],
                                       [      rot_mat_2D[1, 0],      rot_mat_2D[1, 1], self.rot_mat_3D[1, 2]],
                                       [ self.rot_mat_3D[2, 0], self.rot_mat_3D[2, 1], self.rot_mat_3D[2, 2]]])
        else:
            return NotImplementedError()
        orn_3D_mat = self.rot_mat_3D_base.T @ rot_mat_3D
        orn_3D_quat = Rotation.from_matrix(orn_3D_mat).as_quat()
        return pos_3D, orn_3D_quat