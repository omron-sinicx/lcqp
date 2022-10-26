#
# Copyright (c) 2022 OMRON SINIC X Corporation
#

import pybullet
import math
import os 
import numpy as np

from . import projection
from scipy.spatial.transform import Rotation
from typing import List


class EndEffectorForceCommand:
    def __init__(self, link_index, local_position, spatial_force, null_space, op_space):
        self.link_index = link_index
        self.local_position = local_position
        self.spatial_force = spatial_force
        self.null_space = null_space
        self.op_space = op_space


class LCQPController:
    def __init__(self, lcqp_solver, projection_axis, z_offset):
        # self.path_to_urdf = iiwa_description.URDF_PATH
        self.path_to_urdf = os.path.dirname(__file__) + "/iiwa_description/urdf/iiwa14.urdf"
        self.movable_joints = [1, 2, 3, 4, 5, 6, 7, 11, 14]
        self.num_movable_joints = len(self.movable_joints)
        self.gripper_link_index = 17
        if projection_axis == 'x':
            opspace = np.array([0, 1, 1, -1, 0, 0])
            nullspace = np.array([1, 0, 0, 0, 0, 0])
        elif projection_axis == 'y':
            opspace = np.array([1, 0, 1, 0, -1, 0])
            nullspace = np.array([0, 1, 0, 0 , 0, 0])
        else:
            return NotImplementedError()
        self.left_finger_force_command = EndEffectorForceCommand(13, [0,0,0], np.zeros(6), nullspace, opspace)
        self.right_finger_force_command = EndEffectorForceCommand(16, [0,0,0], np.zeros(6), nullspace, opspace)
        self.gripper_surface_force_command = EndEffectorForceCommand(17, [0,0,0], np.zeros(6), nullspace, opspace)
        self.null_space_gain = np.full(6, 1000)
        self.op_space_gain = np.concatenate([np.full(3, 5), np.full(3, 0.5)])
        self.force_gain = np.ones(6)
        #lower limits for null space
        self.joint_lower_limits = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05, -0.0027, 0.0027]
        #upper limits for null space
        self.joint_upper_limits = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05, -0.055, 0.055]
        #joint ranges for null space
        self.joint_ranges = [5.8, 4, 5.8, 4, 5.8, 4, 6, 0.05, 0.05]
        #restposes for null space
        self.joint_rest_poses = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0, 0.05, 0.05]
        #joint damping coefficents
        # self.joint_damping = [0.01 for i in range(9)]
        self.joint_damping = [0.001 for i in range(9)]

        self.position_gain = np.concatenate([np.full(7, 0.02), np.full(2, 0.02)])
        self.max_joint_torque = np.concatenate([np.full(7, 500), np.full(2, 100)])

        self.lcqp_solver = lcqp_solver
        self.robot = None
        self.box = []
        self.cylinder = []
        self.half_cylinder = []

        if projection_axis == 'x':
            rot_base_gripper = np.array([[  0,  1,  0], 
                                         [  1,  0,  0],
                                         [  0,  0, -1]]) 
        elif projection_axis == 'y':
            rot_base_gripper = np.array([[-1,  0,  0], 
                                         [ 0,  0,  1],
                                         [ 0,  1,  0]]) 
            # rot_base_gripper = np.array([[ 1,  0,  0], 
            #                              [ 0,  1,  0],
            #                              [ 0,  0,  1]]) 
        else:
            return NotImplementedError()
        self.proj_gripper = projection.Projection(z_offset=z_offset, 
                                                  axis_pos=projection_axis, axis_rot='y',
                                                  rot_mat_3D_base=rot_base_gripper)
        self.proj_box = [] 
        self.proj_cylinder = [] 
        self.proj_half_cylinder = [] 

        self.projection_axis = projection_axis
        self.z_offset = z_offset

        self.pre_grasp_pos = None
        self.pre_pre_grasp_dist = np.array([0, 0, 0.5])
        self.pre_grasp_orn = None
        self.pre_grasp_tol = 1.0e-02
        self.pre_pre_grasp_tol = 1.0e-02
        self.post_grasp_pos = None
        self.post_grasp_orn = None
        self.post_grasp_tol = 1.0e-02
        self.q_lcqp_pre_grasp = None
        self.q_lcqp = []    
        self.f_lcqp = []    
        self.cd_lcqp = []
        self.fw_lcqp = []
        for e in lcqp_solver.lcqp.contacts:
            self.cd_lcqp.append([])
            self.fw_lcqp.append([])

        self.optuna_score = 0
        self.optuna_q_ref = None
        self.optuna_q_weight = None


    def save_lcqp_data(self):
        q_val = self.q_lcqp[-1]
        dq_val = self.lcqp_solver.get_solution()[0:self.lcqp_solver.dimq]
        self.lcqp_solver.lcqp.update_q_val(q_val)
        self.lcqp_solver.lcqp.update_dq_val(dq_val)
        for i in range(len(self.lcqp_solver.lcqp.contacts)):
            contact = self.lcqp_solver.lcqp.contacts[i]
            self.cd_lcqp[i].append([contact.get_contact_dist(), contact.get_contact_vel()])
            self.fw_lcqp[i].append([self.lcqp_solver.f_val[3*i], self.lcqp_solver.f_val[3*i+1]-self.lcqp_solver.f_val[3*i+2]])
            # self.fw_lcqp[i].append(contact.get_f_world(self.lcqp_solver.f_val[3*i:3*i+3]))


    def set_optuna(self, q_ref, q_weight):
        self.optuna_q_ref = q_ref
        self.optuna_q_weight = q_weight


    def register_robot(self, robot):
        self.robot = robot


    def register_box(self, box):
        self.box.append(box)
        self.proj_box.append(projection.Projection(z_offset=self.z_offset, 
                                                   axis_pos=self.projection_axis,
                                                   axis_rot=self.projection_axis))


    def register_cylinder(self, cylinder):
        self.cylinder.append(cylinder)
        self.proj_cylinder.append(projection.Projection(z_offset=self.z_offset, 
                                                        axis_pos=self.projection_axis,
                                                        axis_rot=self.projection_axis))


    def register_half_cylinder(self, half_cylinder):
        self.half_cylinder.append(half_cylinder)
        self.proj_half_cylinder.append(projection.Projection(z_offset=self.z_offset,
                                                             axis_pos=self.projection_axis,
                                                             axis_rot=self.projection_axis))


    def set_pre_grasp(self, pre_grasp_pos, pre_pre_grasp_dist=np.array([0, 0, 0.1]), pre_grasp_orn=None, 
                      pre_grasp_tol=1.0e-02, pre_pre_grasp_tol=1.0e-02):
        self.pre_grasp_pos = pre_grasp_pos
        self.pre_pre_grasp_dist = pre_pre_grasp_dist
        if pre_grasp_orn is not None:
            self.pre_grasp_orn = pre_grasp_orn
        else:
            self.pre_grasp_orn = Rotation.from_matrix([[  1,  0,  0], 
                                                       [  0, -1,  0], 
                                                       [  0,  0, -1]]).as_quat()
        self.pre_grasp_tol = pre_grasp_tol
        self.pre_pre_grasp_tol = pre_pre_grasp_tol


    def set_post_grasp(self, post_grasp_pos, post_grasp_orn=None, post_grasp_tol=1.0e-02):
        self.post_grasp_pos = post_grasp_pos
        if post_grasp_orn is not None:
            self.post_grasp_orn = post_grasp_orn
        else:
            self.post_grasp_orn = Rotation.from_matrix([[  1,  0,  0], 
                                                        [  0, -1,  0], 
                                                        [  0,  0, -1]]).as_quat()
        self.post_grasp_tol = post_grasp_tol


    def init(self, q=None):
        if isinstance(self.joint_lower_limits, np.ndarray):
            self.joint_lower_limits = self.joint_lower_limits.tolist()
        if isinstance(self.joint_upper_limits, np.ndarray):
            self.joint_upper_limits = self.joint_upper_limits.tolist()
        if isinstance(self.joint_ranges, np.ndarray):
            self.joint_ranges = self.joint_ranges.tolist()
        if isinstance(self.joint_rest_poses, np.ndarray):
            self.joint_rest_poses = self.joint_rest_poses.tolist()
        if isinstance(self.joint_damping, np.ndarray):
            self.joint_damping = self.joint_damping.tolist()


    def control_update(self, phase, debug=True, l_noise=None):
        q_lcqp = self.from_sim_to_lcqp()
        q_lcqp[2] = - q_lcqp[2]
        self.q_lcqp.append(q_lcqp.copy())

        if l_noise is not None:
            q_lcqp = q_lcqp + l_noise

        next_phase = phase
        success = True
        debug = False
        if debug:
            print('======== LCQP constraints ========')
            self.lcqp_solver.lcqp.update_q_val(q_lcqp)
            for e in self.lcqp_solver.lcqp.contacts:
                e.print()
            for e in self.lcqp_solver.lcqp.position_limits:
                e.print()
            print('======== LCQP constraints ========')
        if phase == 'pre-pre-grasp':
            goal_pos = self.pre_grasp_pos + self.pre_pre_grasp_dist
            gripper_diff = self.joint_rest_poses[8] - self.joint_rest_poses[7]
            joint_poses, diff_pos, diff_orn = self.ik(goal_pos=goal_pos, 
                                                      goal_orn=self.pre_grasp_orn,
                                                      gripper_diff=gripper_diff)

            self.apply_position_control(joint_poses)
            self.f_lcqp.append(np.zeros(self.lcqp_solver.lcqp.dimf))
            if np.max(np.abs(diff_pos)) < self.pre_pre_grasp_tol:
                next_phase = 'pre-grasp'
        elif phase == 'pre-grasp':
            gripper_diff = self.joint_rest_poses[8] - self.joint_rest_poses[7]
            joint_poses, diff_pos, diff_orn = self.ik(goal_pos=self.pre_grasp_pos, 
                                                      goal_orn=self.pre_grasp_orn,
                                                      gripper_diff=gripper_diff)
            self.apply_position_control(joint_poses)
            self.f_lcqp.append(np.zeros(self.lcqp_solver.lcqp.dimf))
            if np.max(np.abs(diff_pos)) < self.pre_grasp_tol:
                next_phase = 'grasp'
        elif phase == 'grasp':
            success = self.lcqp_solver.solve(q_lcqp)
            self.f_lcqp.append(self.lcqp_solver.f_val)
            if success:
                dimq = self.lcqp_solver.lcqp.dimq
                # print('lcqp: success, q_lcqp: ', q_lcqp, 'dq_lcqp: ', self.lcqp_solver.get_solution()[0:dimq])
                dq_lcqp = self.lcqp_solver.get_solution()[0:dimq]
                q1_lcqp = q_lcqp + dq_lcqp
                q1_lcqp[2] = - q1_lcqp[2]
                joint_poses, contact_forces, goal_pos, goal_orn, goal_gripper_diff = self.from_lcqp_to_sim(q1_lcqp)
                # print('contact_forces: ', contact_forces)
                if self.projection_axis == 'x':
                    self.right_finger_force_command.spatial_force = - np.array([contact_forces[4][0], contact_forces[4][0], contact_forces[4][1], 0, 0, 0])
                elif self.projection_axis == 'y':
                    self.right_finger_force_command.spatial_force = - np.array([contact_forces[8][0], contact_forces[8][0], contact_forces[8][1], 0, 0, 0])
                else:
                    return NotImplementedError()
                # force_commands = [self.left_finger_force_command, self.right_finger_force_command, self.gripper_surface_force_command]
                force_commands = [self.right_finger_force_command]
                self.apply_hybrid_control(joint_poses, goal_pos, goal_orn, goal_gripper_diff, force_commands)
                # self.apply_force_control(joint_poses, force_commands)
                # self.apply_position_control(joint_poses)
            else:
                print('lcqp: failure')
                next_phase = 'post-grasp'
                return next_phase, success  
        elif phase == 'post-grasp': 
            joint_poses, diff_pos, diff_orn = self.post_grasp_ik(goal_pos=self.post_grasp_pos, 
                                                                 goal_orn=self.post_grasp_orn)
            self.apply_position_control(joint_poses)
            self.f_lcqp.append(np.zeros(self.lcqp_solver.lcqp.dimf))
        else:
            return NotImplementedError() 
        return next_phase, success  


    def get_gripper_state(self):
        link_state = pybullet.getLinkState(self.robot, self.gripper_link_index)
        gripper_dist = pybullet.getJointState(self.robot, self.movable_joints[-1])[0] - pybullet.getJointState(self.robot, self.movable_joints[-2])[0]
        return link_state[4], link_state[5], gripper_dist


    def from_sim_to_lcqp(self):
        gripper_pos, gripper_orn, gripper_dist = self.get_gripper_state()
        gripper_pos_2D, gripper_orn_2D = self.proj_gripper.projection_3D_to_2D(gripper_pos, gripper_orn)
        q = np.concatenate([gripper_pos_2D, np.array([gripper_orn_2D, gripper_dist])]) 
        for box, proj in zip(self.box, self.proj_box):
            pos, orn = pybullet.getBasePositionAndOrientation(box)
            pos_2D, orn_2D = proj.projection_3D_to_2D(pos, orn)
            q = np.concatenate([q, pos_2D, np.array([orn_2D])])
        for cylinder, proj in zip(self.cylinder, self.proj_cylinder):
            pos, orn = pybullet.getBasePositionAndOrientation(cylinder)
            pos_2D, orn_2D = proj.projection_3D_to_2D(pos, orn)
            q = np.concatenate([q, pos_2D, np.array([orn_2D])])
        for half_cylinder, proj in zip(self.half_cylinder, self.proj_half_cylinder):
            pos, orn = pybullet.getBasePositionAndOrientation(half_cylinder)
            pos_2D, orn_2D = proj.projection_3D_to_2D(pos, orn)
            q = np.concatenate([q, pos_2D, np.array([orn_2D])])
        return q


    def from_lcqp_to_sim(self, q):
        gripper_pos_3D, gripper_orn_3D = self.proj_gripper.projection_2D_to_3D(q[0:2], q[2], 
                                                                               reset_nonprojected_direction=False)
                                                                            #    reset_nonprojected_direction=True)
        if self.projection_axis == 'x':
            gripper_pos_3D[0] = self.pre_grasp_pos[0]
        elif self.projection_axis == 'y':
            gripper_pos_3D[1] = self.pre_grasp_pos[1]
        else:
            return NotImplementedError()
        # for y-axis projection 
        if self.projection_axis == 'y':
            gripper_orn_3D_mat =  np.array([[ 1,  0,  0], 
                                            [ 0, -1,  0],
                                            [ 0,  0, -1]]) @ Rotation.from_quat(gripper_orn_3D).as_matrix()
            gripper_orn_3D = Rotation.from_matrix(gripper_orn_3D_mat).as_quat()
        joint_poses, diff_pos, diff_orn = self.ik(gripper_pos_3D, gripper_orn_3D, gripper_diff=q[3])
        joint_poses = np.array(joint_poses)
        contact_forces = []
        f = self.lcqp_solver.f_val
        for i in range(len(self.lcqp_solver.lcqp.contacts)):
            contact = self.lcqp_solver.lcqp.contacts[i]
            f_world = contact.get_f_world(f[i*3:i*3+3])
            contact_forces.append(f_world)
        # print('contact_forces:', contact_forces)
        return joint_poses, contact_forces, gripper_pos_3D, gripper_orn_3D, q[3]


    def ik(self, goal_pos, goal_orn, gripper_diff=0.0):
        joint_poses = pybullet.calculateInverseKinematics(self.robot, self.gripper_link_index, 
                                                          goal_pos, goal_orn, 
                                                          self.joint_lower_limits, 
                                                          self.joint_upper_limits, 
                                                          self.joint_ranges, 
                                                          self.joint_rest_poses,
                                                          self.joint_damping)
        joint_poses = np.array(joint_poses)
        joint_poses[7] = - 0.5 * gripper_diff
        joint_poses[8] =   0.5 * gripper_diff
        diff_pos, diff_orn = self.get_end_effector_difference(goal_pos, goal_orn)
        return joint_poses, diff_pos, diff_orn

    
    def get_end_effector_difference(self, goal_pos, goal_orn):
        link_state = pybullet.getLinkState(self.robot, self.gripper_link_index)
        diff_pos = link_state[4] - goal_pos
        diff_orn = pybullet.getDifferenceQuaternion(goal_orn, link_state[5])
        return diff_pos, diff_orn


    def apply_position_control(self, joint_poses):
        if type(joint_poses) is np.ndarray:
            joint_poses = joint_poses.tolist()
        q, dq  = self.get_joint_state()
        ff_torqes = pybullet.calculateInverseDynamics(self.robot, q.tolist(),
                                                      [0 for i in range(self.num_movable_joints)],
                                                      [0 for i in range(self.num_movable_joints)])
        ff_torqes = np.array(ff_torqes)
        for i in range(len(self.movable_joints)):
            pybullet.setJointMotorControl2(bodyIndex=self.robot,
                                           jointIndex=self.movable_joints[i],
                                           controlMode=pybullet.TORQUE_CONTROL,
                                           force=ff_torqes[i])
            pybullet.setJointMotorControl2(bodyIndex=self.robot,
                                           jointIndex=self.movable_joints[i],
                                           controlMode=pybullet.POSITION_CONTROL,
                                           targetPosition=joint_poses[i],
                                           positionGain=self.position_gain[i])


    def apply_hybrid_control(self, joint_poses, goal_pos, goal_orn, gripper_diff, ee_force: List[EndEffectorForceCommand]=[]):
        if type(joint_poses) is np.ndarray:
            joint_poses = joint_poses.tolist()
        q, dq  = self.get_joint_state()
        q = q.tolist()
        # ff_torqes = pybullet.calculateInverseDynamics(self.robot, q,
        ff_torqes = pybullet.calculateInverseDynamics(self.robot, joint_poses,
                                                      [0 for i in range(self.num_movable_joints)],
                                                      [0 for i in range(self.num_movable_joints)])
        ff_torqes = np.array(ff_torqes)
        if len(ee_force) >= 1:
            for e in ee_force:
                J_trans, J_rot = pybullet.calculateJacobian(self.robot, e.link_index, 
                                                            # e.local_position, q, 
                                                            e.local_position, joint_poses, 
                                                            [0 for i in range(self.num_movable_joints)],
                                                            [0 for i in range(self.num_movable_joints)])
                ee_force = e.spatial_force
                if type(ee_force) is not np.ndarray:
                    ee_force = np.array(ee_force)
                J = np.concatenate([np.array(J_trans), np.array(J_rot)])
                ff_torqes = ff_torqes - J.T @ (self.force_gain*ee_force)
            J_trans, J_rot = pybullet.calculateJacobian(self.robot, self.gripper_link_index, 
                                                        [0, 0, 0], joint_poses, 
                                                        [0 for i in range(self.num_movable_joints)],
                                                        [0 for i in range(self.num_movable_joints)])
            J = np.concatenate([np.array(J_trans), np.array(J_rot)])
            diff_pos, diff_orn = self.get_end_effector_difference(goal_pos, goal_orn)
            diff_ee = np.concatenate([np.array(diff_pos), np.array(diff_orn)[0:3]])
            # print('diff_ee:', diff_ee)
            ff_torqes = ff_torqes - J.T @ (self.gripper_surface_force_command.null_space*diff_ee*self.null_space_gain)
            ff_torqes = ff_torqes - J.T @ (self.gripper_surface_force_command.op_space*diff_ee*self.op_space_gain)
        for i in range(len(self.movable_joints)-2):
            pybullet.setJointMotorControl2(bodyIndex=self.robot,
                                           jointIndex=self.movable_joints[i],
                                           controlMode=pybullet.TORQUE_CONTROL,
                                           force=ff_torqes[i])
            pybullet.setJointMotorControl2(bodyIndex=self.robot,
                                           jointIndex=self.movable_joints[i],
                                           controlMode=pybullet.POSITION_CONTROL,
                                           targetPosition=joint_poses[i],
                                           positionGain=self.position_gain[i])
                                        #    force=0)
            # pybullet.setJointMotorControl2(bodyIndex=self.robot,
            #                                jointIndex=self.movable_joints[i],
            #                                controlMode=pybullet.VELOCITY_CONTROL,
            #                                force=0)
        for i in [7, 8]:
            pybullet.setJointMotorControl2(bodyIndex=self.robot,
                                           jointIndex=self.movable_joints[i],
                                           controlMode=pybullet.TORQUE_CONTROL,
                                           force=ff_torqes[i])
            pybullet.setJointMotorControl2(bodyIndex=self.robot,
                                           jointIndex=self.movable_joints[i],
                                           controlMode=pybullet.POSITION_CONTROL,
                                           targetPosition=joint_poses[i],
                                           positionGain=self.position_gain[i])


    def apply_force_control(self, joint_poses, ee_force: List[EndEffectorForceCommand]=[]):
        if type(joint_poses) is np.ndarray:
            joint_poses = joint_poses.tolist()
        q, dq  = self.get_joint_state()
        q = q.tolist()
        ff_torqes = pybullet.calculateInverseDynamics(self.robot, q,
                                                      [0 for i in range(self.num_movable_joints)],
                                                      [0 for i in range(self.num_movable_joints)])
        ff_torqes = np.array(ff_torqes)
        if len(ee_force) >= 1:
            for e in ee_force:
                J_trans, J_rot = pybullet.calculateJacobian(self.robot, e.link_index, 
                                                            e.local_position, q, 
                                                            [0 for i in range(self.num_movable_joints)],
                                                            [0 for i in range(self.num_movable_joints)])
                ee_force = e.spatial_force
                if type(ee_force) is not np.ndarray:
                    ee_force = np.array(ee_force)
                J = np.concatenate([np.array(J_trans), np.array(J_rot)])
                # ff_torqes = ff_torqes - J.T @ ee_force
                # ff_torqes = ff_torqes + J.T @ ee_force
        for i in range(len(self.movable_joints)-2):
            pybullet.setJointMotorControl2(bodyIndex=self.robot,
                                           jointIndex=self.movable_joints[i],
                                           controlMode=pybullet.TORQUE_CONTROL,
                                           force=ff_torqes[i])
            pybullet.setJointMotorControl2(bodyIndex=self.robot,
                                           jointIndex=self.movable_joints[i],
                                           controlMode=pybullet.POSITION_CONTROL,
                                           force=0)
            pybullet.setJointMotorControl2(bodyIndex=self.robot,
                                           jointIndex=self.movable_joints[i],
                                           controlMode=pybullet.VELOCITY_CONTROL,
                                           force=0)
        for i in [len(self.movable_joints)-2, len(self.movable_joints)-1]:
            pybullet.setJointMotorControl2(bodyIndex=self.robot,
                                           jointIndex=self.movable_joints[i],
                                           controlMode=pybullet.POSITION_CONTROL,
                                           targetPosition=joint_poses[i],
                                           positionGain=self.position_gain[i])


    def test_projection(self):
        print("----- projection test gripper -----")
        gripper_pos, gripper_orn, gripper_dist = self.get_gripper_state()
        self.proj_gripper.test_projection(gripper_pos, gripper_orn)
        print("-----------------------------------")
        # for obj, proj in zip(self.box, self.proj_box):
        #     print("----- projection test box -----")
        #     pos, orn = pybullet.getBasePositionAndOrientation(obj)
        #     proj.test_projection(pos, orn)
        #     print("-----------------------------------")
        # for obj, proj in zip(self.cylinder, self.proj_box):
        #     print("----- projection test cylinder -----")
        #     pos, orn = pybullet.getBasePositionAndOrientation(obj)
        #     proj.test_projection(pos, orn)
        #     print("-----------------------------------")
        # for obj, proj in zip(self.half_cylinder, self.proj_box):
        #     print("----- projection test half_cylinder -----")
        #     pos, orn = pybullet.getBasePositionAndOrientation(obj)
        #     proj.test_projection(pos, orn)
        #     print("-----------------------------------")
        # for obj, proj in zip(self.half_cylinder, self.proj_box):
        #     print("----- projection test half_cylinder -----")
        #     pos, orn = pybullet.getBasePositionAndOrientation(obj)
        #     proj.test_projection(pos, orn)
        #     print("-----------------------------------")


    def calc_optuna_score(self):
        if self.optuna_q_ref is not None and self.optuna_q_weight is not None:
            q = self.from_sim_to_lcqp()
            qdiff = q - self.optuna_q_ref
            score = qdiff.T @ np.diag(self.optuna_q_weight) @ qdiff
            self.optuna_score = self.optuna_score + score


    def get_joint_state(self):
        q = np.zeros(self.num_movable_joints)
        dq = np.zeros(self.num_movable_joints)
        for i in range(self.num_movable_joints):
            q[i] = pybullet.getJointState(self.robot, self.movable_joints[i])[0]
            dq[i] = pybullet.getJointState(self.robot, self.movable_joints[i])[1]
        return q, dq