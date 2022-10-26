#
# Copyright (c) 2022 OMRON SINIC X Corporation
#

import pybullet
from datetime import datetime
import pybullet_data
import time
import math
import numpy as np

import os
from . import utils
from .controller import LCQPController


class PyBulletSimulation:
    def __init__(self, time_step, sim_time, gui=True):
        self.time_step = time_step
        self.sim_time = sim_time 
        self.sim_steps = math.floor(sim_time/time_step)
        # self.path_to_urdf = iiwa_description.URDF_PATH
        self.path_to_urdf = os.path.dirname(__file__) + "/iiwa_description/urdf/iiwa14.urdf"
        self.movable_joints = [1, 2, 3, 4, 5, 6, 7, 11, 13]
        self.num_movable_joints = len(self.movable_joints)

        self.robot = None
        self.robot_q0 = np.concatenate([np.zeros(7), np.array([-0.05, 0.05])])
        self.table_width  = 2.0
        self.table_length = 1.0
        self.table_height = 0.75
        self.table_thickness = 0.04
        self.table_x0 = 0.0
        self.table_y0 = 0.0
        self.table_surface_height = self.table_height # + 0.5 * self.table_thickness
        self.box = []
        self.box_config = []
        self.box_q0 = []
        self.gui = gui
        self.camera = 2.5, 45, -30, [0, 0, 0.75]
        self.contact_dist = []
        self.f_normal = []

        self.noise_var_model = 0
        self.noise_var_measurement = 0

    def set_camera_default(self):
        self.camera = 2.5, 45, -30, [0, 0, 0.75]

    def set_camera_front(self):
        self.camera = 2.5, 100, -10, [0, 0, 0.75]

    def set_camera_side(self):
        self.camera = 2.5, 85, -12, [-1, 0.0, 0.5]

    def print_info(self):
        pybullet.connect(pybullet.DIRECT)
        robot = pybullet.loadURDF(self.path_to_urdf, [0, 0, 0], useFixedBase=True)
        for j in range(pybullet.getNumJoints(robot)):
            info = pybullet.getJointInfo(robot, j)
            print(info)
        for b in range(pybullet.getNumBodies(robot)):
            info = pybullet.getBodyInfo(robot, b)
            print(info)
        pybullet.disconnect()

    def add_box(self, lx, ly, lz, mass, x, y, z=None):
        noise_var = self.noise_var_model
        noise = np.random.normal(0, noise_var, 3)

        l_noise = [noise[0]*lx, noise[1]*ly, noise[2]*lz]
        self.l_noise = [lx, ly, lz]
        self.box_config.append([lx+l_noise[0], ly+l_noise[1], lz+l_noise[2], mass])
        if z is None:
            z = self.table_surface_height + 0.5 * (lz + noise[2])
        self.box_q0.append([x, y, z])

    def run(self, controller: LCQPController, debug: bool=False, anim_2D: bool=False, record: bool=False, optuna_score=True):
        if self.gui:
            pybullet.connect(pybullet.GUI)
        else:
            pybullet.connect(pybullet.DIRECT)
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        pybullet.resetDebugVisualizerCamera(self.camera[0], self.camera[1], self.camera[2], self.camera[3])
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
        pybullet.setGravity(0, 0, -9.81)
        import math
        pybullet.setPhysicsEngineParameter(collisionFilterMode=1)
        # pybullet.setPhysicsEngineParameter(numSubSteps=math.floor(self.time_step/0.001))
        plane = pybullet.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)
        table = utils.create_table(width=self.table_width, length=self.table_length, 
                                   height=self.table_height, thickness=self.table_thickness,
                                   box=True)
        pybullet.changeDynamics(table, -1, lateralFriction=0.5, rollingFriction=0, spinningFriction=0)
        self.robot = pybullet.loadURDF(self.path_to_urdf, [0, 0, self.table_length+0.5*self.table_thickness], useFixedBase=True)
        pybullet.changeDynamics(self.robot, -1, lateralFriction=0.5, rollingFriction=0.0, spinningFriction=0.0)
        pybullet.resetBasePositionAndOrientation(self.robot, [0, 0, 0.7], [0, 0, 0, 1])
        controller.register_robot(self.robot)
        for config in self.box_config:
            lx, ly, lz, mass = config
            box = utils.create_box(lx, ly, lz, mass)
            pybullet.changeDynamics(box, -1, lateralFriction=0.5, rollingFriction=0, spinningFriction=0)
            self.box.append(box)
            controller.register_box(box)

        # init 
        for i in range(self.num_movable_joints):
            pybullet.resetJointState(self.robot, self.movable_joints[i], self.robot_q0[i])
        for box, box_q0 in zip(self.box, self.box_q0):
            for i in range(3):
                x, y, z = box_q0
                pybullet.resetBasePositionAndOrientation(box, [x, y, z], [0, 0, 0, 1])
        controller.init(self.robot_q0)
        t = 0.

        use_real_time_simulation = False
        pybullet.setRealTimeSimulation(use_real_time_simulation)
        #trail_duration is duration (in seconds) after debug lines will be removed automatically (use 0 for no-removal)

        if record:
            pybullet.startStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4, 'sim.mp4')

        noise_var = self.noise_var_measurement
        gripper = controller.lcqp_solver.lcqp.objects[0]
        box = controller.lcqp_solver.lcqp.objects[1]
        var_scaling = np.array([gripper.h, gripper.h, 0.001, gripper.rmax, box.w, box.h, 0.001])

        phase = 'pre-pre-grasp'
        for sim_step in range(self.sim_steps):
            if use_real_time_simulation:
                dt = datetime.now()
                t = (dt.second / 60.) * 2. * math.pi
            else:
                t = t + self.time_step
            pybullet.stepSimulation()
            time.sleep(self.time_step)

            if sim_step%10 == 0:
                noise = np.random.normal(np.zeros(7), noise_var*var_scaling)
            next_phase, success = controller.control_update(phase, debug, noise)

            if phase == 'grasp':
                controller.save_lcqp_data()
            phase = next_phase

            if optuna_score:
                controller.calc_optuna_score()

            if not success:
                if optuna_score:
                    penalty = 1.0e08
                    controller.optuna_score = controller.optuna_score + penalty * (self.sim_steps-sim_step) 
                break

        pybullet.disconnect()

        return success