import lcqp_manip
import casadi
import numpy as np
import lcqpow


# create the objects
box_ly = 0.15
vmax = 0.035
wmax = 0.035
box = lcqp_manip.Box(w=0.24, h=0.08, m=0.1, g=9.81, vmax=casadi.SX([vmax, vmax, wmax]))
gripper = lcqp_manip.Gripper(h=0.08, rmax=0.115, rmin=0, vmax=casadi.SX([vmax, vmax, wmax, vmax]))
ground = lcqp_manip.Ground()
wall = lcqp_manip.Wall(-0.25)
objects = [gripper, box, ground, wall]

# create the contact points and surfaces 
box_TR_corner = lcqp_manip.BoxTopRightCorner(box)
box_BR_corner = lcqp_manip.BoxBottomRightCorner(box)
box_TL_corner = lcqp_manip.BoxTopLeftCorner(box)
box_BL_corner = lcqp_manip.BoxBottomLeftCorner(box)
box_T_surface = lcqp_manip.BoxTopSurface(box)
box_B_surface = lcqp_manip.BoxBottomSurface(box)
box_R_surface = lcqp_manip.BoxRightSurface(box)
box_L_surface = lcqp_manip.BoxLeftSurface(box)
gripper_BR_corner = lcqp_manip.GripperBottomRightCorner(gripper, offset=0.0035)
gripper_BL_corner = lcqp_manip.GripperBottomLeftCorner(gripper, offset=0.0035)


# create contacts 
fmax = casadi.SX([10*box.m*box.g, 10*box.m*box.g, 10*box.m*box.g])
mu_ground = 0.4
mu = 0.5
# contacts between the box (contact points) and ground (contact surface)
contact_box_TR_ground = lcqp_manip.RelaxedContact(contact_point=box_TR_corner, contact_surface=ground, 
                                                  contact_name="box_TR_ground", mu=mu, fmax=fmax,
                                                  inv_force_dir=False)
contact_box_BR_ground = lcqp_manip.RelaxedContact(contact_point=box_BR_corner, contact_surface=ground, 
                                                  contact_name="box_BR_ground", mu=mu, fmax=fmax,
                                                  inv_force_dir=False)
contact_box_TL_ground = lcqp_manip.RelaxedContact(contact_point=box_TL_corner, contact_surface=ground, 
                                                  contact_name="box_TL_ground", mu=mu, fmax=fmax,
                                                  inv_force_dir=False)
contact_box_BL_ground = lcqp_manip.RelaxedContact(contact_point=box_BL_corner, contact_surface=ground, 
                                                  contact_name="box_BL_ground", mu=mu, fmax=fmax,
                                                  inv_force_dir=False)
# contacts between the box (contact points) and ground (contact surface)
contact_box_TR_wall = lcqp_manip.RelaxedContact(contact_point=box_TR_corner, contact_surface=wall, 
                                                contact_name="box_TR_wall", mu=mu, 
                                                inv_force_dir=False)
contact_box_BR_wall = lcqp_manip.RelaxedContact(contact_point=box_BR_corner, contact_surface=wall, 
                                                contact_name="box_BR_wall", mu=mu, 
                                                inv_force_dir=False)
contact_box_TL_wall = lcqp_manip.RelaxedContact(contact_point=box_TL_corner, contact_surface=wall, 
                                                contact_name="box_TL_wall", mu=mu, 
                                                inv_force_dir=False)
contact_box_BL_wall = lcqp_manip.RelaxedContact(contact_point=box_BL_corner, contact_surface=wall, 
                                                contact_name="box_BL_wall", mu=mu, 
                                                inv_force_dir=False)
# contacts between the gripper (contact points) and box (contact surfaces)
contact_gripper_BR_box_R = lcqp_manip.RelaxedContact(contact_point=gripper_BR_corner, contact_surface=box_R_surface, 
                                                     contact_name="gripper_BR_box_R", mu=mu, fmax=fmax,
                                                     inv_force_dir=True)
contact_gripper_BL_box_R = lcqp_manip.RelaxedContact(contact_point=gripper_BL_corner, contact_surface=box_R_surface, 
                                                     contact_name="gripper_BL_box_L", mu=mu, fmax=fmax,
                                                     inv_force_dir=True)

contacts = [contact_box_TL_ground, contact_box_BL_ground, contact_box_TR_ground, contact_box_BR_ground,
            contact_box_TL_wall, contact_box_BL_wall, contact_box_TR_wall, contact_box_BR_wall,
            contact_gripper_BR_box_R]


# create an LCQP
lcqp = lcqp_manip.LCQP(objects, contacts)
lcqp.set_force_balance(box)
lcqp.set_position_limit(gripper_BR_corner, box_T_surface, margin=0.02, inv_dir=True)
lcqp.set_position_limit(gripper_BR_corner, box_B_surface, margin=0.02, inv_dir=True)

box_x0 = 0.5
box_y0 =  wall.w0 + box.w / 2 + 0.03

# goal configuration
box_center_to_gripper_top = gripper.h / 2
goal_height = 0.0
goal_angle = np.pi / 2

gripper_z_start = box.h + box_center_to_gripper_top 
box_z_goal = box.w / 2
q_goal = np.array([box_y0, 0, -np.pi/6, 0.01, box_y0, box_z_goal, goal_angle])

# set config cost
q_weight = np.array([0.0, 0.0, 0.001, 1000, 0, 1, 1000]) 
v_weight = 1.0e-02 
f_weight = 1.0e-02 
slack_penalty = 1.0e04
lcqp.set_config_cost(q_goal, q_weight, v_weight, f_weight, slack_penalty)

# create the LCQP solver
lcqp_solver = lcqp_manip.LCQPSolver(lcqp)
lcqp_solver.options.setMaxRho(1.0e12)
lcqp_solver.options.setComplementarityTolerance(1.0e-06)
lcqp_solver.options.setStationarityTolerance(1.0e-04)

# create the simulation environment
sim = lcqp_manip.PyBulletSimulation(time_step=0.05, sim_time=30, gui=True)

sim.robot_q0 = np.array([-np.pi/2, 0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, -0.01, 0.01])
sim.joint_rest_poses = sim.robot_q0.tolist()
sim.add_box(lx=box_ly, ly=box.w, lz=box.h, mass=box.m, x=box_x0, y=box_y0)

# create the controller
controller = lcqp_manip.LCQPController(lcqp_solver=lcqp_solver, projection_axis='x', z_offset=sim.table_height)
controller.joint_rest_poses = sim.robot_q0 
controller.position_gain = np.concatenate([np.full(7, 1), np.full(2, 0.1)])
controller.op_space_gain = np.concatenate([np.full(3, 100), np.full(3, 100)])

# run a simulation
pre_grasp_orn = np.array([[  0,  1,  0], 
                          [  0,  0, -1],
                          [ -1,  0,  0]])
from scipy.spatial.transform import Rotation
pre_grasp_orn = Rotation.from_matrix(pre_grasp_orn).as_quat()
controller.set_pre_grasp(pre_grasp_pos=np.array([box_x0, box_y0+box.w/2+gripper.h, sim.table_surface_height+box.h/2+gripper.h/2]), 
                            pre_grasp_orn=pre_grasp_orn, 
                            pre_pre_grasp_dist=np.array([0.0, 0.1, 0.0]), 
                            pre_grasp_tol=0.01, pre_pre_grasp_tol=0.02)
controller.set_post_grasp(post_grasp_pos=np.array([box_x0, 0.0, gripper_z_start+sim.table_surface_height+0.1]), 
                            post_grasp_tol=0.01)

sim.set_camera_side()
sim.run(controller, record=False)
# sim.run(controller, anim_2D=False, record=True)
