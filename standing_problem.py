import crocoddyl
from crocoddyl import ResidualModelContactForce, ConstraintModelResidual, ActivationModelQuadraticBarrier
import numpy as np
from numpy import pi
import pinocchio as pin
from pinocchio import SE3
import matplotlib.pyplot as plt
import time
from copy import deepcopy


robot = pin.RobotWrapper.BuildFromURDF("simple_grace.urdf",
                                       root_joint=pin.JointModelFreeFlyer())
robot.model.createData()

state_dim = robot.nq + robot.nv
# robot.model.gravity = np.array([0, 90000.81,0])
robot.model.gravity = pin.Motion(np.array([0, 0, 9.81]), np.zeros(3))  # Linear gravity, zero angular velocity

period = 10
dt = 0.02
T = int(period *0.25/ dt)

mu = 0.7
Rsurf = np.eye(3)

q_ref = np.array([
    0, 0, 0,
    0, 0, 0, 1,
    -pi / 4, 0, pi / 2,
    pi / 4, 0, pi / 2,
    pi / 4, 0, pi / 2,
    -pi / 4, 0, pi / 2,
])

meshcat_display = crocoddyl.MeshcatDisplay(robot)
meshcat_display.display([q_ref])

qp_ref = np.zeros(robot.nv)

robot.model.referenceConfigurations["default"] = q_ref
pin.forwardKinematics(robot.model, robot.data, q_ref)
pin.updateFramePlacements(robot.model, robot.data)

state = crocoddyl.StateMultibody(robot.model)
actuation = crocoddyl.ActuationModelFloatingBase(state)

contact_models = crocoddyl.ContactModelMultiple(state, actuation.nu)
# Add specific foot contacts
contact_points = ['LF-FOOT', 'LR-FOOT', 'RF-FOOT', 'RR-FOOT']
foot_frame_ids = [robot.model.getFrameId(foot_name) for foot_name in contact_points]

common_costs = crocoddyl.CostModelSum(state, actuation.nu)

for foot_frame_id, foot_name in zip(foot_frame_ids, contact_points):
    # Define the 3D contact model for the foot
    contact_model = crocoddyl.ContactModel3D(state, foot_frame_id, pin.SE3.Identity().translation,
                                             pin.LOCAL_WORLD_ALIGNED, actuation.nu,
                                             np.array([0, 0]))
    contact_models.addContact(f'{foot_name}_contact', contact_model)

    cone = crocoddyl.FrictionCone(Rsurf, mu, 4, 0, 10)
    wrenchResidual = crocoddyl.ResidualModelContactFrictionCone(state, foot_frame_id, cone, actuation.nu, True)
    wrenchActivation = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(cone.lb, cone.ub))
    wrenchCone = crocoddyl.CostModelResidual(state, wrenchActivation, wrenchResidual)
    common_costs.addCost(robot.model.frames[foot_frame_id].name + "_wrenchCone", wrenchCone, 100)

grace_frame_id = robot.model.getFrameId('grace')
start_grace_body_pos = robot.model.frames[grace_frame_id].placement.translation

costs = []
integrated_action_model = []
for t in range(T):
    # Composite cost
    costs = deepcopy(common_costs)
    target_pos = start_grace_body_pos + np.array([0, 0, -0.1 * np.sin(2 * np.pi * t * dt / period)])
    target_body_pose = SE3(np.eye(3), target_pos)
    residual_frame_placement = crocoddyl.ResidualModelFramePlacement(state, grace_frame_id, target_body_pose,
                                                                     actuation.nu)
    tracking_cost = crocoddyl.CostModelResidual(state, crocoddyl.ActivationModelQuad(residual_frame_placement.nr),
                                                residual_frame_placement)
    costs.addCost('body_height_cost', tracking_cost, weight=1.0)

    # constraint_manager = crocoddyl.ConstraintModelManager(state, actuation.nu)
    #
    # for i, foot_frame_id in enumerate(foot_frame_ids):
    #     force_lower = np.array([-np.inf, -np.inf, -100.0])  # Min forces in X, Y, Z directions
    #     force_upper = np.array([np.inf, np.inf, 100.0])  # Max forces in X, Y, Z directions
    #     bounds = crocoddyl.ActivationBounds(force_lower, force_upper)
    #     activation = ActivationModelQuadraticBarrier(bounds)
    #
    #     residual = ResidualContactForceWorldFrame(robot, state, foot_frame_id, np.zeros(3), actuation.nu)
    #     constraint_manager.addConstraint(f'foot{i}_suction_constr',
    #                                      ConstraintModelResidual(state, residual))

    # action_model = crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, contact_models, costs, constraint_manager)
    action_model = crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, contact_models, costs)
    integrated_action_model.append(crocoddyl.IntegratedActionModelEuler(action_model, dt))

x0 = np.concatenate([q_ref, qp_ref])

problem = crocoddyl.ShootingProblem(x0, integrated_action_model, integrated_action_model[-1])
solver = crocoddyl.SolverDDP(problem)

# Set an initial guess for the trajectory
xs = [x0] * (T + 1)
us = [np.zeros(actuation.nu)] * T

# Solve the problem
solver.solve(xs, us, maxiter=100)

contact_forces_world = []

# Iterate over the time horizon
contact_forces = [[] for _ in range(4)]
positions = []
for t, data in enumerate(solver.problem.runningDatas):
    # Extract the forces at time t
    for i, ct_point in enumerate(contact_points):
        contact_forces[i].append(data.differential.multibody.contacts.contacts[f'{ct_point}_contact'].fext.linear)

    pin.framesForwardKinematics(robot.model, robot.data, solver.xs[t + 1][:robot.model.nq])
    pin.updateFramePlacements(robot.model, robot.data)

    # Get the frame's position in the world frame
    positions.append(robot.data.oMf[grace_frame_id].translation.copy())

_, ax0 = plt.subplots(nrows=3, ncols=1)
for i in range(3):
    ax0[i].plot([contact_forces[0][t][i] for t in range(len(contact_forces[0]))])

_, ax1 = plt.subplots(nrows=3, ncols=1)
for i in range(3):
    ax1[i].plot([contact_forces[1][t][i] for t in range(len(contact_forces[1]))])

_, ax2 = plt.subplots(nrows=3, ncols=1)
for i in range(3):
    ax2[i].plot([contact_forces[2][t][i] for t in range(len(contact_forces[2]))])

_, ax3 = plt.subplots(nrows=3, ncols=1)
for i in range(3):
    ax3[i].plot([contact_forces[3][t][i] for t in range(len(contact_forces[3]))])

_, ax4 = plt.subplots(nrows=3, ncols=1)
for i in range(3):
    ax4[i].plot([positions[t][i] for t in range(len(positions))])

plt.show()

while True:
    # meshcat_display.display([q_ref])
    meshcat_display.displayFromSolver(solver, factor=0.2)
    time.sleep(1.0)
