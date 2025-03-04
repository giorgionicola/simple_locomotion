import crocoddyl
from crocoddyl import ResidualModelContactForce, ConstraintModelResidual, ActivationModelQuadraticBarrier, \
    CostDataResidual, CostModelResidual
import numpy as np
from numpy import pi
import pinocchio as pin
from pinocchio import SE3
import matplotlib.pyplot as plt
import time
from copy import deepcopy


class ResidualContactForceWorldFrame(crocoddyl.ResidualModelAbstract):
    def __init__(self, state, frame_id, contact_id, desired_force, nu=None):
        """
        Residual model for contact forces in the world frame.

        Args:
            state: State of the system (crocoddyl.StateMultibody).
            frame_id: Frame ID of the contact point.
            desired_force: Desired contact force in world frame (3D vector).
            nu: Dimension of the control input (optional, defaults to state.nv).
        """
        nu = state.nv if nu is None else nu  # Default to state.nv if not provided
        nr = 3
        super().__init__(state, nr, nu, True, True, True)  # 3D residual for Fx, Fy, Fz in world frame
        self.frame_id = frame_id
        self.contact_id = contact_id
        self.desired_force = desired_force

    def calc(self, data, x, u):
        """
        Compute the residual vector: r = world_force - desired_force.
        """
        # Forward kinematics
        pin.framesForwardKinematics(self.state.pinocchio, data.shared.pinocchio, x[:self.state.nq])

        # Get the contact frame's rotation matrix (local to world)
        R_world_contact = data.shared.pinocchio.oMf[self.frame_id].rotation

        # Get the local contact force from the action model's contact data
        local_force = data.shared.contacts.contacts[f'{self.contact_id}'].fext.linear

        # Transform the force to the world frame
        world_force = R_world_contact @ local_force

        # Compute the residual
        data.r = world_force - self.desired_force

    def calcDiff(self, data, x, u):
        """
        Compute the Jacobians of the residual (dr/dx and dr/du).
        """
        q = x[:self.state.nq]
        v = x[self.state.nq:]

        # Update Pinocchio kinematics
        pin.framesForwardKinematics(self.state.pinocchio, data.shared.pinocchio, x[:self.state.nq])

        # Compute frame Jacobian (local frame to world)

        J_frame = pin.computeFrameJacobian(self.state.pinocchio, self.state.pinocchio.createData(), q, self.frame_id, pin.ReferenceFrame.LOCAL)

        # Compute residual Jacobians
        data.Rx[:, :18] = -J_frame[:3]  # Derivative of f_world w.r.t. state
        if u is not None:
            # Assume no direct dependence on u in this example
            data.Ru[:, :] = 0


robot = pin.RobotWrapper.BuildFromURDF("simple_grace.urdf",
                                       root_joint=pin.JointModelFreeFlyer())
robot.model.createData()

state_dim = robot.nq + robot.nv
# robot.model.gravity = np.array([0, 90000.81,0])
robot.model.gravity = pin.Motion(np.array([0, 0, -9.81]), np.zeros(3))  # Linear gravity, zero angular velocity

period = 5
dt = 0.02
T = int(period / dt)

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

force_bounds = crocoddyl.ActivationBounds(
    lb=np.array([-50.0, -50.0, 0.0]),  # Lower bounds for Fx, Fy, Fz
    ub=np.array([50.0, 50.0, 100.0])  # Upper bounds for Fx, Fy, Fz
)
force_activation = ActivationModelQuadraticBarrier(force_bounds)

constraint_manager = crocoddyl.ConstraintModelManager(state, actuation.nu)

for foot_frame_id, foot_name in zip(foot_frame_ids, contact_points):
    # Define the 3D contact model for the foot
    contact_model = crocoddyl.ContactModel3D(state, foot_frame_id, pin.SE3.Identity().translation,
                                             pin.LOCAL, actuation.nu,
                                             np.array([0, 50]))
    contact_models.addContact(f'{foot_name}_contact', contact_model)

    # cone = crocoddyl.FrictionCone(Rsurf, mu, 8, True, 0, 10)
    # wrenchResidual = crocoddyl.ResidualModelContactFrictionCone(state, foot_frame_id, cone, actuation.nu, True)
    # wrenchActivation = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(cone.lb, cone.ub))
    # wrenchCone = crocoddyl.CostModelResidual(state, wrenchActivation, wrenchResidual)
    # common_costs.addCost(robot.model.frames[foot_frame_id].name + "_wrenchCone", wrenchCone, weight=1)

    residual_world_force = ResidualContactForceWorldFrame(state=state,
                                                          frame_id=foot_frame_id,
                                                          contact_id=f'{foot_name}_contact',
                                                          desired_force=np.array([0.0, 0.0, 0.0]),
                                                          # Desired force in world frame
                                                          nu=actuation.nu
                                                          # Match the control dimension of the action model
                                                          )
    # force_cost = CostModelResidual(state, force_activation, residual_world_force, )
    force_cost =CostModelResidual(state,crocoddyl.ActivationModelQuad(residual_world_force.nr), residual_world_force)
    common_costs.addCost(robot.model.frames[foot_frame_id].name + '_force_cost', force_cost, 100)

grace_frame_id = robot.model.getFrameId('grace')
start_grace_body_pos = robot.model.frames[grace_frame_id].placement.translation

costs = []
integrated_action_model = []
for t in range(T):
    # Composite cost
    costs.append(deepcopy(common_costs))
    target_pos = start_grace_body_pos + np.array([0, -0.1 * np.sin(2 * np.pi * t * dt / period) * 0, 0])
    target_body_pose = SE3(np.eye(3), target_pos)
    residual_frame_placement = crocoddyl.ResidualModelFramePlacement(state, grace_frame_id, target_body_pose,
                                                                     actuation.nu)
    tracking_cost = crocoddyl.CostModelResidual(state, crocoddyl.ActivationModelQuad(residual_frame_placement.nr),
                                                residual_frame_placement)
    costs[-1].addCost('body_height_cost', tracking_cost, weight=1.)

    action_model = crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, contact_models, costs[-1],
                                                                       constraint_manager)
    integrated_action_model.append(crocoddyl.IntegratedActionModelEuler(action_model, dt))

x0 = np.concatenate([q_ref, qp_ref])

problem = crocoddyl.ShootingProblem(x0, integrated_action_model, integrated_action_model[-1])
solver = crocoddyl.SolverDDP(problem)

# Set an initial guess for the trajectory
xs = [x0] * (T + 1)
us = [np.zeros(actuation.nu)] * T

# Solve the problem
solver.solve(xs, us, maxiter=1)

contact_forces_world = []

# Iterate over the time horizon
contact_forces = [[] for _ in range(4)]
positions = []
pred_costs = {
    'body_height_cost': [],
              'LF-FOOT_force_cost': [],
              'LR-FOOT_force_cost': [],
              'RF-FOOT_force_cost': [],
              'RR-FOOT_force_cost': [],
              }
cost_keys = list(pred_costs.keys())

for t, data in enumerate(solver.problem.runningDatas):
    # Extract the forces at time t
    pin.framesForwardKinematics(robot.model, robot.data, solver.xs[t + 1][:robot.model.nq])
    pin.updateFramePlacements(robot.model, robot.data)

    for i, (ct_point, frame_id) in enumerate(zip(contact_points, foot_frame_ids)):
        local_force = data.differential.multibody.contacts.contacts[f'{ct_point}_contact'].fext.linear
        R_world_contact = data.differential.pinocchio.oMf[frame_id].rotation

        contact_forces[i].append(R_world_contact @ local_force)

    # Get the frame's position in the world frame
    positions.append(robot.data.oMf[grace_frame_id].translation.copy())
    for key in cost_keys:
        pred_costs[key].append(data.differential.costs.costs[key].cost)

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

_, ax5 = plt.subplots()
for key in cost_keys:
    ax5.plot(pred_costs[key], label=key)
plt.legend()

plt.show()

while True:
    # meshcat_display.display([q_ref])
    meshcat_display.displayFromSolver(solver, factor=0.2)
    time.sleep(1.0)
