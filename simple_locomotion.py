import crocoddyl
import numpy as np
from numpy import pi
import pinocchio as pin
import matplotlib.pyplot as plt
import time


class MeshcatCallback(crocoddyl.CallbackAbstract):
    def __init__(self, viz_model):
        super().__init__()
        self.viz_model = viz_model

    def __call__(self, solver):
        # Display the current state (solver.xs contains state trajectories)
        for q in solver.xs:
            self.viz_model.display(q)


# Load the URDF model of the robot

robot = pin.RobotWrapper.BuildFromURDF("model/simple_grace.urdf", root_joint=pin.JointModelFreeFlyer())
# robot_data = robot_model.createData()

# geom_model = pin.buildGeomFromUrdf(robot_model, "simple_grace.urdf", "visual")
# geom_data = pin.GeometryData(geom_model)
# viz_model = pin.visualize.MeshcatVisualizer(robot_model, robot_data, geom_model, geom_data, viz)


state_dim = robot.nq + robot.nv
control_dim = robot.nv  # Control input is joint torques

# Define the robot's state (q: joint positions, v: joint velocities)
state = crocoddyl.StateMultibody(robot.model)

# Define the control (joint torques)
control = crocoddyl.ActivationModelQuad(control_dim)

# Get the state dimension

q_ref = np.array([-pi / 4, pi / 4, pi / 4,
                  pi / 4, pi / 4, pi / 4,
                  pi / 4, pi / 4, pi / 4,
                  -pi / 4, pi / 4, pi / 4])
qp_ref = np.zeros(robot.nv)

x_ref = np.concatenate([q_ref, qp_ref])  # q_ref: target position, v_ref: target velocity

meshcat_display = crocoddyl.MeshcatDisplay(robot)
# meshcat_display.display([q_ref])

# Create the Differential Dynamics model
# actuation = crocoddyl.ActuationModelFull(state)  # Assumes full control actuation
actuation = crocoddyl.ActuationModelFloatingBase(state)

# Define cost functions (e.g., tracking or torque regularization)
costs = crocoddyl.CostModelSum(state, actuation.nu)  # Composite cost

costs.addCost("tracking", crocoddyl.CostModelResidual(
    state, crocoddyl.ResidualModelFramePlacement(state, 'grace', desired_placement)), 1.0)

# Add a control regularization cost
u_ref = np.zeros(actuation.nu)  # Reference control (typically zero)

# Define the control residual
residual_control = crocoddyl.ResidualModelControl(state, u_ref)

# Create the control cost
residual_control_cost = crocoddyl.CostModelResidual(state, crocoddyl.ActivationModelQuad(actuation.nu),
                                                    residual_control)

costs.addCost("control_cost", residual_control_cost, weight=1e-5)

contact_model = crocoddyl.ContactModelMultiple(state)
# Add specific foot contacts
contact_points = ['LF-FOOT', 'LR-FOOT','RF-FOOT','RR-FOOT',]
for foot_name in contact_points:
    foot_frame_id = robot.model.getFrameId(foot_name)
    # Define the 3D contact model for the foot
    contact_model.addContact(crocoddyl.ContactModel3D(state, foot_frame_id))

differential_model = crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, contact_model, costs)

# Create the DifferentialActionModel for rigid body dynamics
dynamic_model = crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, costs)

# Time step for integration
dt = 1e-2
T = int(0.5 / dt)

# Wrap the differential dynamics with an integration scheme
running_model = crocoddyl.IntegratedActionModelEuler(dynamic_model, dt)

# Define the initial state
x0 = np.concatenate([q_ref, qp_ref])


# Create a shooting problem
problem = crocoddyl.ShootingProblem(x0, [running_model] * T, running_model)

# Define the solver
solver = crocoddyl.SolverDDP(problem)

# Set an initial guess for the trajectory
xs = [x0] * (T + 1)
us = [np.zeros(actuation.nu)] * T

# Solve the problem
solver.solve(xs, us, maxiter=100)
optimal_controls = solver.us

plt.plot(range(T), [solver.xs[i][0] for i in range(T)])
plt.show()

while True:
    # meshcat_display.display([q_ref])
    meshcat_display.displayFromSolver(solver, factor=0.2)
    time.sleep(1.0)

