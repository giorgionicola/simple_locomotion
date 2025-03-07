import crocoddyl as croc
import numpy as np
from numpy import pi
import pinocchio as pin
from pinocchio import SE3
import matplotlib.pyplot as plt
import time
from copy import deepcopy
from spider import SimpleSpiderGaitProblem

robot = pin.RobotWrapper.BuildFromURDF("model/simple_grace.urdf",
                                       root_joint=pin.JointModelFreeFlyer())

q_ref = np.array([
    0, 0, 0,
    0, 0, 0, 1,
    -pi / 4, 0, pi / 2,
    pi / 4, 0, pi / 2,
    pi / 4, 0, pi / 2,
    -pi / 4, 0, pi / 2,
    # 0, -pi/4, pi / 2,
    # 0 , -pi/4, pi / 2,
    # 0, -pi/4, pi / 2,
    # 0  , -pi/4, pi / 2,
])

x0 = np.concatenate([q_ref, np.zeros(robot.nv)])

problem = SimpleSpiderGaitProblem(rmodel=robot.model,
                                  right_front_foot_name='RF-FOOT',
                                  left_front_foot_name='LF-FOOT',
                                  right_back_foot_name='RR-FOOT',
                                  left_back_foot_name='LR-FOOT',
                                  body_name='grace',
                                  q_default=q_ref,
                                  control_type='cubic',
                                  impact_model='impulse'
                                  )

solvers = []
for i in range(1):
    solvers.append(croc.SolverFDDP(problem.create_walking_problem(x0=x0,
                                                                  step_height=0.1,
                                                                  step_length=0.25,
                                                                  step_knots=20,
                                                                  support_knots=2,
                                                                  timestep=0.02,))
                   )
    solvers[i].setCallbacks([croc.CallbackVerbose()])

    xs = [x0] * (solvers[i].problem.T + 1)
    us = solvers[i].problem.quasiStatic([x0] * solvers[i].problem.T)
    solvers[i].solve(xs, us, 100, False)

    x0 = solvers[i].xs[-1]



problem.plot_solution(solvers)

meshcat_display = croc.MeshcatDisplay(robot)
while True:
    for solv in solvers:
        # meshcat_display.display([q_ref])
        meshcat_display.displayFromSolver(solv, factor=2)
        time.sleep(1.0)
