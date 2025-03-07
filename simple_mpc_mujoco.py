import time
import mujoco
import mujoco.viewer
import numpy as np
from numpy import pi
import crocoddyl as croc
import pinocchio as pin
from pinocchio import SE3
import matplotlib.pyplot as plt
import time

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
])

gait_problem = SimpleSpiderGaitProblem(rmodel=robot.model,
                                  right_front_foot_name='RF-FOOT',
                                  left_front_foot_name='LF-FOOT',
                                  right_back_foot_name='RR-FOOT',
                                  left_back_foot_name='LR-FOOT',
                                  body_name='grace',
                                  q_default=q_ref,
                                  control_type='cubic',
                                  impact_model='impulse',
                                  timestep=0.02
                                  )


model = mujoco.MjModel.from_xml_path('../model/simple_grace.xml')
data = mujoco.MjData(model)

data.qpos[7:] = q_ref[7:]
data.ctrl = q_ref[7:]

viewer = mujoco.viewer.launch_passive(model, data)
t = 0


for _ in range(100):
    time.sleep(0.001)
    mujoco.mj_step(model, data)
    viewer.sync()

x0 = np.concatenate([data.qpos, data.qvel])
locomotion_models = gait_problem.create_locomotion_models(x0=x0,
                                                          step_height=0.1,
                                                          step_length=0.25,
                                                          step_knots=20,
                                                          support_knots=2, )


problem = croc.ShootingProblem(x0, locomotion_models[:-1], locomotion_models[-1])
solver = croc.SolverFDDP(problem)
xs = [x0] * (solver.problem.T + 1)
us = solver.problem.quasiStatic([x0] * solver.problem.T)

for t in range(10):
    solver.solve(xs, us, 10, False)

    data.ctrl = solver.xs[0][7:19]
    for _ in range(20):
        time.sleep(0.01)
        mujoco.mj_step(model, data)
        viewer.sync()

    x0 = np.concatenate([data.qpos, data.qvel])

