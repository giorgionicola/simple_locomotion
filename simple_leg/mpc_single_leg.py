import time
import mujoco
import mujoco.viewer
import numpy as np
from numpy import pi
import crocoddyl as croc
import pinocchio as pin
from pinocchio import SE3
import matplotlib
import matplotlib.pyplot as plt
from leg_problem import leg_gait

model = mujoco.MjModel.from_xml_path('/home/kildall/python_projects/simple_locomotion/model/single_leg.xml')
data = mujoco.MjData(model)
body_id = model.body(name='LF-FOOT').id

leg_gait_problem = leg_gait()
q_disturb = np.zeros(3)
q_start = np.array([0, -pi / 4, pi / 2 + pi / 4])

data.qpos = np.concatenate((q_disturb, q_start))
data.ctrl = np.concatenate((q_disturb, q_start, [0, 0, 0]))
viewer = mujoco.viewer.launch_passive(model, data)

control_dt = 0.02
control_time = []
control = []

sim_time = []
pos = []
foot_pos = []

dt = model.opt.timestep

pin.forwardKinematics(leg_gait_problem.rmodel, leg_gait_problem.rdata, q_start)
pin.updateFramePlacements(leg_gait_problem.rmodel, leg_gait_problem.rdata)
start_foot_pos = leg_gait_problem.rdata.oMf[leg_gait_problem.foot_id].translation

radius = 0.1
center_x = start_foot_pos[0] - radius
center_y = start_foot_pos[1]
center_z = start_foot_pos[2]

circle_fun = lambda phi, cx, cy, cz: np.array([cx + radius * np.cos(phi),
                                               cy + radius * np.sin(phi),
                                               cz])
omega = 2 * np.pi / 3
original_traj = [circle_fun(omega * j * 0.02, center_x, center_y, center_z) for j in range(1, int(5 / 0.02) + 1)]
traj = [foot_pos - data.qpos[:3] for foot_pos in original_traj]

models_to_solve = leg_gait_problem.create_foot_trajectory_problem(foot_traj=traj)
x0 = np.concatenate([data.qpos[3:], data.qvel[3:]])

problem = croc.ShootingProblem(x0, models_to_solve[:-1], models_to_solve[-1])
solver = croc.SolverFDDP(problem)
xs = [x0] * (solver.problem.T + 1)
us = solver.problem.quasiStatic([x0] * solver.problem.T)
solver.solve(xs, us, 500, False)

original_sol = solver.xs.copy()
solutions = []

j = 0
counter = 0
t = 0
action_duration = 1
while t < 5:
    sim_time.append(t)
    pos.append(data.qpos[3:].copy())
    foot_pos.append(data.xpos[body_id].copy())

    q_disturb = np.array([0.01 * np.sin(2 * np.pi * 0.1 * t),
                          0.01 * np.sin(2 * np.pi * 0.1 * t),
                          0.02 * np.sin(2 * np.pi *0.1 * t)])

    if t % control_dt < dt:
        if j == action_duration:
            if len(traj) - action_duration -1 < action_duration:
                break
            counter += 1
            x0 = np.concatenate([data.qpos[3:], data.qvel[3:]])
            traj = [p - q_disturb for p in original_traj[action_duration * counter - 1:]]
            models_to_solve = leg_gait_problem.create_foot_trajectory_problem(traj)
            xs = [x0] + [solver.xs[action_duration + i] for i in range(len(traj) - 1)]
            us = [solver.us[action_duration + i - 1] for i in range(len(traj) - 1)]

            problem = croc.ShootingProblem(x0, models_to_solve[:-1], models_to_solve[-1])
            solver = croc.SolverFDDP(problem)
            solver.solve(xs, us, 500, False)
            solutions.append(np.array(solver.xs, copy=True))
            j = 0

        mpc_control = solver.xs[j + 1]
        control_time.append(t)
        control.append(np.copy(mpc_control))
        j += 1

    data.ctrl = np.concatenate((q_disturb, mpc_control))
    mujoco.mj_step(model, data)
    viewer.sync()
    time.sleep(dt)
    t += dt

sol_to_print = [1, 20, 40]
for i in range(3):
    plt.figure()
    plt.plot(sim_time, [p[i] for p in pos], label='joint position')
    plt.plot(control_time, [c[i] for c in control], label=' joint control')
    plt.plot(control_time, [o[i] for o in original_sol[:len(control_time)]], label=' original sol')
    for s in sol_to_print:
        plt.plot(control_time[-solutions[s].shape[0]:], solutions[s][:, i], label=f'sol t={s}')
    plt.legend()

plt.figure()
plt.plot([pos[0] for pos in foot_pos], [pos[1] for pos in foot_pos])
plt.plot([pos[0] for pos in original_traj], [pos[1] for pos in original_traj])
plt.axis('equal')

plt.show()
