import numpy as np
import pinocchio as pin
import crocoddyl as croc
from typing import List



class leg_gait:
    def __init__(self):
        self.pin_model = pin.RobotWrapper.BuildFromURDF("/home/kildall/python_projects/simple_locomotion/model/single_leg.urdf")
        self.rmodel = self.pin_model.model

        self.rdata = self.rmodel.createData()
        self.state = croc.StateMultibody(self.rmodel)
        self.actuation = croc.ActuationModelFull(self.state)

        self.nu = self.actuation.nu
        self.control = croc.ControlParametrizationModelPolyZero(self.nu)
        self.timestep = 0.02

        self.foot_id = self.rmodel.getFrameId('LF-FOOT')

    def create_foot_trajectory_problem(self, foot_traj: List[np.array]):
        models = []
        for foot_pos in foot_traj:
            cost_model = croc.CostModelSum(self.state, self.nu)

            frame_translation_residual = croc.ResidualModelFrameTranslation(self.state, self.foot_id, foot_pos, self.nu)
            foot_track = croc.CostModelResidual(self.state, frame_translation_residual)

            ctrlResidual = croc.ResidualModelControl(self.state, self.nu)
            ctrlReg = croc.CostModelResidual(self.state, ctrlResidual)

            cost_model.addCost("foot_track", foot_track, 1e6)
            cost_model.addCost("ctrl", ctrlReg, 1e-1)

            dmodel = croc.DifferentialActionModelFreeFwdDynamics(self.state, self.actuation, cost_model)

            models.append(croc.IntegratedActionModelRK(dmodel, self.control, croc.RKType.three, self.timestep))

        return models