import crocoddyl as croc
import pinocchio as pin
import numpy as np
from scipy.spatial.transform import Rotation as R
from copy import deepcopy
from typing import List, Tuple
import matplotlib.pyplot as plt


class SimpleSpiderGaitProblem:
    LINEAR_CONTROL = 'linear'
    CUBIC_CONTROL = 'cubic'

    IMPULSE_MODEL = 'impulse'
    PSEUDO_IMPULSE_MODEL = 'pseudo_impulse'

    def __init__(self,
                 rmodel: pin.Model,
                 right_front_foot_name: str,
                 left_front_foot_name: str,
                 right_back_foot_name: str,
                 left_back_foot_name: str,
                 body_name: str,
                 q_default: np.ndarray,
                 timestep: float,
                 fwddyn: bool = True,
                 control_type: str = 'linear',
                 impact_model: str = 'impulse', ):

        self.debug = True

        if impact_model not in [self.PSEUDO_IMPULSE_MODEL, self.IMPULSE_MODEL]:
            raise RuntimeError('Unknown impulse model')
        self.impact_model = impact_model
        self.rmodel = rmodel
        self.rdata = self.rmodel.createData()
        self.state = croc.StateMultibody(self.rmodel)
        self.actuation = croc.ActuationModelFloatingBase(self.state)

        self.right_front_foot_name = right_front_foot_name
        self.left_front_foot_name = left_front_foot_name
        self.right_back_foot_name = right_back_foot_name
        self.left_back_foot_name = left_back_foot_name

        self.rf_foot_id = self.rmodel.getFrameId(right_front_foot_name)
        self.lf_foot_id = self.rmodel.getFrameId(left_front_foot_name)
        self.rb_foot_id = self.rmodel.getFrameId(right_back_foot_name)
        self.lb_foot_id = self.rmodel.getFrameId(left_back_foot_name)
        self.n_legs = 4
        self.feet_id = [self.rf_foot_id, self.lf_foot_id, self.rb_foot_id, self.lb_foot_id]

        self.center_body_id = self.rmodel.getFrameId(body_name)

        self.q0 = q_default
        self.rmodel.defaultState = np.concatenate([q_default, np.zeros(self.rmodel.nv)])
        self.timestep = timestep

        self.mu = 0.8
        self.Rsurf = np.eye(3)

        self.state_weights_moving = np.array([0, 0, 500] +
                                             [500, 500, 500] +
                                             [500, 500, 500] * 4 +
                                             [0.1] * 18
                                             )

        self.state_weights_standing = np.array([0, 0, 100] +
                                               [500, 500, 500] +
                                               [500, 500, 500] * 4 +
                                               [0.1] * 18
                                               )

        self.com_ref_traj = []
        self.foot_traj = {f'{self.rf_foot_id}': [],
                          f'{self.lf_foot_id}': [],
                          f'{self.rb_foot_id}': [],
                          f'{self.lb_foot_id}': [], }

        self.last_foot_pos = {f'{self.rf_foot_id}': None,
                              f'{self.lf_foot_id}': None,
                              f'{self.rb_foot_id}': None,
                              f'{self.lb_foot_id}': None, }

        self.last_com_pos = None

        self.com_template_traj = None
        self.foot_template_traj = None

        self._fwddyn = fwddyn
        if self._fwddyn:
            self.nu = self.actuation.nu
        else:
            self.nu = self.state.nv + 3 * 4

        if control_type not in [self.LINEAR_CONTROL, self.CUBIC_CONTROL]:
            raise RuntimeError('Unknown control type')
        else:
            self.control_type = control_type
            if self.control_type == 'linear':
                self.control = croc.ControlParametrizationModelPolyOne(self.nu)
            elif self.control_type == 'cubic':
                self.control = croc.ControlParametrizationModelPolyTwoRK(self.nu, croc.RKType.three)

        # self.gait_pattern = [{'swinging_feet': [self.rf_foot_id],
        #                       'support_feet': [[self.lf_foot_id,
        #                                         self.rb_foot_id,
        #                                         self.lb_foot_id]]},
        #                      {'swinging_feet': [self.lb_foot_id],
        #                       'support_feet': [self.lf_foot_id,
        #                                        self.rb_foot_id,
        #                                        self.rf_foot_id]},
        #                      {'swinging_feet': [self.rb_foot_id],
        #                       'support_feet': [[self.lf_foot_id,
        #                                         self.rf_foot_id,
        #                                         self.lb_foot_id]]},
        #                      {'swinging_feet': [self.lb_foot_id],
        #                       'support_feet': [self.lf_foot_id,
        #                                        self.rb_foot_id,
        #                                        self.rf_foot_id]},
        #                      ]

    def create_standing_problem(self,
                                x0: np.ndarray,
                                q_ref: np.ndarray,
                                timestep: float,
                                support_steps: int):

        q0 = x0[:self.rmodel.nq]
        pin.forwardKinematics(self.rmodel, self.rdata, q0)
        pin.updateFramePlacements(self.rmodel, self.rdata)

        rf_foot_pos = self.rdata.oMf[self.rf_foot_id].translation
        lf_foot_pos = self.rdata.oMf[self.lf_foot_id].translation
        rb_foot_pos = self.rdata.oMf[self.rb_foot_id].translation
        lb_foot_pos = self.rdata.oMf[self.lb_foot_id].translation
        com_pos0 = (rf_foot_pos + lf_foot_pos + rb_foot_pos + lb_foot_pos) / 4
        com_pos0[2] = pin.centerOfMass(self.rmodel, self.rdata, q0)[2].item()

        state_ref = np.zeros_like(x0)
        state_ref[: self.rmodel.nq] = q_ref

        standing_models = [self.create_support_model(timestep=timestep, state_ref=state_ref) for _ in
                           range(support_steps)]

        problem = croc.ShootingProblem(x0, standing_models[:-1], standing_models[-1])
        return problem

    def create_locomotion_models(self,
                                 x0: np.ndarray,
                                 step_height: float,
                                 step_length: float,
                                 step_knots: int,
                                 support_knots: int,
                                 ) -> List[croc.IntegratedActionModelAbstract]:

        q0 = x0[:self.rmodel.nq]
        pin.forwardKinematics(self.rmodel, self.rdata, q0)
        pin.updateFramePlacements(self.rmodel, self.rdata)

        rf_foot_pos = self.rdata.oMf[self.rf_foot_id].translation
        lf_foot_pos = self.rdata.oMf[self.lf_foot_id].translation
        rb_foot_pos = self.rdata.oMf[self.rb_foot_id].translation
        lb_foot_pos = self.rdata.oMf[self.lb_foot_id].translation

        self.last_com_pos = (rf_foot_pos + lf_foot_pos + rb_foot_pos + lb_foot_pos) / 4
        self.last_com_pos[2] = pin.centerOfMass(self.rmodel, self.rdata, q0)[2].item()
        for id in self.feet_id:
            self.last_foot_pos[f'{id}'] = self.rdata.oMf[id].translation

        self.foot_template_traj = self.compute_foot_template_trajectory(step_length=step_length,
                                                                        step_height=step_height,
                                                                        step_knots=step_knots)
        self.com_template_traj = self.compute_com_template_traj(step_length=step_length,
                                                                com_percentage=1 / 4,
                                                                step_knots=step_knots)

        locomotion_models = []

        locomotion_models += [self.create_support_model() for _ in range(support_knots)]
        locomotion_models += self.create_single_step_model(com_pos0=self.last_com_pos,  # self.com_ref_traj[-1],
                                                           swing_feet_ids=[self.rf_foot_id],
                                                           support_feet_ids=[self.lf_foot_id,
                                                                             self.rb_foot_id,
                                                                             self.lb_foot_id],
                                                           swing_feet_pos0=[rf_foot_pos],
                                                           step_knots=step_knots)

        locomotion_models += self.create_single_step_model(com_pos0=self.last_com_pos,
                                                           swing_feet_ids=[self.lb_foot_id],
                                                           support_feet_ids=[self.lf_foot_id,
                                                                             self.rb_foot_id,
                                                                             self.rf_foot_id],
                                                           swing_feet_pos0=[lb_foot_pos],
                                                           step_knots=step_knots)

        locomotion_models += [self.create_support_model() for _ in range(support_knots)]

        locomotion_models += self.create_single_step_model(com_pos0=self.last_com_pos,
                                                           swing_feet_ids=[self.lf_foot_id],
                                                           support_feet_ids=[self.rf_foot_id,
                                                                             self.rb_foot_id,
                                                                             self.lb_foot_id],
                                                           swing_feet_pos0=[lf_foot_pos],
                                                           step_knots=step_knots)

        locomotion_models += self.create_single_step_model(com_pos0=self.last_com_pos,
                                                           swing_feet_ids=[self.rb_foot_id],
                                                           support_feet_ids=[self.lf_foot_id,
                                                                             self.rf_foot_id,
                                                                             self.lb_foot_id],
                                                           swing_feet_pos0=[rb_foot_pos],
                                                           step_knots=step_knots)

        locomotion_models += [self.create_support_model() for _ in range(support_knots)]
        return locomotion_models

    def create_support_model(self,
                             state_ref: np.ndarray = None,
                             ) -> croc.IntegratedActionModelAbstract:

        if state_ref is None:
            state_ref = self.rmodel.defaultState

        cost_model = croc.CostModelSum(self.state, self.nu)
        contact_model = croc.ContactModelMultiple(self.state, self.nu)
        for id in [self.lf_foot_id, self.rf_foot_id, self.lb_foot_id, self.rb_foot_id]:
            contact_model_support_foot = croc.ContactModel3D(self.state,
                                                             id,
                                                             np.array([0.0, 0.0, 0.0]),
                                                             pin.LOCAL_WORLD_ALIGNED,
                                                             self.nu,
                                                             np.array([0.0, 50.0]))
            contact_model.addContact(self.rmodel.frames[id].name + "_contact", contact_model_support_foot)

            cone = croc.FrictionCone(self.Rsurf, self.mu, 4, False)
            cone_residual = croc.ResidualModelContactFrictionCone(self.state, id, cone, self.nu, self._fwddyn)
            cone_activation = croc.ActivationModelQuadraticBarrier(croc.ActivationBounds(cone.lb, cone.ub))
            friction_cone = croc.CostModelResidual(self.state, cone_activation, cone_residual)
            cost_model.addCost(self.rmodel.frames[id].name + "_frictionCone", friction_cone, 1e1)

        state_residual = croc.ResidualModelState(self.state, state_ref, self.nu)
        state_activation = croc.ActivationModelWeightedQuad(self.state_weights_standing)
        state_reg = croc.CostModelResidual(self.state, state_activation, state_residual)

        if self._fwddyn:
            ctrl_reg = croc.CostModelResidual(self.state, croc.ResidualModelControl(self.state, self.nu))
        else:
            ctrl_reg = croc.CostModelResidual(self.state,
                                              croc.ResidualModelJointEffort(self.state, self.actuation, self.nu))
        cost_model.addCost("state_reg", state_reg, 1)
        cost_model.addCost("ctrl_reg", ctrl_reg, 1)

        lb = np.concatenate([self.state.lb[1: self.state.nv + 1], self.state.lb[-self.state.nv:]])
        ub = np.concatenate([self.state.ub[1: self.state.nv + 1], self.state.ub[-self.state.nv:]])
        state_bounds_residual = croc.ResidualModelState(self.state, self.nu)
        state_bounds_activation = croc.ActivationModelQuadraticBarrier(croc.ActivationBounds(lb, ub))
        state_bounds = croc.CostModelResidual(self.state, state_bounds_activation, state_bounds_residual)
        cost_model.addCost("state_bounds", state_bounds, 1e3)

        # Creating the action model for the KKT dynamics with simpletic Euler
        # integration scheme
        if self._fwddyn:
            dmodel = croc.DifferentialActionModelContactFwdDynamics(self.state, self.actuation, contact_model,
                                                                    cost_model, 0.0, True)
        else:
            dmodel = croc.DifferentialActionModelContactInvDynamics(self.state, self.actuation, contact_model,
                                                                    cost_model)

        # Vedi quadruped.py per alternative più evolute
        if self.control_type == self.LINEAR_CONTROL:
            model = croc.IntegratedActionModelEuler(dmodel, self.control, self.timestep)
        elif self.control_type == self.CUBIC_CONTROL:
            model = croc.IntegratedActionModelRK(dmodel, self.control, croc.RKType.three, self.timestep)

        if self.debug:
            self.com_ref_traj.append(deepcopy(self.last_com_pos))
            for id in self.feet_id:
                self.foot_traj[f'{id}'].append(deepcopy(self.last_foot_pos[f'{id}']))
        return model

    def create_single_step_model(self,
                                 com_pos0: np.ndarray,
                                 swing_feet_ids: List[int],
                                 support_feet_ids: List[int],
                                 swing_feet_pos0: List[np.ndarray],
                                 step_knots: int) -> List[croc.IntegratedActionModelAbstract]:

        angle = 0
        swing_feet_traj = [deepcopy(self.foot_template_traj) for _ in swing_feet_ids]
        com_traj = deepcopy(self.com_template_traj)
        rotation = R.from_euler('xyz', [0, 0, angle], degrees=False).as_matrix()

        step_models = []
        for knot in range(step_knots):
            swing_feet_pos = [swing_feet_pos0[i] + rotation @ swing_feet_traj[i][knot] for i in
                              range(len(swing_feet_ids))]
            com_pos = com_pos0 + rotation @ self.com_template_traj[knot]

            step_models.append(self.create_swing_foot_model(swing_feet_ids=swing_feet_ids,
                                                            support_feet_ids=support_feet_ids,
                                                            swing_feet_pos=swing_feet_pos,
                                                            com_pos=com_pos))
        if self.debug:
            self.com_ref_traj += [com for com in com_traj]
            self.last_com_pos = self.com_ref_traj[-1].copy()
            for i, id in enumerate(swing_feet_ids):
                self.foot_traj[f'{id}'] += [deepcopy(swing_feet_traj[i][j]) for j in range(step_knots)]
                self.last_foot_pos[f'{id}'] = deepcopy(self.foot_traj[f'{id}'][-1])
            for i, id in enumerate(support_feet_ids):
                self.foot_traj[f'{id}'] += [deepcopy(self.last_foot_pos[f'{id}']) for _ in range(step_knots)]

        if self.impact_model == self.IMPULSE_MODEL:
            step_models.append(self.create_pseudo_impulse_model(support_foot_ids=support_feet_ids,
                                                                swing_feet_ids=swing_feet_ids,
                                                                swing_feet_final_pos=[foot_traj[-1] for foot_traj in
                                                                                      swing_feet_traj]))
        elif self.impact_model == self.PSEUDO_IMPULSE_MODEL:
            step_models.append(self.create_impulse_model(support_foot_ids=support_feet_ids,
                                                         swing_feet_ids=swing_feet_ids,
                                                         swing_feet_final_pos=[foot_traj[-1] for foot_traj in
                                                                               swing_feet_traj]))

        if self.debug:
            self.com_ref_traj.append(self.last_com_pos)
            for id in swing_feet_ids + support_feet_ids:
                self.foot_traj[f'{id}'].append(deepcopy(self.last_foot_pos[f'{id}']))

        return step_models

    def compute_foot_template_trajectory(self,
                                         step_height: float,
                                         step_length: float,
                                         step_knots: int):

        swing_feet_traj = [np.zeros(3) for _ in range(step_knots)]
        length_incr = step_length / step_knots

        if step_knots % 2 == 0:
            height_incr = step_height / (step_knots // 2 + 0.5)
            for knot in range(step_knots // 2):
                swing_feet_traj[knot] = np.array([length_incr * knot, 0, height_incr * knot])
                swing_feet_traj[-(knot + 1)] = np.array([step_length - (length_incr * knot), 0, height_incr * knot])
        else:
            height_incr = step_height / (step_knots // 2)
            for knot in range(step_knots // 2):
                swing_feet_traj[knot] = np.array([length_incr * knot, 0, height_incr * knot])
                swing_feet_traj[-(knot + 1)] = np.array([step_length - (length_incr * knot), 0, height_incr * knot])
            swing_feet_traj[step_knots // 2 + 1] = np.array([length_incr * (knot + 1), 0, height_incr * (knot + 1)])

        return swing_feet_traj

    def compute_com_template_traj(self, step_knots: int, step_length: float, com_percentage: float):
        length_incr = step_length / step_knots
        return np.array([[length_incr * knot * com_percentage, 0, 0] for knot in range(step_knots)])

    def compute_feet_trajectories(self,
                                  swing_feet_pos0: List[np.ndarray],
                                  com_pos0: np.ndarray,
                                  step_height: float,
                                  step_length: float,
                                  angle: float,
                                  step_knots: int,
                                  ) -> Tuple[List[np.ndarray], np.ndarray]:

        swing_feet_traj = [np.zeros((step_knots, 3)) for _ in range(len(swing_feet_pos0))]
        rotation = R.from_euler('xyz', [0, 0, angle], degrees=False).as_matrix()
        length_incr = step_length / step_knots
        com_percentage = len(swing_feet_pos0) / self.n_legs

        for i, foot_pos in enumerate(swing_feet_pos0):
            # We use as template foot trajectory a triangle
            if step_knots % 2 == 0:
                height_incr = step_height / (step_knots // 2 + 0.5)
                for knot in range(step_knots // 2):
                    swing_feet_traj[i][knot] = foot_pos + rotation @ np.array(
                        [length_incr * knot, 0, height_incr * knot])
                    swing_feet_traj[i][-(knot + 1)] = foot_pos + rotation @ np.array(
                        [step_length - (length_incr * knot), 0, height_incr * knot])
            else:
                height_incr = step_height / (step_knots // 2)
                for knot in range(step_knots // 2):
                    swing_feet_traj[i][knot] = foot_pos + np.array([length_incr * knot, 0, height_incr * knot])
                    swing_feet_traj[i][-(knot + 1)] = foot_pos + np.array(
                        [step_length - (length_incr * knot), 0, height_incr * knot])
                swing_feet_traj[i][step_knots // 2 + 1] = foot_pos + rotation @ np.array(
                    [length_incr * (knot + 1), 0, height_incr * (knot + 1)])

        com_traj = np.array(
            [com_pos0 + rotation @ np.array([length_incr * knot * com_percentage, 0, 0]) for knot in range(step_knots)])

        return swing_feet_traj, com_traj

    def create_swing_foot_model(self,
                                swing_feet_ids: List[int],
                                support_feet_ids: List[int],
                                swing_feet_pos: List[np.ndarray],
                                com_pos: np.ndarray,
                                ):

        cost_model = croc.CostModelSum(self.state, self.nu)
        contact_model = croc.ContactModelMultiple(self.state, self.nu)
        for id in support_feet_ids:
            contact_model_support_foot = croc.ContactModel3D(self.state,
                                                             id,
                                                             np.array([0.0, 0.0, 0.0]),
                                                             pin.LOCAL_WORLD_ALIGNED,
                                                             self.nu,
                                                             np.array([0.0, 50.0]))
            contact_model.addContact(self.rmodel.frames[id].name + "_contact", contact_model_support_foot)

            cone = croc.FrictionCone(self.Rsurf, self.mu, 4, False)
            cone_residual = croc.ResidualModelContactFrictionCone(self.state, id, cone, self.nu, self._fwddyn)
            cone_activation = croc.ActivationModelQuadraticBarrier(croc.ActivationBounds(cone.lb, cone.ub))
            friction_cone = croc.CostModelResidual(self.state, cone_activation, cone_residual)
            cost_model.addCost(self.rmodel.frames[id].name + "_frictionCone", friction_cone, 1e1)

        for id, foot_trajectory in zip(swing_feet_ids, swing_feet_pos):
            frame_translation_residual = croc.ResidualModelFrameTranslation(self.state, id, foot_trajectory, self.nu)
            foot_track = croc.CostModelResidual(self.state, frame_translation_residual)
            cost_model.addCost(self.rmodel.frames[id].name + "_footTrack", foot_track, 1e6)

        com_track_residual = croc.ResidualModelCoMPosition(self.state, com_pos, self.nu)
        com_track = croc.CostModelResidual(self.state, com_track_residual)
        cost_model.addCost("comTrack", com_track, 1e6)

        state_residual = croc.ResidualModelState(self.state, self.rmodel.defaultState, self.nu)
        state_activation = croc.ActivationModelWeightedQuad(self.state_weights_moving)
        state_reg = croc.CostModelResidual(self.state, state_activation, state_residual)
        if self._fwddyn:
            ctrl_reg = croc.CostModelResidual(self.state, croc.ResidualModelControl(self.state, self.nu))
        else:
            ctrl_reg = croc.CostModelResidual(self.state,
                                              croc.ResidualModelJointEffort(self.state, self.actuation, self.nu))
        cost_model.addCost("state_reg", state_reg, 1e1)
        cost_model.addCost("ctrl_reg", ctrl_reg, 1e-1)

        lb = np.concatenate([self.state.lb[1: self.state.nv + 1], self.state.lb[-self.state.nv:]])
        ub = np.concatenate([self.state.ub[1: self.state.nv + 1], self.state.ub[-self.state.nv:]])
        state_bounds_residual = croc.ResidualModelState(self.state, self.nu)
        state_bounds_activation = croc.ActivationModelQuadraticBarrier(croc.ActivationBounds(lb, ub))
        state_bounds = croc.CostModelResidual(self.state, state_bounds_activation, state_bounds_residual)
        cost_model.addCost("state_bounds", state_bounds, 1e3)

        # Creating the action model for the KKT dynamics with simpletic Euler
        # integration scheme
        if self._fwddyn:
            dmodel = croc.DifferentialActionModelContactFwdDynamics(self.state, self.actuation, contact_model,
                                                                    cost_model, 0.0, True)
        else:
            dmodel = croc.DifferentialActionModelContactInvDynamics(self.state, self.actuation, contact_model,
                                                                    cost_model)

        # Vedi quadruped.py per alternative più evolute
        if self.control_type == 'linear':
            model = croc.IntegratedActionModelEuler(dmodel, self.control, self.timestep)
        elif self.control_type == 'cubic':
            model = croc.IntegratedActionModelRK(dmodel, self.control, croc.RKType.three, self.timestep)
        else:
            raise RuntimeError('Unknown control type')

        return model

    def update_swing_foot_model(self,
                                swing_foot_model: croc.CostModelSum,
                                ):

        pass

    def update_support_model(self,
                             support_model):

        pass

    def create_impulse_model(self,
                             support_foot_ids: List[int],
                             swing_feet_ids: List[int],
                             swing_feet_final_pos: List[np.ndarray],
                             JMinvJt_damping: float = 1e-6,
                             r_coeff: float = 1e-6):
        """Action model for impulse models.

        An impulse model consists of describing the impulse dynamics against a set of
        contacts.
        :param support_foot_ids: Ids of the constrained feet
        :param swing_feet_final_pos: swinging foot task
        :return impulse action model
        """
        # Creating a 3D multi-contact model, and then including the supporting foot
        impulse_model = croc.ImpulseModelMultiple(self.state)
        for i in support_foot_ids:
            support_contact_model = croc.ImpulseModel3D(self.state, i, pin.LOCAL_WORLD_ALIGNED)
            impulse_model.addImpulse(self.rmodel.frames[i].name + "_impulse", support_contact_model)

        # Creating the cost model for a contact phase
        cost_model = croc.CostModelSum(self.state, 0)
        for id, foot_pos in zip(swing_feet_ids, swing_feet_final_pos):
            frame_translation_residual = croc.ResidualModelFrameTranslation(self.state, id, foot_pos, 0)
            foot_track = croc.CostModelResidual(self.state, frame_translation_residual)
            cost_model.addCost(self.rmodel.frames[id].name + "_footTrack", foot_track, 1e7)

        state_weights = np.array([1.0] * 6 + [10.0] * (self.rmodel.nv - 6) + [10.0] * self.rmodel.nv)
        state_residual = croc.ResidualModelState(self.state, self.rmodel.defaultState, 0)
        state_activation = croc.ActivationModelWeightedQuad(state_weights ** 2)
        state_reg = croc.CostModelResidual(self.state, state_activation, state_residual)
        cost_model.addCost("state_reg", state_reg, 1e1)

        # Creating the action model for the KKT dynamics with simpletic Euler
        # integration scheme
        model = croc.ActionModelImpulseFwdDynamics(self.state, impulse_model, cost_model)
        model.JMinvJt_damping = JMinvJt_damping
        model.r_coeff = r_coeff
        return model

    def create_pseudo_impulse_model(self,
                                    support_foot_ids: List[int],
                                    swing_feet_ids: List[int],
                                    swing_feet_final_pos: [List[np.ndarray]]) -> croc.IntegratedActionModelAbstract:

        """Action model for pseudo-impulse models.

        A pseudo-impulse model consists of adding high-penalty cost for the contact
        velocities.
        :param support_foot_ids: Ids of the constrained feet
        :param swing_feet_ids: swinging foot task
        :return pseudo-impulse differential action model
        """
        # Creating a 3D multi-contact model, and then including the supporting
        # foot
        if self._fwddyn:
            nu = self.actuation.nu
        else:
            nu = self.state.nv + 3 * len(support_foot_ids)
        contact_model = croc.ContactModelMultiple(self.state, nu)
        for id in support_foot_ids:
            supportContactModel = croc.ContactModel3D(self.state,
                                                      id,
                                                      np.array([0.0, 0.0, 0.0]),
                                                      pin.LOCAL_WORLD_ALIGNED,
                                                      nu,
                                                      np.array([0.0, 50.0]),
                                                      )
            contact_model.addContact(self.rmodel.frames[id].name + "_contact", supportContactModel)

        # Creating the cost model for a contact phase
        cost_model = croc.CostModelSum(self.state, nu)
        for id in support_foot_ids:
            cone = croc.FrictionCone(self.Rsurf, self.mu, 4, False)
            cone_residual = croc.ResidualModelContactFrictionCone(self.state, id, cone, nu, self._fwddyn)
            cone_activation = croc.ActivationModelQuadraticBarrier(croc.ActivationBounds(cone.lb, cone.ub))
            friction_cone = croc.CostModelResidual(self.state, cone_activation, cone_residual)
            cost_model.addCost(self.rmodel.frames[id].name + "_frictionCone", friction_cone, 1e1)

        for id, foot_pos in zip(swing_feet_ids, swing_feet_final_pos):
            frame_translation_residual = croc.ResidualModelFrameTranslation(self.state, id, foot_pos, nu)
            frame_velocity_residual = croc.ResidualModelFrameVelocity(self.state,
                                                                      id,
                                                                      pin.Motion.Zero(),
                                                                      pin.LOCAL_WORLD_ALIGNED,
                                                                      nu,
                                                                      )
            foot_track = croc.CostModelResidual(self.state, frame_translation_residual)
            impulse_foot_vel_cost = croc.CostModelResidual(self.state, frame_velocity_residual)
            cost_model.addCost(self.rmodel.frames[id].name + "_footTrack", foot_track, 1e7)
            cost_model.addCost(self.rmodel.frames[id].name + "_impulseVel", impulse_foot_vel_cost, 1e6, )

        # state_residual = croc.ResidualModelState(self.state, self.rmodel.defaultState, nu)
        # state_activation = croc.ActivationModelWeightedQuad(self.state_weights_moving)
        # state_reg = croc.CostModelResidual(self.state, state_activation, state_residual)
        if self._fwddyn:
            ctrl_residual = croc.ResidualModelControl(self.state, nu)
            ctrl_reg = croc.CostModelResidual(self.state, ctrl_residual)
        else:
            ctrl_residual = croc.ResidualModelJointEffort(self.state, self.actuation, nu)
            ctrl_reg = croc.CostModelResidual(self.state, ctrl_residual)
        # cost_model.addCost("state_reg", state_reg, 1e1)
        cost_model.addCost("ctrl_reg", ctrl_reg, 1e-3)

        # Creating the action model for the KKT dynamics with simpletic Euler
        # integration scheme
        if self._fwddyn:
            dmodel = croc.DifferentialActionModelContactFwdDynamics(self.state, self.actuation, contact_model,
                                                                    cost_model,
                                                                    0.0,
                                                                    True)
        else:
            dmodel = croc.DifferentialActionModelContactInvDynamics(self.state, self.actuation, contact_model,
                                                                    cost_model)
        if self.control_type == self.LINEAR_CONTROL:
            model = croc.IntegratedActionModelEuler(dmodel, 0.0)
        elif self.control_type == self.CUBIC_CONTROL:
            model = croc.IntegratedActionModelRK(dmodel, croc.RKType.three, 0.0)

        return model

    def plot_solution(self,
                      solvers: List[croc.SolverAbstract]) -> None:

        com_pos = []
        left_front_foot_pos = []
        left_back_foot_pos = []
        right_front_foot_pos = []
        right_back_foot_pos = []

        for solver in solvers:

            for xs in solver.xs:
                pin.framesForwardKinematics(self.rmodel, self.rdata, xs[:self.rmodel.nq])
                pin.updateFramePlacements(self.rmodel, self.rdata)

                com_pos.append(deepcopy(self.rdata.oMf[self.center_body_id].translation))
                left_front_foot_pos.append(deepcopy(self.rdata.oMf[self.lf_foot_id].translation))
                left_back_foot_pos.append(deepcopy(self.rdata.oMf[self.lb_foot_id].translation))
                right_front_foot_pos.append(deepcopy(self.rdata.oMf[self.rf_foot_id].translation))
                right_back_foot_pos.append(deepcopy(self.rdata.oMf[self.rb_foot_id].translation))

        fig0, ax0 = plt.subplots(nrows=3, ncols=1)
        fig1, ax1 = plt.subplots(nrows=3, ncols=1)
        fig2, ax2 = plt.subplots(nrows=3, ncols=1)
        fig3, ax3 = plt.subplots(nrows=3, ncols=1)
        fig4, ax4 = plt.subplots(nrows=3, ncols=1)
        axes = ['x', 'y', 'z']
        for i in range(3):
            ax0[i].plot([pos[i] for pos in com_pos])
            ax0[i].plot([pos[i] for pos in self.com_ref_traj])

            ax1[i].plot([pos[i] for pos in left_front_foot_pos])
            ax1[i].plot([pos[i] for pos in self.foot_traj[f'{self.lf_foot_id}']])

            ax2[i].plot([pos[i] for pos in left_back_foot_pos])
            ax2[i].plot([pos[i] for pos in self.foot_traj[f'{self.lb_foot_id}']])

            ax3[i].plot([pos[i] for pos in right_front_foot_pos])
            ax3[i].plot([pos[i] for pos in self.foot_traj[f'{self.rf_foot_id}']])

            ax4[i].plot([pos[i] for pos in right_back_foot_pos])
            ax4[i].plot([pos[i] for pos in self.foot_traj[f'{self.rb_foot_id}']])

            ax0[i].set_ylabel(f'{axes[i]}')
            ax1[i].set_ylabel(f'{axes[i]}')
            ax2[i].set_ylabel(f'{axes[i]}')
            ax3[i].set_ylabel(f'{axes[i]}')
            ax4[i].set_ylabel(f'{axes[i]}')

        fig0.suptitle('COM')
        fig1.suptitle('Left Front foot')
        fig2.suptitle('Left Back foot')
        fig3.suptitle('Right Front foot')
        fig4.suptitle('Right Front foot')
        plt.show()
