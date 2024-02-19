""" A set of environments with a shared observation/state space for multi-task and transfer learning """
import random

import numpy as np
import copy

from manipulator_learning.sim.envs.manipulator_env_generic import ManipulatorEnv
from manipulator_learning.sim.envs.configs.panda_default import CONFIG as DEF_CONFIG
import manipulator_learning.sim.envs.rewards.generic as rew_tools
import manipulator_learning.sim.envs.rewards.lift as lift_rew
import manipulator_learning.sim.envs.rewards.reach as reach_rew
import manipulator_learning.sim.envs.rewards.stack as stack_rew
import manipulator_learning.sim.envs.rewards.bring as bring_rew
import manipulator_learning.sim.envs.rewards.move as move_rew
from rl_sandbox.planning.plan import get_condition
from rl_sandbox.utils import get_colors, get_vel_slices


class PandaPlayGeneric(ManipulatorEnv):

    def __init__(self,
                 task,
                 camera_in_state,
                 dense_reward,
                 block_num,
                 init_gripper_pose=((0.0, 0.5, 0.25), (np.pi, 0., 0.)),
                 # init_gripper_pose=((0.0, 0.5, 0.25), (np.pi, 5/12 * np.pi, 0.)),
                 init_gripper_random_lim=(.15, .15, .06, 0., 0., 0.),
                 obj_random_lim=((.15, .15, 0), (.15, .15, 0), (0, 0, 0), (0, 0, 0)),
                 obj_init_pos=((0, 0, 0), (0, 0, 0), (0.05, 0.0, -.0675), (-0.05, 0.0, -.0675)),  # for non-insert tray
                 obj_rgba=((0, 0, 1, 1), (0, 1, 0, 1), (0, 0, 1, .25), (0, 1, 0, .25)),
                 obj_urdf_names=('cube_blue_small', 'cube_blue_small', 'coaster', 'coaster'),
                 objs_in_state=(0, 1),
                 obj_targets=(2, 3),
                 # rel_pos_in_state=(0, 1, (0, 1), (0, 2), (1, 3)),
                 rel_pos_in_state=None,
                 tray_type=None,
                 state_data=('pos', 'obj_pos', 'grip_pos', 'goal_pos'),
                 max_real_time=18,
                 n_substeps=5,
                 image_width=160,
                 image_height=120,
                 limits_cause_failure=False,
                 failure_causes_done=False,
                 success_causes_done=False,
                 egl=True,
                 action_multiplier=0.1,
                 valid_t_dofs=(1, 1, 1),
                 valid_r_dofs=(1, 1, 1),
                 control_method='v',
                 gripper_control_method='bool_p',
                 pos_limits=((.85, -.35, .655), (1.15, -0.05, 0.8)),   # for non-insert tray
                 main_task='stack_01',  # suffix must be integer, series of integers (for certain tasks), or nothing
                 config_dict_mods=None,
                 force_pb_direct=False,
                 sparse_cond_time=0.5,
                 pos_ctrl_max_arm_force=50,
                 **kwargs):
        self.block_num = block_num
        objs_in_state = tuple(range(self.block_num))
        self.sparse_cond_start_time_main = None
        config_dict = copy.deepcopy(DEF_CONFIG)
        config_dict.update(dict(
            block_num=block_num,
            init_gripper_pose=init_gripper_pose,
            init_gripper_random_lim=init_gripper_random_lim,
            obj_random_lim=obj_random_lim,
            obj_init_pos=obj_init_pos,
            obj_rgba=obj_rgba,
            obj_urdf_names=obj_urdf_names,
            objs_in_state=objs_in_state,
            rel_pos_in_state=rel_pos_in_state,
            tray_type=tray_type,
            control_method=control_method,
            gripper_control_method=gripper_control_method,
            pos_limits=pos_limits,
            pos_ctrl_max_arm_force=pos_ctrl_max_arm_force
        ))

        if config_dict_mods is not None:
            config_dict.update(config_dict_mods)
        super().__init__(task, camera_in_state,
                         dense_reward, block_num, True, 'b', state_data, max_real_time=max_real_time,
                         n_substeps=n_substeps, gap_between_prev_pos=.2,
                         image_width=image_width, image_height=image_height,
                         failure_causes_done=failure_causes_done, success_causes_done=success_causes_done,
                         egl=egl,
                         control_frame='b', action_multiplier=action_multiplier,
                         valid_t_dofs=valid_t_dofs, valid_r_dofs=valid_r_dofs,
                         new_env_with_fixed_depth=True, config_dict=config_dict,
                         generate_spaces=True, vel_ref_frame='b', force_pb_direct=force_pb_direct, **kwargs)
        self.sparse_cond_time = sparse_cond_time   # time to "hold" conditions for triggering sparse reward
        self.sparse_cond_start_time = None
        self.limits_cause_failure = limits_cause_failure
        self.done_success_reward = 100  # hard coded for now, may not work
        self.done_failure_reward = -5  # hard coded for now, may not work


        # assert len(objs_in_state) == len(obj_targets), "Number of objects in states must equal number of objects" \
        #                                                "acting as target positions, got %s state objects and %s" \
        #                                                "target objects" % (objs_in_state, obj_targets)
        # self.obj_targets = obj_targets  # indices of objects that act as targets for objs in state

        # for defining specific task reward
        # suffix must be integer or series of integers (for stack)
        und_loc = main_task.rfind('_')
        if und_loc > -1:
            self.task_suffix = [int(suf_char) for suf_char in list(main_task[und_loc + 1:])]
            self.main_task = main_task


        # initial setting for unstack_stack -- can be modified for eval
        if self.main_task in ('unstack_stack', 'unstack_move_obj', 'unstack_lift', 'unstack_stack_env_only') and \
                hasattr(self.env, 'green_on_blue'):
            self.env.green_on_blue = True
            self._cube_rot_fix = True

    def reset_episode_success_data(self):
        """
        Call to reset latched_task_successes and all_task_sparse_timers properly.
        """
        if hasattr(self, "_latched_task_successes"):
            for task in self._latched_task_successes:
                self._latched_task_successes[task] = False
        if hasattr(self, "all_task_sparse_timers"):
            for task in self.all_task_sparse_timers:
                self.all_task_sparse_timers[task] = None

    def get_task_successes(self, tasks, task, observation, action, env_info):
        """
        Get success eval for list of tasks.

        Current options for tasks:
            - open_action (includes sparse reach and low velocity as well)
            - close_action (includes sparse reach and low velocity as well)
            - stack_0, stack_1
            - insert, insert_0, insert_1
            - bring, bring_0, bring_1
            - lift_0, lift_1
            - reach_0, reach_1
            - move_obj_0, move_obj_1
            - pick_and_place_0
            - unstack_stack_0
            - unstack_0
        """

        successes = []
        if not hasattr(self, "_latched_task_successes") or self.ep_timesteps <= 1:
            self._latched_task_successes = dict()

        if not hasattr(self, "all_task_sparse_timers"):
            self.all_task_sparse_timers = dict()
        block_num = env_info["block_num"]
        arm_vel = env_info['vel'][:3]
        table_height, ee_pos, task_obj_indices, task_obj_poss, pbc, task_obj_pb_ids, arm_pb_id, table_pb_id = \
            self._get_reward_state_info(self.main_task)
        for task in tasks:
            suc = []

            if task not in self.all_task_sparse_timers.keys():
                self.all_task_sparse_timers[task] = None

            main_task = task
            if main_task in ("openGripper", "closeGripper"):
                reach = any(reach_rew.reach_sparse(task_obj_pos, ee_pos, .1) for task_obj_pos in task_obj_poss)
                if task == 'openGripper':
                    open_or_close = True if action[-1] < 0 else 0
                else:
                    open_or_close = True if action[-1] > 0 else 0
                arm_vel_norm = np.linalg.norm(arm_vel)
                suc_cur_timestep = bool(reach and open_or_close and arm_vel_norm < .08)

            elif main_task == 'stack':
                suc_cur_timestep = stack_rew.stack_sparse_eval(pbc, task_obj_pb_ids, table_pb_id, task_obj_poss)

            elif main_task == 'lift':
                suc_height = .04
                for obj in task_obj_indices:
                    suc.append(lift_rew.lift_sparse_multiple(task_obj_poss[obj], suc_height, bottom_height=table_height))
                suc_cur_timestep = any(suc)

            elif main_task == 'reach':
                thresh = .02
                for obj in task_obj_indices:
                    suc.append(reach_rew.reach_sparse(task_obj_poss[obj], ee_pos, thresh))
                suc_cur_timestep = any(suc)

            elif main_task == 'move':
                obj_slice = get_vel_slices(block_num, 3)
                for obj in task_obj_indices:
                    obj_vel = env_info['obj_vel'][obj_slice[obj]]
                    obj_acc = env_info['obj_acc'][obj_slice[obj]]
                    suc.append(move_rew.move_sparse(obj_vel, obj_acc))
                suc_cur_timestep = any(suc)

            if main_task == 'move':
                sparse_cond_time_override = 1.0
            else:
                sparse_cond_time_override = self.sparse_cond_time

            suc, self.all_task_sparse_timers[task] = rew_tools.hold_timer(
                suc_cur_timestep, self.ep_timesteps, self.real_t_per_ts, sparse_cond_time_override, self.all_task_sparse_timers[task])
            if task in self._latched_task_successes:
                self._latched_task_successes[task] = suc or self._latched_task_successes[task]
            else:
                self._latched_task_successes[task] = suc

            successes.append(self._latched_task_successes[task])

        return successes

    def _get_reward_state_info(self, task):
        und_loc = task.rfind('_')
        task_suffix = [int(suf_char) for suf_char in list(task[und_loc + 1:])]

        if self.env.tray_type is not None:
            table_height = .665
        else:
            table_height = .645
        ee_pos = np.array(rew_tools.get_world_ee_pose(self.env)[:3])
        obj_poss = []
        for obj_id in self.env._obj_ids:
            obj_poss.append(rew_tools.get_world_obj_pose(self.env, obj_id)[:3])
        obj_poss = np.array(obj_poss)
        task_obj_indices = task_suffix
        task_obj_poss = obj_poss[task_obj_indices]

        # get pb ids of objects, needed for some rewards
        pbc = self.env._pb_client
        task_obj_pb_ids = np.array(self.env._obj_ids)[task_obj_indices]
        arm_pb_id = self.env.gripper.body_id
        table_pb_id = self.env.table if self.env.tray_type is None else self.env.tray

        # return table_height, ee_pos, task_obj_indices, task_obj_poss, target_obj_poss, pbc, task_obj_pb_ids, arm_pb_id, table_pb_id
        return table_height, ee_pos, task_obj_indices, task_obj_poss, pbc, task_obj_pb_ids, arm_pb_id, table_pb_id

    def _calculate_reward_and_done(self, action, task, base_block):
        reward = 0
        main_task_state = False
        effect = {}
        task_state = []

        # NOTE: for eval, since eval use deterministic action, include all suffix, eg, stack_012
        if isinstance(task, list):
            main_task_no_suff = self.main_task[:self.main_task.rfind("_")]
            if main_task_no_suff == "stack":
                table_height, ee_pos, task_obj_indices, task_obj_poss, pbc, task_obj_pb_ids, arm_pb_id, table_pb_id = \
                    self._get_reward_state_info(self.main_task)

                main_suc = stack_rew.stack_sparse_eval(pbc, task_obj_pb_ids, table_pb_id, task_obj_poss)
                main_task_state, self.sparse_cond_start_time_main = rew_tools.hold_timer(
                    main_suc, self.ep_timesteps, self.real_t_per_ts, self.sparse_cond_time, self.sparse_cond_start_time_main)

                for t in task:
                    idx = t.rfind("_")
                    sub_suffix = [int(suf_char) for suf_char in list(t[idx + 1:])]
                    sub_task = t[:idx]
                    if sub_task == "stack":
                        sub_obj_pb_ids = task_obj_pb_ids[sub_suffix]
                        subtask_obj_poss = task_obj_poss[sub_suffix]
                        sub_stack_suc = stack_rew.stack_sparse(pbc, sub_obj_pb_ids, subtask_obj_poss)
                        subtask_state, self.sparse_cond_start_time_main = rew_tools.hold_timer(
                            sub_stack_suc, self.ep_timesteps, self.real_t_per_ts, self.sparse_cond_time,
                            self.sparse_cond_start_time_main)

                        effect[t] = subtask_state
                    else:
                        effect[t] = False

        # NOTE: for training process, since only use target suffix, eg, stack_0
        else:
            if self.ep_timesteps == 0:
                return 0, False, False, {}, None
            if self.ep_timesteps == 1:
                und_loc = self.main_task.rfind('_')
                suffix = [int(suf_char) for suf_char in list(self.main_task[und_loc + 1:])]
                base_block = suffix[-1]

            curr_task = task[:task.rfind("_")]
            idx = task.rfind("_")
            curr_block = [int(suf_char) for suf_char in list(task[idx + 1:])]
            curr_condition = get_condition(self, curr_block, base_block, action)
            block, eff = next(iter(curr_condition.items()))
            all_suc = False

            if curr_task == 'openGripper':
                all_suc = True if action[-1] < 0 else 0
                self.sparse_cond_time = 0.2

            elif curr_task == 'closeGripper':
                all_suc = True if action[-1] > 0 else 0
                self.sparse_cond_time = 0.2

            elif curr_task == 'reach':
                if eff["on_table"] == True and eff["graspable"] == True:
                    all_suc = True
                    self.sparse_cond_time = 0.1

            elif curr_task == 'lift':
                if eff["on_table"] == False and eff["open_gripper"] == False and eff["lifted"] == True:
                    all_suc = True
                    self.sparse_cond_time = 0.3

            elif curr_task == 'move':
                if eff["open_gripper"] == False and eff["lifted"] == True and eff["moving"] == True:
                    all_suc = True
                    self.sparse_cond_time = 0.2

            elif curr_task == 'stack':
                if eff["on_table"] == False and eff["open_gripper"] == True and eff["on_top"] == True:
                    all_suc = True
                    self.sparse_cond_time = 0.5

            task_state, self.sparse_cond_start_time = rew_tools.hold_timer(
                all_suc, self.ep_timesteps, self.real_t_per_ts, self.sparse_cond_time, self.sparse_cond_start_time)
            if task_state:
                effect = curr_condition

            # NOTE: during training, when all block are stacked, update the network
            main_task_no_suff = self.main_task[:self.main_task.rfind("_")]
            if main_task_no_suff == "stack":
                self.sparse_cond_time = 0.5
                table_height, ee_pos, task_obj_indices, task_obj_poss, pbc, task_obj_pb_ids, arm_pb_id, table_pb_id = \
                    self._get_reward_state_info(self.main_task)
                suc = stack_rew.stack_sparse_eval(pbc, task_obj_pb_ids, table_pb_id, task_obj_poss)
                main_task_state, self.sparse_cond_start_time_main = rew_tools.hold_timer(
                    suc, self.ep_timesteps, self.real_t_per_ts, self.sparse_cond_time, self.sparse_cond_start_time_main)

        return reward, task_state, main_task_state, effect, base_block


class PandaPlayXYZState(PandaPlayGeneric):
    # obs space is 59, act space is 4
    # n_substeps of 5 plus action multiplier of .002 means action mag of 1.0 moves desired position 2mm*5 = 1cm
    def __init__(self, max_real_time=18, n_substeps=5, dense_reward=True, action_multiplier=0.002,
                 main_task='stack_01', force_pb_direct=True, **kwargs):
        super().__init__('None', False, dense_reward, max_real_time=max_real_time, n_substeps=n_substeps,
                         action_multiplier=action_multiplier,
                         state_data=('pos', 'vel', 'grip_pos', 'prev_grip_pos', 'obj_pos', 'obj_rot', 'obj_vel',
                                     'obj_rot_vel', 'force_torque'),
                         valid_t_dofs=(1, 1, 1), valid_r_dofs=(0, 0, 0), control_method='dp',
                         main_task=main_task, force_pb_direct=force_pb_direct,
                         **kwargs)



class PandaPlayInsertTrayXYZState(PandaPlayGeneric):
    # obs space is 59, act space is 4
    # obs indices:
    #   - pos:               0:3
    #   - vel:               3:6
    #   - grip_pos:          6:8
    #   - prev_grip_pos:    8:12
    #   - obj_pos:         12:26
    #   - obj_vel:         26:38
    #   - obj_rel_pos:     38:53
    #   - force_torque:    53:59
    # n_substeps of 5 plus action multiplier of .002 means action mag of 1.0 moves desired position 2mm*5 = 1cm
    def __init__(self, max_real_time=18, n_substeps=5, dense_reward=True, action_multiplier=0.002,
                 main_task='insert', force_pb_direct=True, **kwargs):
                 # main_task='unstack_stack_0', force_pb_direct=True, **kwargs):
        self.block_num = kwargs.pop("block_num")
        super().__init__('None', False, dense_reward, n_substeps=n_substeps,
                         action_multiplier=action_multiplier,
                         state_data=('pos', 'vel', 'grip_pos', 'prev_grip_pos', 'obj_pos', 'obj_rot', 'obj_vel',
                                     'obj_rot_vel', 'force_torque'),
                         block_num=self.block_num,
                         max_real_time=max_real_time,
                         tray_type='2_cube_insert',
                         obj_init_pos=tuple([(0, 0, 0) for _ in range(self.block_num)] + [(0.075, 0.0, -.055), (-0.075, 0.0, -.055)]),
                         obj_rgba=get_colors(self.block_num),
                         valid_t_dofs=(1, 1, 1), valid_r_dofs=(0, 0, 0), control_method='dp',
                         pos_limits=((.85, -.35, .665), (1.15, -0.05, 0.8)),
                         main_task=main_task, force_pb_direct=force_pb_direct,
                         **kwargs)

        # hardcode the insertion locations, since they correspond to the urdf itself
        # self._tray_insert_poss_world = np.array([[1.075, -.2, .655], [.925, -.2, .655]])
        self.main_task = main_task

        # extra info for LFGP
        # self._extra_info_dict = dict(
        #     insert_goal_poss=np.array([[.075, .5, .135], [-.075, .5, .135]]),  # relative to robot base frame
        #     bring_goal_poss=np.array([[.075, .5, .145], [-.075, .5, .145]])
        # )


class PandaPlayInsertTrayDPGripXYZState(PandaPlayInsertTrayXYZState):
    def __init__(self, max_real_time=18, n_substeps=5, dense_reward=True, action_multiplier=0.002,
                 main_task='insert', **kwargs):
        super().__init__(max_real_time, n_substeps, dense_reward, action_multiplier, main_task,
                         gripper_control_method='dp', grip_multiplier=0.25)


class PandaPlayInsertTrayPlusPickPlaceXYZState(PandaPlayGeneric):
    # Same as PandaPlayInsertTrayXYZState, but with added Pick and Place main task
    #
    # obs space is 65, act space is 4
    # obs indices:
    #   - pos:               0:3
    #   - vel:               3:6
    #   - grip_pos:          6:8
    #   - prev_grip_pos:    8:12
    #   - obj_pos:         12:29    # sphere target has no rotation in state, so 7 + 7 + 3
    #   - obj_vel:         29:41    # sphere not included here
    #   - obj_rel_pos:     41:59    # sphere rel to blue block added here (at end)
    #   - force_torque:    59:65
    # n_substeps of 5 plus action multiplier of .002 means action mag of 1.0 moves desired position 2mm*5 = 1cm
    def __init__(self, max_real_time=18, n_substeps=5, dense_reward=True, action_multiplier=0.002,
                 main_task='pick_and_place_01', force_pb_direct=True, **kwargs):
        super().__init__('None', False, dense_reward, max_real_time=max_real_time, n_substeps=n_substeps,
                         action_multiplier=action_multiplier,
                         obj_random_lim=((.15, .15, 0), (.15, .15, 0), (.15, .15, .1), (0, 0, 0), (0, 0, 0)),
                         # for non-insert tray
                         obj_rgba=((0, 0, 1, 1), (0, 1, 0, 1), (.5, .8, .95, .75), (0, 0, 1, .25), (0, 1, 0, .25)),
                         obj_urdf_names=('cube_blue_small', 'cube_blue_small', 'sphere_no_col_fit_small_cube_bigger', 'coaster', 'coaster'),
                         objs_in_state=(0, 1, 2),
                         objs_no_rot_no_vel=(2,),
                         obj_targets=({'pick_and_place': 2, 'bring': 3}, 4, -1),
                         rel_pos_in_state=(0, 1, (0, 1), (0, 3), (1, 4), (0, 2)),
                         state_data=('pos', 'vel', 'grip_pos', 'prev_grip_pos', 'obj_pos', 'obj_rot', 'obj_vel',
                                     'obj_rot_vel', 'force_torque'),
                         tray_type='2_cube_insert',
                         obj_init_pos=((0, 0, 0), (0, 0, 0), (0, 0, .04), (0.075, 0.0, -.055), (-0.075, 0.0, -.055)), # for insert tray
                         valid_t_dofs=(1, 1, 1), valid_r_dofs=(0, 0, 0), control_method='dp',
                         pos_limits=((.85, -.35, .665), (1.15, -0.05, 0.8)),  # for insert tray
                         main_task=main_task, force_pb_direct=force_pb_direct,
                         **kwargs)

        # hardcode the insertion locations, since they correspond to the urdf itself
        self._tray_insert_poss_world = np.array([[1.075, -.2, .655], [.925, -.2, .655]])

        # extra info for LFGP
        self._extra_info_dict = dict(
            insert_goal_poss=np.array([[.075, .5, .135], [-.075, .5, .135]]),  # relative to robot base frame
            bring_goal_poss=np.array([[.075, .5, .145], [-.075, .5, .145]])
        )

    def _get_reward_state_info(self, task_suffix=None):
        if task_suffix is None:
            task_suffix = self.task_suffix

        if self.env.tray_type is not None:
            table_height = .665
        else:
            table_height = .6247

        ee_pos = np.array(rew_tools.get_world_ee_pose(self.env)[:3])
        obj_poss = []
        for obj_id in self.env._obj_ids:
            obj_poss.append(rew_tools.get_world_obj_pose(self.env, obj_id)[:3])
        obj_poss = np.array(obj_poss)

        # sort between obj pos and task obj pos, "task" objects are the ones that directly contribute to reward
        if task_suffix is not None:
            task_obj_indices = np.array(task_suffix)
        else:
            task_obj_indices = np.array(self.env.objs_in_state)
            task_obj_indices = task_obj_indices[:-1]  # remove pick and place target from task_obj_indices

        task_obj_poss = obj_poss[task_obj_indices]

        # target objects
        obj_targets = []
        for target in self.obj_targets:
            if type(target) == dict:
                if self.main_task in target.keys():
                    obj_targets.append(target[self.main_task])
                else:
                    obj_targets.append(0)
            elif type(target) == int and target > -1:
                obj_targets.append(target)

        obj_targets_array = np.array(obj_targets)

        target_obj_poss = obj_poss[obj_targets_array[task_obj_indices]]

        # get pb ids of objects, needed for some rewards
        pbc = self.env._pb_client
        task_obj_pb_ids = np.array(self.env._obj_ids)[task_obj_indices]
        arm_pb_id = self.env.gripper.body_id
        table_pb_id = self.env.table if self.env.tray_type is None else self.env.tray

        return table_height, ee_pos, task_obj_indices, task_obj_poss, target_obj_poss, pbc, task_obj_pb_ids, arm_pb_id, table_pb_id
