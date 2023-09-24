import numpy as np
import torch
from numpy.linalg import norm
from rl_sandbox.auxiliary_rewards.manipulator_learning.panda.lift_xyz_state import close, away #, open_action, close_action
from rl_sandbox.utils import get_pos_slice, get_vel_slices

require_reach_radius = .1  # could be None -- for bring tasks, only get reward if reached in this radius,
                           # for other tasks, give a general sparse reward for being in this radius
include_pos_limits_penalty = True
all_rew_include_dense_reach = True
all_rew_reach_multiplier = .1
tray_block_height = .145


include_unstack_in_aux = True

# for grip pos, 1 is fully open, 0 is fully closed, .52-.53 is closed on block


def dense_reach_bonus(task_rew, b_pos, arm_pos, max_reach_bonus=1.5, reach_thresh=.02,
                      reach_multiplier=all_rew_reach_multiplier):
    """ Convenience function for adding a conditional dense reach bonus to an aux task.

    If the task_rew is > 1, this indicates that the actual task is complete, and instead of giving a reach
    bonus, the max amount of reward given for a reach should be given (regardless of whether reach is satisfied).
    If it is < 1, a dense reach reward is given, and the actual task reward is given ONLY if the reach
    condition is satisfied. """
    if task_rew > 1:
        total_rew = task_rew + reach_multiplier * max_reach_bonus
    else:
        reach_rew = close(reach_thresh, b_pos, arm_pos, close_rew=max_reach_bonus)
        new_task_rew = task_rew * int(reach_rew > 1)
        total_rew = reach_multiplier * reach_rew + new_task_rew
    return total_rew


def pos_limits_penalty(e_info, action, penalty_mag=-.1):
    lim = e_info['at_limits'].flatten()
    penalty_quantities = penalty_mag * np.clip(np.concatenate([-action[:3], action[:3]]), 0, 1)
    penalties = lim * penalty_quantities

    return np.sum(penalties)


def torch_multi_close_or_open(open_bool, acts, obss):
    action_mag = torch.norm(acts[:, :3], dim=-1)
    if open_bool:
        open_or_close_rew = (acts[:, -1] < 0).float()
    else:
        open_or_close_rew = (acts[:, -1] > 0).float()
    total_rew = open_or_close_rew - .5 * action_mag
    return total_rew


# OPEN AND CLOSE AUX
def close_open_gen(open_bool, include_reach, obs_act_only=False):
    """ If open_bool is False, then this is a close reward instead. """
    def close_or_open_action(info, action, observation, torch_multi=False, **kwargs):
        if torch_multi:
            assert obs_act_only, "obs_act_only must be True for torch_multi"
            assert not include_reach, "include_reach must be False for torch_multi"
            return torch_multi_close_or_open(open_bool, action, observation)

        action_mag = norm(action[:3])
        if open_bool:
            open_or_close_rew = 1 if action[-1] < 0 else 0
        else:
            open_or_close_rew = 1 if action[-1] > 0 else 0
        total_rew = open_or_close_rew - .5 * action_mag
        e_info = info["infos"][-1]
        if include_reach and require_reach_radius is not None:
            block_num = e_info["block_num"]
            obj_pos_slice = get_pos_slice(block_num, 7)
            obj_pos = e_info["obj_pos"]
            arm_pos = e_info["pos"]
            curr_task = e_info["curr_task"]
            block = int(curr_task[curr_task.rfind("_") + 1:][0])
            base_block = int(e_info["base_block"])
            b0_pos = obj_pos[obj_pos_slice[block]][:3]
            b1_pos = obj_pos[obj_pos_slice[base_block]][:3]
            reach_dist = min(norm(b0_pos - arm_pos), norm(b1_pos - arm_pos))
            total_rew *= int(reach_dist <= require_reach_radius)

        if include_pos_limits_penalty and not obs_act_only:
            e_info = info["infos"][-1]
            total_rew += pos_limits_penalty(e_info, action)

        return total_rew

    if open_bool:
        close_or_open_action.__qualname__ = "openGripper" if include_reach else "pure_open"
    else:
        close_or_open_action.__qualname__ = "closeGripper" if include_reach else "pure_close"
    return close_or_open_action


openGripper = close_open_gen(True, True)
closeGripper = close_open_gen(False, True)
pure_open = close_open_gen(True, False, True)
pure_close = close_open_gen(False, False, True)


# LIFT AUX
def lift_gen(max_rew_height=.08, block_on_table_height=.145, close_shaping=True,
             include_reach_bonus=all_rew_include_dense_reach):
    def lift(info, action, **kwargs):
        e_info = info["infos"][-1]
        curr_task = e_info["curr_task"]
        block = int(curr_task[curr_task.rfind("_") + 1:][0])
        block_num = e_info["block_num"]
        obj_pos_slice = get_pos_slice(block_num, 7)
        b_pos = e_info['obj_pos'][obj_pos_slice[block]][:3]
        block_height = b_pos[2] - block_on_table_height
        arm_pos = e_info['pos']

        reach_close = close(.01, b_pos, arm_pos)

        if close_shaping:
            grip_pos = np.array(e_info['grip_pos'])

            close_bonus = .3 * int(reach_close > 1 and np.all(grip_pos < .7) and np.all(grip_pos > .4)
                                   and (action[2] > .1 or block_height > .03))
            # close_bonus = .3 * int(reach_close > 1 and np.all(grip_pos < .7) and np.all(grip_pos > .4))
            # if reach_close > 1:
            #     close_bonus = .1 * int(np.all(grip_pos < .7) and np.all(grip_pos > .4) and block_height > .005)
            # else:
            #     close_bonus = .02 * int(reach_close < 1 and np.all(grip_pos > .7))
        else:
            close_bonus = 0

        if np.all(grip_pos < .7) and np.all(grip_pos > .4):  # so block doesn't just push up walls
            if block_height > max_rew_height:
                lift_rew = 1.5
            elif block_height < .005:
                lift_rew = 0
            else:
                lift_rew = block_height / max_rew_height
        else:
            lift_rew = 0

        if require_reach_radius is not None:
            reach_dist = np.linalg.norm(arm_pos - b_pos)
            # lift_rew += all_rew_reach_multiplier * int(reach_dist <= require_reach_radius)
            lift_rew += all_rew_reach_multiplier * close(0.1, arm_pos, b_pos)

        if include_reach_bonus:
            lift_rew = dense_reach_bonus(lift_rew, b_pos, arm_pos, reach_thresh=.03)

        if include_pos_limits_penalty:
            lift_rew += pos_limits_penalty(e_info, action)

        if include_unstack_in_aux:
            # if too low, means block sank below tray, if too high, means it rode up edges, 0 if on tray, otherwise -.1
            base_block = int(e_info["base_block"])
            unstack_pen = -.1 * (1 - int(
                tray_block_height - .001 <= e_info['obj_pos'][obj_pos_slice[base_block]][:3][2] <= tray_block_height + .001))
        else:
            unstack_pen = 0

        return lift_rew + close_bonus + unstack_pen
    lift.__qualname__ = "lift"  # name MUST be lift_X to be pickleable
    return lift


lift = lift_gen()


# STACK AUX
# blocks are 4cm tall, so "goal" is having block 0 be 4cm above block 1
def stack_gen(block_on_table_height=.145, include_reach_bonus=True,
              include_lift_bonus=True, req_lift_height=.035, close_shaping=True, obs_act_only=True):
    def stack(info, action, observation, **kwargs):
        e_info = info["infos"][-1]
        curr_task = e_info["curr_task"]
        block = int(curr_task[curr_task.rfind("_") + 1:][0])
        block_num = e_info["block_num"]
        obj_pos_slice = get_pos_slice(block_num, 7)
        b_pos = e_info['obj_pos'][obj_pos_slice[block]][:3]
        base_block = int(e_info["base_block"])
        stack_above = e_info['obj_pos'][obj_pos_slice[base_block]][:3] + np.array([0, 0, .041])
        stack_close = close(0.005, b_pos, stack_above)
        block_height = b_pos[2] - block_on_table_height
        is_lifted = block_height > req_lift_height
        stack_close = int(is_lifted) * stack_close
        grip_pos = np.array(e_info['grip_pos'])
        block_vel_slice = get_vel_slices(block_num, 3)
        block_vel = e_info['obj_vel'][block_vel_slice[block]]
        block_vel_mag = np.linalg.norm(block_vel)

        stack_dist = np.linalg.norm(b_pos - stack_above)

        # see if "stacked" and if arm isn't close to give a dense "stacked" bonus
        arm_pos = e_info['pos']
        is_stacked_bonus = is_lifted * (block_vel_mag < .01)

        if include_reach_bonus:
            stack_close = dense_reach_bonus(stack_close, b_pos, arm_pos)

        close_bonus = 0
        open_bonus = 0
        away_bonus = 0
        if close_shaping:
            reach_close = close(.005, b_pos, arm_pos)
            close_bonus = .1 * int(reach_close > 1 and np.all(grip_pos < .7) and np.all(grip_pos > .4)
                                   and (action[2] > .1 or block_height > .03))

            # action[3] < 0 corresponds to open. gives positive only reward for eventually opening
            if stack_dist < .01:
                if action[3] < 0:
                    away_bonus = 0.5 * away(b_pos[2], arm_pos[2])
                    open_bonus = 1.5 * (-action[3] + 1) * int(block_height > .03) * is_stacked_bonus

        lift_bonus = 0
        if include_lift_bonus:
            if block_height > .005 and np.all(grip_pos < .7) and np.all(grip_pos > .4):
                lift_bonus = 2 * all_rew_reach_multiplier * np.clip(block_height / req_lift_height, 0, req_lift_height)
            else:
                lift_bonus = 0

        total_rew = stack_close + lift_bonus + close_bonus + open_bonus + away_bonus

        if require_reach_radius is not None:
            reach_dist = np.linalg.norm(arm_pos - b_pos)
            total_rew *= int(reach_dist <= require_reach_radius)

        if include_pos_limits_penalty and not obs_act_only:
            total_rew += pos_limits_penalty(e_info, action)

        # add a penalty for block to be stacked on not being on tray
        # unstack_rew = .1 * min(-stack_block_height / .04, 0)  # 0 if on tray, otherwise -.1
        # if too low, means block sank below tray, if too high, means it rode up edges, 0 if on tray, otherwise -.1
        if include_unstack_in_aux:
            unstack_pen = -.3 * (1 - int(
                tray_block_height - .001 <= e_info['obj_pos'][obj_pos_slice[base_block]][:3][2] <= tray_block_height + .001))
        else:
            unstack_pen = 0

        return total_rew + unstack_pen

    stack.__qualname__ = "stack"  # name MUST be stack_X to be pickleable
    return stack


stack = stack_gen()


# REACH AUX
def reach_gen():
    def reach(info, action, **kwargs):
        e_info = info["infos"][-1]
        curr_task = e_info["curr_task"]
        block = int(curr_task[curr_task.rfind("_") + 1:][0])
        block_num = e_info["block_num"]
        obj_pos_slice = get_pos_slice(block_num, 7)
        b_pos = e_info['obj_pos'][obj_pos_slice[block]][:3]
        close_rew = close(0.0, b_pos[:3], e_info['pos'])
        base_block = e_info["base_block"]
        if include_pos_limits_penalty:
            close_rew += pos_limits_penalty(e_info, action)

        # if too low, means block sank below tray, if too high, means it rode up edges, 0 if on tray, otherwise -.1
        if include_unstack_in_aux:
            unstack_pen = -.1 * (1 - int(
                tray_block_height - .001 <= e_info['obj_pos'][obj_pos_slice[base_block]][:3][2] <= tray_block_height + .001))
        else:
            unstack_pen = 0

        return close_rew + unstack_pen
    reach.__qualname__ = "reach"
    return reach


reach = reach_gen()


# MOVE AUX
def move_gen(require_reach=True, include_lift_bonus=True, block_on_table_height=.145,
                 include_reach_bonus=all_rew_include_dense_reach, acc_pen_mult=.1):
    def move(info, action, **kwargs):
        e_info = info["infos"][-1]
        curr_task = e_info["curr_task"]
        block = int(curr_task[curr_task.rfind("_") + 1:][0])
        block_num = e_info["block_num"]
        block_vel_slice = get_vel_slices(block_num, 3)
        obj_t_vel_mag = 5 * np.linalg.norm(e_info['obj_vel'][block_vel_slice[block]])  # since .3 is max, scale by 5
        obj_pos_slice = get_pos_slice(block_num, 7)
        b_pos = e_info['obj_pos'][obj_pos_slice[block]][:3]
        arm_pos = e_info['pos']
        base_block = e_info["base_block"]
        # is_close = np.linalg.norm(arm_pos - b_pos) < .02
        is_close = np.linalg.norm(arm_pos - b_pos) < .04

        if include_lift_bonus:
            block_height = b_pos[2] - block_on_table_height
            grip_pos = e_info['grip_pos']
            is_lifted = np.all(grip_pos < .7) and np.all(grip_pos > .4) and block_height > .005
            bonus = all_rew_reach_multiplier * int(is_lifted)
        else:
            bonus = 0

        if require_reach:  # without this could learn to pick up and drop
            obj_t_vel_mag *= int(is_close)

        if require_reach_radius is not None:
            reach_dist = np.linalg.norm(arm_pos - b_pos)
            obj_t_vel_mag += all_rew_reach_multiplier * int(reach_dist <= require_reach_radius)

        if include_reach_bonus:
            bonus += all_rew_reach_multiplier * close(.03, b_pos, arm_pos, close_rew=1.5)

        if include_pos_limits_penalty:
            obj_t_vel_mag += pos_limits_penalty(e_info, action)

        acc_pen = 0
        if acc_pen_mult is not None:
            obj_t_acc_mag = min(np.linalg.norm(e_info['obj_acc'][block_vel_slice[block]]), 1.5)
            obj_t_acc_mag *= int(is_close)
            acc_pen = acc_pen_mult * obj_t_acc_mag

        # if too low, means block sank below tray, if too high, means it rode up edges, 0 if on tray, otherwise -.1
        if include_unstack_in_aux:
            unstack_pen = -.1 * (1 - int(
                tray_block_height - .001 <= e_info['obj_pos'][obj_pos_slice[base_block]][:3][2] <= tray_block_height + .001))
        else:
            unstack_pen = 0

        return obj_t_vel_mag + bonus - acc_pen + unstack_pen
    move.__qualname__ = "move"  # name MUST be move_obj_X to be pickleable
    return move


move = move_gen()


# CLASSES
class AuxiliaryReward:
    def __init__(self, aux_rewards=(), include_main=True):
        self._aux_rewards = aux_rewards
        self._include_main = include_main

        # get a list of the aux rewards as strings
        ar_strs = []
        for ar in self._aux_rewards:
            ar_strs.append(ar.__qualname__)
        self._aux_rewards_str = ar_strs

    @property
    def num_auxiliary_rewards(self):
        return len(self._aux_rewards)

    def set_aux_rewards_str(self):
        """ For older loaded classes that don't call init. """
        ar_strs = []
        for ar in self._aux_rewards:
            ar_strs.append(ar.__qualname__)
        self._aux_rewards_str = ar_strs

    def reward(self, observation, action, reward, done, next_observation, info, eval_flag):
        observation = observation.reshape(-1)
        next_observation = next_observation.reshape(-1)
        reward_vector = []
        main_task = info["infos"][-1]["main_task"]
        if self._include_main:
            reward_vector.append(reward)
        if eval_flag is False:
            # for timestep 0
            if info["infos"][-1]["curr_task"] is None:
                info["infos"][-1]["curr_task"] = "openGripper_0"
                base_block = int(main_task[main_task.rfind("_") + 1:][-1])
                info["infos"][-1]["base_block"] = int(base_block)
                for task_reward in self._aux_rewards:
                    reward_vector.append(task_reward(
                        observation=observation, action=action, reward=reward, next_observation=next_observation,
                        done=done, info=info))
                info["infos"][-1]["curr_task"] = None
            else:
                base_block = info["infos"][-1]["base_block"]
                curr_task = info["infos"][-1]["curr_task"]
                curr_block = int(curr_task[curr_task.rfind("_") + 1:])
                # for a very rare scene that one block on top of another after reset
                # or at the beginning of new scheduling period, so done is false and haven't end up this simulation
                if base_block == curr_block:
                    base_block = int(main_task[main_task.rfind("_") + 1:][-1])
                    info["infos"][-1]["base_block"] = int(base_block)
                for task_reward in self._aux_rewards:
                    reward_vector.append(task_reward(
                        observation=observation, action=action, reward=reward, next_observation=next_observation,
                        done=done, info=info))
        else:
            env_info = info["infos"][-1]
            state_dict = env_info["effect"]
            reward_dict = {}
            for key, value in state_dict.items():
                info["infos"][-1]["curr_task"] = key
                if value is True:
                    for task_reward in self._aux_rewards:
                        reward_vector.append(task_reward(
                            observation=observation, action=action, reward=reward, next_observation=next_observation,
                            done=done, info=info))
                    break
                else:
                    reward_vector = []
                    for task_reward in self._aux_rewards:
                        reward_vector.append(task_reward(
                            observation=observation, action=action, reward=reward, next_observation=next_observation,
                            done=done, info=info))
                    reward_dict[key] = reward_vector
            if len(reward_dict) != 0:
                task = max(reward_dict.keys(), key=lambda k: sum(reward_dict[k]))
                reward_vector = reward_dict[task]
            else:
                reward_vector = reward_vector
        return np.array(reward_vector, dtype=np.float32)


# some convenience functions for generating each play class
aux_rewards_all = [openGripper, closeGripper]
# aux_rewards_all = []


# def get_aux_rewards(block, aux_rewards_added_str):
#     aux_rewards_added = []
#     if block is None:
#         blocks = list(range(num_blocks))
#     else:
#         blocks = [block]
#     for i in blocks:
#         aux_rewards_added.extend([globals()[f_str + '_' + str(i)] for f_str in aux_rewards_added_str])
#     return aux_rewards_added

def get_aux_rewards(aux_rewards_added_str):
    aux_rewards_added = []
    aux_rewards_added.extend([globals()[f_str] for f_str in aux_rewards_added_str])
    return aux_rewards_added

class PandaPlayXYZStateAuxiliaryReward(AuxiliaryReward):
    """ Play aux reward class set up to take main_task as argument. Should be of form:
        {main_task}_{block_index}, for tasks where a single block are either optional or mandatory.

        Examples:
            - all
            - stack_01  (stack must have 01 or 10)
            - insert  (both)
            - insert_0
            - lift_0 (lift must have 0 or 1)
    """
    def __init__(self, main_task, include_main=True, aux_rewards_all=aux_rewards_all):
        und_loc = main_task.rfind('_')
        if und_loc > -1:
            main_task_no_suf = main_task[:und_loc]
            block_suf = [int(suf_char) for suf_char in list(main_task[und_loc + 1:])]

            # hard-coded if argument is 01 or 10 for stack, only take 0 or 1
            block_suf = block_suf[0]
        else:
            main_task_no_suf = main_task
            block_suf = None

        if main_task_no_suf == 'all':
            aux_rewards_added = get_aux_rewards(block_suf, ['stack', 'insert', 'bring', 'lift', 'reach', 'move_obj'])
            aux_rewards_added.append(blocks_together)
        elif main_task_no_suf == 'stack':
            # print("Warning: Ensure that env is PandaPlayInsertTrayXYZState. If using "
            #       "PandaPlayInsertTrayPlusPickPlaceXYZState, main_task should be stack_pp_env_X.")
            aux_rewards_added = get_aux_rewards(['reach', 'lift', 'move', 'stack'])
            # aux_rewards_added.append(open_action)
        elif main_task_no_suf == 'stack_pp_env':
            aux_rewards_added = get_aux_rewards(block_suf, ['stack_pp_env', 'lift', 'reach', 'move_obj'])
        elif main_task_no_suf == 'insert':
            aux_rewards_added = get_aux_rewards(block_suf, ['insert', 'bring', 'lift', 'reach', 'move_obj'])
        elif main_task_no_suf == 'bring':
            aux_rewards_added = get_aux_rewards(block_suf, ['bring', 'lift', 'reach', 'move_obj'])
        elif main_task_no_suf == 'lift':
            # aux_rewards_added = get_aux_rewards(block_suf, ['lift', 'reach', 'move_obj'])
            aux_rewards_added = get_aux_rewards(block_suf, ['lift', 'reach'])
        elif main_task_no_suf == 'move_obj':
            aux_rewards_added = get_aux_rewards(block_suf, ['lift', 'reach', 'move_obj'])
        elif main_task_no_suf == 'reach':
            aux_rewards_added = get_aux_rewards(block_suf, ['reach'])
        elif main_task_no_suf == 'together':
            aux_rewards_added = get_aux_rewards(block_suf, ['lift', 'reach', 'move_obj'])
            aux_rewards_added.append(blocks_together)
        elif main_task_no_suf == 'bring_and_remove':
            aux_rewards_added = get_aux_rewards(block_suf, ['bring', 'lift', 'reach', 'move_obj'])
            aux_rewards_added += get_aux_rewards(1 - block_suf, ['lift', 'reach', 'move_obj'])
        elif main_task_no_suf == 'lift_open_close':
            aux_rewards_added = get_aux_rewards(block_suf, ['lift'])
        elif main_task_no_suf == 'stack_open_close':
            aux_rewards_added = get_aux_rewards(block_suf, ['stack'])
        elif main_task_no_suf == 'pick_and_place':
            aux_rewards_added = get_aux_rewards(block_suf, ['pick_and_place', 'lift', 'reach', 'move_obj'])
        elif main_task_no_suf == 'unstack_stack':
            aux_rewards_added = get_aux_rewards(block_suf, ['stack', 'unstack', 'lift', 'reach', 'move_obj'])
        elif main_task_no_suf == 'unstack_stack_env_only':
            aux_rewards_added = get_aux_rewards(block_suf, ['stack', 'lift', 'reach', 'move_obj'])
        elif main_task_no_suf == 'unstack_move_obj':
            aux_rewards_added = get_aux_rewards(block_suf, ['unstack', 'lift', 'reach', 'move_obj'])
        elif main_task_no_suf == 'unstack_lift':
            aux_rewards_added = get_aux_rewards(block_suf, ['unstack', 'lift', 'reach'])
        else:
            raise NotImplementedError("PandaPlayXYZStateAuxiliaryReward not implemented for main_task %s" % main_task)
        # if we want to make the main task the first task, switch this comment
        # super().__init__(aux_rewards=tuple(aux_rewards_added + aux_rewards_all), include_main=include_main)
        super().__init__(aux_rewards=tuple(aux_rewards_all + aux_rewards_added), include_main=include_main)
