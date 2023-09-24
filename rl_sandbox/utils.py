import _pickle as pickle
import numpy as np
import os
import timeit
import torch
import re
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import collections

import rl_sandbox.constants as c
import colorsys

class DummySummaryWriter():
    def add_scalar(self, arg_1, arg_2, arg_3):
        pass

    def add_scalars(self, arg_1, arg_2, arg_3):
        pass

    def add_text(self, arg_1, arg_2, arg_3):
        pass


def make_summary_writer(save_path, algo, cfg):
    summary_writer = DummySummaryWriter()
    cfg[c.ALGO] = algo
    if save_path is not None:
        time_tag = datetime.strftime(datetime.now(), "%m-%d-%y_%H_%M_%S")
        save_path = f"{save_path}/{time_tag}"
        os.makedirs(save_path, exist_ok=True)
        summary_writer = SummaryWriter(log_dir=f"{save_path}/tensorboard")
        pickle.dump(
            cfg,
            open(f'{save_path}/{algo}_experiment_setting.pkl', 'wb'))

    return summary_writer, save_path


def set_seed(seed=None):
    if seed is None:
        seed = np.random.randint(0, c.MAX_INT)

    np.random.seed(seed)
    torch.manual_seed(seed)


def get_block_num(task):
    num = task.rfind("_")
    return int(task[num + 1])


def get_main_task(task):
    num = task.rfind("_")
    return task[:num]


def get_traj_dict(traj):
    traj_dict = collections.OrderedDict()
    for i in traj:
        name_no_suff = get_main_task(i)
        if name_no_suff == "reach":
            assign_value = 0
        elif name_no_suff == "lift":
            assign_value = 1
        elif name_no_suff == "move":
            assign_value = 2
        else:
            assign_value = 3
        traj_dict.update({i: assign_value})
    return traj_dict


def get_task_list(task_no_suf):
    if task_no_suf == "stack":
        task_list = ["open_action", "close_action", "reach", "lift", "move", "stack"]
        return task_list


def get_pos_slice(num, length):
    slices = []
    for i in range(num):
        start_index = i * length
        end_index = start_index + length
        block_slice = slice(start_index, end_index)
        slices.append(block_slice)
    return slices


def get_vel_slices(num, length):
    slices = []
    for i in range(num):
        start_index = i * length * 2
        end_index = start_index + length
        block_slice = slice(start_index, end_index)
        slices.append(block_slice)
    return slices


def get_colors(num_colors, alpha=1):
    colors = []
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i / 360.
        lightness = (50 + np.random.rand() * 10) / 100.
        saturation = (90 + np.random.rand() * 10) / 100.
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        rgba = tuple(list(rgb) + [alpha])
        colors.append(rgba)

    colors.append(tuple(list(colors[0][:3]) + [0.25]))
    colors.append(tuple(list(colors[1][:3]) + [0.25]))

    return colors


def idx(objs_in_state, obj_pb_ids, target_block_pbid):
    mapping = dict(zip(obj_pb_ids, objs_in_state))
    return [mapping[i] for i in target_block_pbid]


def get_target(env):
    pbc = env._pb_client
    obj_pb_ids = env._obj_ids[:-2]
    table_pb_id = env.tray
    objs_in_state = list(env.objs_in_state)
    arm_pb_id = env.gripper.body_id
    o2t_contacts = []
    o2ee_contact = []
    target_block_pbid = []
    for i in range(0, len(obj_pb_ids) - 1):
        o2ee_contact = len(pbc.getContactPoints(obj_pb_ids[i], arm_pb_id)) > 0
        if o2ee_contact is True:
            target_block_pbid.append(obj_pb_ids[i])
        else:
            o2t_contacts = (len(pbc.getContactPoints(obj_pb_ids[i], table_pb_id)) > 0)
            if o2t_contacts is True:
                target_block_pbid.append(obj_pb_ids[i])
    target_block = idx(objs_in_state, obj_pb_ids, target_block_pbid)
    return target_block


def generate_substack(block_num, base_block, completed_blocks):
    subtasks = []
    for block in range(block_num):
        if block != base_block and block not in completed_blocks:
            subtask = [block, base_block]
            subtasks.append("stack_" + "".join(map(str, subtask)))
    return subtasks


def solve_stack(main_task, curr_subtasks, block_on_table, base_block):
    total_blocks = set(int(suf_char) for suf_char in main_task[main_task.rfind("_") + 1:])

    if curr_subtasks is None:
        completed_blocks = set()
    else:
        block_on_table.append(base_block)
        completed_blocks = total_blocks - set(block_on_table)

    block_num = max(total_blocks) + 1
    next_subtasks = generate_substack(block_num, base_block, completed_blocks)
    return next_subtasks


class EpochSummary:
    def __init__(self, default_key_length=10, padding=11):
        self._key_length = default_key_length
        self._padding = padding
        self._summary = dict()
        self._epoch = 0
        self._init_tic = timeit.default_timer()

    def log(self, key, value, track_std=True, track_min_max=True, axis=None):
        self._key_length = max(self._key_length, len(key))
        self._summary.setdefault(key, {
            c.LOG_SETTING: {
                c.STANDARD_DEVIATION: track_std,
                c.MIN_MAX: track_min_max,
                c.AXIS: axis,
            },
            c.CONTENT: []
        })
        self._summary[key][c.CONTENT].append(value)

    def new_epoch(self):
        self._epoch += 1
        self._summary.clear()
        self._curr_tic = timeit.default_timer()

    def print_summary(self):
        toc = timeit.default_timer()
        key_length = self._key_length + self._padding
        print("=" * 100)
        print(f"Epoch: {self._epoch}")
        print(f"Epoch Time Spent: {toc - self._curr_tic}")
        print(f"Total Time Spent: {toc - self._init_tic}")
        print("=" * 100)
        print('|'.join(str(x).ljust(key_length) for x in ("Key", "Content")))
        print("-" * 100)

        # temp fix for scheduler trajs that are not always same length
        if 'update_info/scheduler_traj' in self._summary:
            del self._summary['update_info/scheduler_traj']

        for key in sorted(self._summary):
            val = self._summary[key][c.CONTENT]
            setting = self._summary[key][c.LOG_SETTING]
            try:
                print('|'.join(str(x).ljust(key_length) for x in (f"{key} - AVG", np.mean(val, axis=setting[c.AXIS]))))
                if setting[c.STANDARD_DEVIATION]:
                    print('|'.join(
                        str(x).ljust(key_length) for x in (f"{key} - STD DEV", np.std(val, axis=setting[c.AXIS]))))
                if setting[c.MIN_MAX]:
                    print(
                        '|'.join(str(x).ljust(key_length) for x in (f"{key} - MIN", np.min(val, axis=setting[c.AXIS]))))
                    print(
                        '|'.join(str(x).ljust(key_length) for x in (f"{key} - MAX", np.max(val, axis=setting[c.AXIS]))))
            except:
                print(val)
                print(key)
                assert 0
        print("=" * 100)
