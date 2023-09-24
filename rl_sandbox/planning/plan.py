import random

import numpy as np
from numpy.linalg import norm
from rl_sandbox.utils import get_pos_slice, get_vel_slices


class Predicate:
    def __init__(self, train_env, base_block, action):
        self.env_info = getattr(train_env, "_return_obs")
        self.ee_pos = self.env_info["pos"]
        self.obj_pos = self.env_info["obj_pos"]
        self.grip_pos = self.env_info["grip_pos"]
        self.obj_vel = self.env_info["obj_vel"]
        self.obj_acc = self.env_info["obj_acc"]
        self.env = getattr(train_env, "env")
        self.pb_client = self.env._pb_client
        self.table_id = self.env.tray
        self.block_on_table_height = .145
        self.block_num = len(self.env.objs_in_state)
        self.pos_slice = get_pos_slice(self.block_num, 7)
        self.pb_block_id = np.array(self.env._obj_ids)[np.array(self.env.objs_in_state)]
        self.vel_slice = get_vel_slices(self.block_num, 3)
        self.action = action
        self.base_block = base_block

    def graspable(self, idx):
        block_pos = self.obj_pos[self.pos_slice[idx]][:3]
        dist = norm(self.ee_pos - block_pos)
        return dist < 0.01 and np.all(self.grip_pos > 0.4)

    def lifted(self, idx):
        block_height = self.obj_pos[self.pos_slice[idx]][2] - self.block_on_table_height
        return block_height > 0.01

    def moving(self, idx):
        curr_obj_vel = self.obj_vel[self.vel_slice[idx]]
        curr_obj_acc = self.obj_acc[self.vel_slice[idx]]
        return norm(curr_obj_vel) > 0.05 and norm(curr_obj_acc) < 5

    def on_top(self, pb_client, idx):
        curr_block_idx = self.pb_block_id[idx]
        base_block = self.pb_block_id[int(self.base_block)]
        block_height = self.obj_pos[self.pos_slice[idx]][2]
        target_block_height = self.obj_pos[self.pos_slice[int(self.base_block)]][2]
        o2o_contacts = (len(pb_client.getContactPoints(curr_block_idx, base_block)) > 0)
        return o2o_contacts and (block_height - target_block_height > 0.035)

    def on_table(self, pb_client, idx):
        curr_block_idx = self.pb_block_id[idx]
        o2t_contacts = (len(pb_client.getContactPoints(curr_block_idx, self.table_id)) > 0)
        block_height = self.obj_pos[self.pos_slice[idx]][2] - self.block_on_table_height
        return o2t_contacts or block_height < 0.01

    def open_gripper(self):
        return self.action[-1] < 0

    def above(self, idx):
        dist = norm(self.obj_pos[self.pos_slice[idx]][:2] - self.obj_pos[self.pos_slice[int(self.base_block)]][:2])
        z_dist = self.obj_pos[self.pos_slice[idx]][2] - self.obj_pos[self.pos_slice[int(self.base_block)]][2]
        return dist < 0.01 and z_dist > 0.05

    def current_state(self, suffix):
        blocks_state_dict = {f"block_{block}": {} for block in suffix}
        for idx in suffix:
            block_key = f"block_{idx}"
            blocks_state_dict[block_key]["graspable"] = self.graspable(idx)
            blocks_state_dict[block_key]["moving"] = self.moving(idx)
            blocks_state_dict[block_key]["lifted"] = self.lifted(idx)
            blocks_state_dict[block_key]["on_top"] = self.on_top(self.pb_client, idx)
            blocks_state_dict[block_key]["on_table"] = self.on_table(self.pb_client, idx)
            blocks_state_dict[block_key]["open_gripper"] = self.open_gripper()
            blocks_state_dict[block_key]["above"] = self.above(idx)
        return blocks_state_dict


def get_condition(train_env, suffix, base_block, action):
    predicate = Predicate(train_env, base_block, action)
    precondition = predicate.current_state(suffix)
    return precondition


def get_task(precondition):
    task_dict = {}
    for block, pre in precondition.items():
        und_loc = block.rfind('_')
        suffix = [int(suf_char) for suf_char in list(block[und_loc + 1:])]
        task_list = []
        if (pre["on_table"] == True and pre["open_gripper"] == False and pre["graspable"] == False) or pre["above"] == True:
            task_list.append("openGripper")
        if pre["on_table"] == True and pre["open_gripper"] == True and pre["graspable"] == True:
            task_list.append("closeGripper")
        if pre["on_table"] == True and pre["open_gripper"] == True and pre["graspable"] == False:
            task_list.append("reach")
        if pre["on_table"] == True and pre["open_gripper"] == False and pre["graspable"] == True:
            task_list.append("lift")
        if pre["on_table"] == False and pre["open_gripper"] == False and pre["graspable"] == True and pre["lifted"] == True and pre["above"] == False:
            task_list.append("move")
        if pre["open_gripper"] == False and pre["graspable"] == True and pre["lifted"] == True and pre["moving"] == True:
            task_list.append("stack")

        task_value_mapping = {
            "openGripper": 0,
            "closeGripper": 1,
            "reach": 2,
            "lift": 3,
            "move": 4,
            "stack": 5
        }

        for task in task_list:
            if task in task_value_mapping:
                task_name = task + "_" + str(suffix[0])
                task_dict[task_name] = task_value_mapping[task]

    if task_dict:
        task = random.choice(list(task_dict.items()))
    else:
        task = ("openGripper_" + str(0), 0)

    return {task[0]: task[1]}

