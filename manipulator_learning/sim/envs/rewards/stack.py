import numpy as np

from manipulator_learning.sim.envs.rewards.reach import dist_tanh


def stack_dense(stack_contact_bool, obj_poss, obj_height, ee_pos, stack_mult=10, stack_pos_mult=3, reach_mult=1,
                tanh_reach_multiplier=5):
    """ Currently only set up for stacking of one block onto another. Assumes stacking obj 0 on obj 1. """
    reach_rew = dist_tanh(obj_poss[0], ee_pos, tanh_reach_multiplier)
    b2b_rew = dist_tanh(obj_poss[0] + np.array([0, 0, obj_height]), obj_poss[1], tanh_reach_multiplier)
    return stack_mult * stack_contact_bool + stack_pos_mult * b2b_rew + reach_mult * reach_rew


def stack_sparse(pb_client, obj_pb_ids, task_obj_poss):
    o2o_contacts = []
    for i in range(0, len(obj_pb_ids) - 1):
        state = (len(pb_client.getContactPoints(obj_pb_ids[i], obj_pb_ids[i + 1])) > 0) and (task_obj_poss[i][2] - task_obj_poss[i + 1][2] > 0.035)
        o2o_contacts.append(state)
    all_o2o_contact = len(o2o_contacts) > 0 and all(o2o_contacts)
    return all_o2o_contact


def stack_sparse_eval(pb_client, obj_pb_ids, table_pb_id, task_obj_poss):
    o2t_contacts = []
    o2o_contacts = []
    for i in range(0, len(obj_pb_ids) - 1):
        o2t_contacts.append(len(pb_client.getContactPoints(obj_pb_ids[i], table_pb_id)) > 0)
    no_o2t_contact = not any(o2t_contacts)
    mapping = {i: task_obj_poss[i] for i in range(len(obj_pb_ids))}
    sorted_pos = sorted(mapping.items(), key=lambda x: x[1][2], reverse=True)
    sorted_idx = [block_id for block_id, _ in sorted_pos]
    sorted_obj_pb_id = obj_pb_ids[sorted_idx]
    for i in range(0, len(sorted_idx) - 1):
        state = (len(pb_client.getContactPoints(sorted_obj_pb_id[i], sorted_obj_pb_id[i + 1])) > 0) and (task_obj_poss[i][2] - task_obj_poss[i + 1][2] > 0.035)
        o2o_contacts.append(state)
    all_o2o_contact = len(o2o_contacts) > 0 and all(o2o_contacts)
    return no_o2t_contact and all_o2o_contact
