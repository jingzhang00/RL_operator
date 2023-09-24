import copy
import random

import numpy as np
import torch

from collections import OrderedDict
from torch.distributions import Categorical
from rl_sandbox.planning.plan import get_condition, get_task
import rl_sandbox.constants as c
from rl_sandbox.utils import get_target, solve_stack


class Scheduler:
    def __init__(self, max_schedule, num_tasks):
        self._max_schedule = max_schedule
        self._num_tasks = num_tasks

    @property
    def max_obs_len(self):
        return self._max_schedule - 1

    def state_dict(self):
        pass

    def load_state_dict(self, state_dict):
        pass

    def compute_action(self, state, h):
        raise NotImplementedError

    def deterministic_action(self, state, h):
        raise NotImplementedError

    def update_qsa(self, state, action, q_value):
        pass

    def compute_qsa(self, state, action):
        return 0.

    def compute_qs(self, state):
        return torch.zeros(self._num_tasks)


class QTableScheduler(Scheduler):
    def __init__(self,
                 max_schedule,
                 num_tasks,
                 temperature=1.,
                 temperature_decay=1.,
                 temperature_min=1.):
        super().__init__(max_schedule, num_tasks)

        self._temperature = temperature
        self._temperature_decay = temperature_decay
        self._temperature_min = temperature_min

        self.table = OrderedDict()
        self._initialize_qtable()

    def state_dict(self):
        state_dict = {
            c.Q_TABLE: self.table,
        }
        return state_dict

    def load_state_dict(self, state_dict):
        self.model.table = state_dict()[c.Q_TABLE]

    def _initialize_qtable(self, state=None):
        if state is None:
            state = [-1] * self.max_obs_len
            self.table[self.check_state(state)] = torch.zeros(self._num_tasks)

        try:
            curr_idx = state.index(-1)
        except ValueError:
            return

        for ii in range(self._num_tasks):
            state = copy.deepcopy(state)
            state[curr_idx] = ii
            self.table[self.check_state(state)] = torch.zeros(self._num_tasks)
            self._initialize_qtable(state=state)

    def check_state(self, state):
        state = list(copy.deepcopy(state))
        for _ in range(len(state), self.max_obs_len):
            state.append(-1)
        return tuple(state)

    def compute_action(self, state, h):
        state = self.check_state(state)
        dist = Categorical(logits=self.table[state] / self._temperature)
        action = dist.sample()
        lprob = dist.log_prob(action)
        value = torch.sum(self.table[state] * dist.probs)

        self._temperature = max(self._temperature_min, self._temperature * self._temperature_decay)

        return action.cpu().numpy(), value.cpu().numpy(), h.cpu().numpy(), lprob.cpu().numpy(), dist.entropy().cpu().numpy(), dist.mean.cpu().numpy(), dist.variance.cpu().numpy()

    def deterministic_action(self, state, h):
        state = self.check_state(state)
        dist = Categorical(logits=self.table[state] / self._temperature)
        action = torch.argmax(self.table[state])
        lprob = dist.log_prob(action)
        value = torch.sum(self.table[state] * dist.probs)
        return action.cpu().numpy(), value.cpu().numpy(), h.cpu().numpy(), lprob.cpu().numpy(), dist.entropy().cpu().numpy()

    def update_qsa(self, state, action, q_value):
        state = self.check_state(state)
        self.table[state][action] = q_value

    def compute_qsa(self, state, action):
        state = self.check_state(state)
        return self.table[state][action]

    def compute_qs(self, state):
        state = self.check_state(state)
        return self.table[state]


class FixedScheduler(Scheduler):
    def __init__(self,
                 intention_i,
                 num_tasks,
                 max_schedule=0):
        super().__init__(max_schedule, num_tasks)
        assert intention_i < num_tasks
        self._intention_i = np.array(intention_i, dtype=np.int)
        self._intention = np.array(intention_i, dtype=np.int)  # for compatibility
        self.zero = np.zeros((1, 1))

    def compute_action(self, state, h, env):
        env_info = getattr(env, "_return_obs")
        main_task = env._env._env._env.main_task
        und_loc = main_task.rfind('_')
        potential_suffix = [int(suf_char) for suf_char in list(main_task[und_loc + 1:])]
        init_block = potential_suffix[-1]
        potential_suffix.remove(init_block)
        target_block = random.choice(potential_suffix)
        value_task_mapping = {
            0: "openGripper",
            1: "closeGripper",
            2: "reach",
            3: "lift",
            4: "move",
        }
        intention = self._intention_i.item()
        if intention in value_task_mapping:
            task = [value_task_mapping[intention] + "_" + str(target_block)]
            base_block = init_block
        else:
            state_dict = env_info.get("effect", None)
            if state_dict is None:
                curr_subtask = None
                base_block = init_block
            else:
                for key, value in state_dict.items():
                    if value is True:
                        curr_subtask = key
                        base_block = [int(suf_char) for suf_char in list(curr_subtask[curr_subtask.rfind("_") + 1:])][0]
            env = getattr(env, "env")
            block_on_table = get_target(env)
            next_subtasks = solve_stack(main_task, curr_subtask, block_on_table, base_block)
            task = next_subtasks
        return self._intention_i, task, base_block, np.zeros(1), h.cpu().numpy(), self.zero, self.zero, None, None

    def deterministic_action(self, state, h, env):
        action, task, base_block, value, h, lprob, entropy, _, _ = self.compute_action(state, h, env)
        return action, task, base_block, value, h, lprob, entropy

    def select_action(self, intention_i, state, h):
        action, value, h, lprob, entropy, _, _ = self.compute_action(state, h)
        return np.array(intention_i, dtype=np.int), value, h, lprob, entropy


class RecycleScheduler(Scheduler):
    def __init__(self,
                 num_tasks,
                 scheduling,
                 max_schedule=0):
        super().__init__(max_schedule, num_tasks)
        self.zero = np.zeros((1, 1))
        assert np.all(np.asarray(scheduling) >= 1)
        # assert num_tasks == len(scheduling)
        self.count = 0
        self.scheduling = np.cumsum(scheduling)
        self._intention = None

    def state_dict(self):
        return {
            c.COUNT: self.count,
            c.SCHEDULING: self.scheduling,
        }

    def load_state_dict(self, state_dict):
        self.count = state_dict[c.COUNT]
        self.scheduling = state_dict[c.SCHEDULING]

    def compute_action(self, state, h, env):
        intention = np.where(self.count < self.scheduling)[0][0]
        self._intention = intention
        self.count = (self.count + 1) % self.scheduling[-1]
        env_info = getattr(env, "_return_obs")
        main_task = env._env._env._env.main_task
        und_loc = main_task.rfind('_')
        potential_suffix = [int(suf_char) for suf_char in list(main_task[und_loc + 1:])]
        init_block = potential_suffix[-1]
        potential_suffix.remove(init_block)
        target_block = random.choice(potential_suffix)
        value_task_mapping = {
            0: "openGripper",
            1: "closeGripper",
            2: "reach",
            3: "lift",
            4: "move",
        }
        intention_i = intention.item()
        if intention_i in value_task_mapping:
            task = [value_task_mapping[intention_i] + "_" + str(target_block)]
            base_block = init_block
        else:
            self._intention = np.array(5)
            state_dict = env_info.get("task_state", None)

            if state_dict is None:
                curr_subtask = None
                base_block = init_block
            else:
                for key, value in state_dict.items():
                    if value is True:
                        curr_subtask = key
                        base_block = [int(suf_char) for suf_char in list(curr_subtask[curr_subtask.rfind("_") + 1:])][0]
            env = getattr(env, "env")
            block_on_table = get_target(env)
            next_subtasks = solve_stack(main_task, curr_subtask, block_on_table, base_block)
            task = next_subtasks

        return np.array(intention, dtype=np.int), task, base_block, np.zeros(1), h.cpu().numpy(), self.zero, self.zero, None, None

    def deterministic_action(self, state, h, env):
        action, task, base_block, value, h, lprob, entropy, _, _ = self.compute_action(state, h, env)
        return action, task, base_block, value, h, lprob, entropy


class UScheduler(Scheduler):
    def __init__(self,
                 num_tasks,
                 intention_i=0,
                 max_schedule=0,
                 task_options=None):
        if task_options is not None:
            num_tasks = len(task_options)
        super().__init__(max_schedule, num_tasks)
        self._intention_i = np.array(intention_i, dtype=np.int)
        self.zero = np.zeros((1, 1))
        self.lprob = np.log(1 / num_tasks)
        self.entropy = np.array([-num_tasks * (1 / num_tasks) * self.lprob])
        if task_options is None:
            self.task_options = list(range(self._num_tasks))
        else:
            self.task_options = task_options

    def compute_action(self, state, h):
        action = np.array(np.random.choice(self.task_options))
        return action, np.zeros(1), h.cpu().numpy(), self.lprob, self.entropy, None, None

    def deterministic_action(self, state, h):
        return self._intention_i, np.zeros(1), h.cpu().numpy(), self.zero, self.entropy

    def select_action(self, intention_i, state, h):
        return np.array(intention_i, dtype=np.int), np.zeros(1), h.cpu().numpy(), self.zero, self.entropy


class ConditionalWeightedScheduler(UScheduler):
    """
    reset_task_probs should be a list of num_tasks probabilities that sums to 1.
    task_conditional_probs should be list of num_tasks lists, each num_tasks long with probabilities summing to 1.
    """

    def __init__(self,
                 task_reset_probs,
                 task_conditional_probs,
                 num_tasks,
                 intention_i=0,
                 max_schedule=0,
                 task_options=None):
        super().__init__(num_tasks, intention_i, max_schedule, task_options)
        self.task_reset_probs = task_reset_probs
        self.task_conditional_probs = task_conditional_probs

    def compute_action(self, state, h):
        if len(state) == 0:
            action = np.array(np.random.choice(self.task_options, p=self.task_reset_probs))
        else:
            action = np.array(np.random.choice(self.task_options, p=self.task_conditional_probs[state[-1]]))

        return action, np.zeros(1), h.cpu().numpy(), self.lprob, self.entropy, None, None


class WeightedRandomScheduler(UScheduler):
    """
    A fixed categorical scheduler
    """

    def __init__(self,
                 task_select_probs,
                 num_tasks,
                 intention_i=0,
                 max_schedule=0,
                 task_options=None):
        super().__init__(num_tasks, intention_i, max_schedule, task_options)
        self.task_select_probs = task_select_probs

    def compute_action(self, state, h):
        action = np.array(np.random.choice(self.task_options, p=self.task_select_probs))
        return action, None, np.zeros(1), h.cpu().numpy(), self.lprob, self.entropy, None, None


class SymbolicScheduler(WeightedRandomScheduler):
    def __init__(self,
                 task_select_probs,
                 num_tasks,
                 intention_i=0,
                 max_schedule=0,
                 task_options=None):
        super().__init__(task_select_probs, num_tasks, intention_i, max_schedule, task_options)
        self.task_select_probs = task_select_probs
        self.cur_traj = None
        self.current_task_index = 0

    def compute_action(self, state, time_step, train_env, action, h):
        if time_step == 0:
            action = np.array(np.random.choice(self.task_options, p=self.task_select_probs))
            return action, None, None, np.zeros(1), h.cpu().numpy(), self.lprob, self.entropy, None, None
        else:
            env_info = getattr(train_env, "_return_obs")
            main_task = train_env._env._env._env.main_task
            und_loc = main_task.rfind('_')
            task_no_suffix = main_task[:und_loc]
            total_suffix = [int(suf_char) for suf_char in list(main_task[und_loc + 1:])]
            if task_no_suffix == "stack":
                if time_step == 1:
                    base_block = total_suffix[-1]
                    total_suffix.remove(base_block)
                    precondition = get_condition(train_env, total_suffix, base_block, action)
                    task_dict = get_task(precondition)
                    task, action = task_dict.popitem()
                    action = np.array(action)
                else:
                    curr_task = env_info["curr_task"]
                    und = curr_task.rfind('_')
                    curr_block = int(curr_task[und + 1:])
                    base_block = env_info["base_block"]

                    if len(env_info["effect"]) == 0:
                        env = getattr(train_env, "env")
                        suffix = get_target(env)
                        precondition = get_condition(train_env, suffix, base_block, action)
                        task_dict = get_task(precondition)
                        task, action = task_dict.popitem()
                        action = np.array(action)
                    else:
                        precondition = env_info["effect"]
                        for block, pre in precondition.items():
                            if pre["on_top"]:
                                base_block = curr_block
                                env = getattr(train_env, "env")
                                suffix = get_target(env)
                                precondition = get_condition(train_env, suffix, base_block, action)
                        task_dict = get_task(precondition)
                        task, action = task_dict.popitem()
                        action = np.array(action)

            return action, task, base_block, np.zeros(1), h.cpu().numpy(), self.lprob, self.entropy, None, None


class WeightedRandomSchedulerPlusHandcraft(WeightedRandomScheduler):
    """
    A weighted random scheduler that, with epsilon probability, chooses uniformly random from
    a set of handcrafted trajectories for a single episode.
    """

    def __init__(self,
                 task_select_probs,
                 num_tasks,
                 handcraft_traj_epsilon,  # fraction of time we choose a handcrafted traj
                 handcraft_traj_options,  # list of trajs that are max_schedule long to choose from
                 intention_i=0,
                 max_schedule=0,
                 task_options=None):
        super().__init__(task_select_probs, num_tasks, intention_i, max_schedule, task_options)
        self.task_select_probs = task_select_probs
        self.handcraft_traj_epsilon = handcraft_traj_epsilon
        self.handcraft_traj_options = handcraft_traj_options
        self.cur_traj = None

    def compute_action(self, state, h):
        # first check observation to see if we're in a new traj
        if len(state) == 0:
            if np.random.rand() < self.handcraft_traj_epsilon:
                # take a handcrafted traj
                rand_int = np.random.randint(0, len(self.handcraft_traj_options))
                self.cur_traj = self.handcraft_traj_options[rand_int]
            else:
                self.cur_traj = None

        if self.cur_traj is None:
            # weighted random action
            return super().compute_action(state, h)
        else:
            # next index in cur traj
            action = np.array(self.cur_traj[len(state)])
            return action, np.zeros(1), h.cpu().numpy(), self.lprob, self.entropy, None, None
