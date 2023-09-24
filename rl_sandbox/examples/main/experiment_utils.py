import os
import rl_sandbox.constants as c


def get_save_path(exp_type, main_task, seed, exp_name, user_machine):
    if user_machine == 'local':
        save_path = f"./results/{main_task}/{seed}/{exp_type}/{exp_name}"
    elif user_machine == "None":
        save_path = None
    else:
        raise NotImplementedError("Invalid option for argument user_machine of %s" % user_machine)

    return save_path
