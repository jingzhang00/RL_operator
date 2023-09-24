import subprocess
import pickle

for num in range(79999, 2000000, 80000):
    for seed in [33, 42, 60, 23, 18]:
        model_path = "42"
        model_name = f"{num}.pt"
        exp_name = "sacx_experiment_setting.pkl"
        intention = 5
        print("evaluation for model:", model_name)
        command = f"bash visualize_model.bash {seed} {model_path} {model_name} {exp_name} 50 {intention} false false ''"
        execution = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True, cwd='/home/omen/Downloads/no_PDDL_more/scripts/evaluation')
        output, error = execution.communicate()


