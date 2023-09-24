import os
import pickle
import numpy as np
import csv
from collections import defaultdict

root_folder_path = "/home/omen/Downloads/eval_data/eval/stack"  # from eval.pkl to csv

for folder_name in os.listdir(root_folder_path):
    folder_path = os.path.join(root_folder_path, folder_name)
    if os.path.isdir(folder_path):
        model_success_rates = defaultdict(list)

        for filename in os.listdir(folder_path):
            if filename.endswith('.pkl'):
                full_path = os.path.join(folder_path, filename)
                with open(full_path, 'rb') as f:
                    data = pickle.load(f)
                    eval_result = data["executed_task_successes"]
                    success_rate = np.sum(eval_result == 1) / len(eval_result)

                    model_name = filename.split('_')[1]
                    model_success_rates[model_name].append(success_rate)

        model_stats = {}

        for model_name, success_rates in model_success_rates.items():
            if len(success_rates) >= 5:
                mean = np.mean(success_rates)
                std_dev = np.std(success_rates)
                model_stats[model_name] = {'mean': mean, 'std_dev': std_dev}

        csv_file = f"{folder_name}.csv"
        with open(csv_file, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Model Name', 'Mean Success Rate', 'Standard Deviation'])
            for model_name, stats in sorted(model_stats.items(), key=lambda x: int(x[0])):
                csvwriter.writerow([model_name, stats['mean'], stats['std_dev']])
