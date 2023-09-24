import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

parent_directory_path = '/home/omen/Downloads/eval_data/eval'  # after read_results.py, csv file path
all_items = os.listdir(parent_directory_path)
subdirectories = [d for d in all_items if os.path.isdir(os.path.join(parent_directory_path, d))]
sns.set(style="whitegrid")
palette = sns.color_palette("husl", 4)
model_color_dict = {}

for subdirectory in subdirectories:
    plt.figure(figsize=(12, 8))
    directory_path = os.path.join(parent_directory_path, subdirectory)
    label = os.path.basename(directory_path)
    all_files = os.listdir(directory_path)
    csv_files = [f for f in all_files if f.endswith('.csv')]
    csv_file_paths = [os.path.join(directory_path, f) for f in csv_files]

    for idx, csv_file_path in enumerate(csv_file_paths):
        df = pd.read_csv(csv_file_path)
        df.rename(columns={"Model Name": "Time Steps"}, inplace=True)
        model_name = os.path.basename(csv_file_path).split('_')[-1].split('.')[0]
        if model_name not in model_color_dict:
            model_color_dict[model_name] = palette[len(model_color_dict) % len(palette)]

        plt.plot(df['Time Steps'], df['Mean Success Rate'], label=f'{model_name}', color=model_color_dict[model_name])
        plt.fill_between(df['Time Steps'],
                         df['Mean Success Rate'] - df['Standard Deviation'],
                         df['Mean Success Rate'] + df['Standard Deviation'],
                         color=model_color_dict[model_name], alpha=0.3)
        xticks_in_steps = [x for x in range(0, 2000001, 400000)]
        xticks_in_million = [x / 1000000 for x in xticks_in_steps]
        plt.xticks(xticks_in_steps, [str(round(x, 1)) for x in xticks_in_million], fontsize=24)
        plt.yticks(fontsize=24)
        plt.xlabel('Time Steps (millions)', fontsize=24)
        plt.ylabel('Success Rate', fontsize=24)
        plt.title(f'{label}', fontsize=28)
        plt.ylim(0, 1)
        plt.savefig(f"{label}_success_rate.png", dpi=300, bbox_inches='tight')
    plt.close()

plt.figure(figsize=(10, 4))
legend_handles = [mpatches.Patch(color=color, label=model_name) for model_name, color in model_color_dict.items()]
plt.legend(handles=legend_handles, loc='center', fontsize=24)
plt.axis('off')
plt.savefig("legend.png", dpi=300, bbox_inches='tight')
plt.close()
