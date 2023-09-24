import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = '/home/omen/Downloads/eval_data/eval/planning_success.csv'  # directly download from tensorboard
df = pd.read_csv(file_path)

success_rate_cols = [col for col in df.columns if 'planning success rate' in col.lower()]

df['Mean Success Rate'] = df[success_rate_cols].mean(axis=1)
df['Standard Deviation'] = df[success_rate_cols].std(axis=1)

sns.set(style="whitegrid")
palette = sns.color_palette("husl", 1)  # Only one line to plot

plt.figure(figsize=(12, 8))

lower_bound = np.maximum(0, df['Mean Success Rate'] - df['Standard Deviation'])
plt.plot(df['timesteps'], df['planning success rate'], label='SAC-X & Operator', color=palette[0])
plt.fill_between(df['timesteps'],
                 lower_bound,
                 df['planning success rate'] + df['Standard Deviation'],
                 color=palette[0], alpha=0.3)

plt.xlabel('Time Steps (millions)', fontsize=24)
plt.ylabel('Success Rate', fontsize=24)
plt.title('Planning for STACK', fontsize=28)
plt.legend(fontsize=24)

xticks_in_steps = [x for x in range(0, 2000001, 400000)]
xticks_in_million = [x / 1000000 for x in xticks_in_steps]
plt.xticks(xticks_in_steps, [str(round(x, 1)) for x in xticks_in_million], fontsize=24)
plt.yticks(fontsize=24)

plt.tight_layout()
plt.savefig('planning_success_rate_plot.png', dpi=300, bbox_inches='tight')

plt.show()
