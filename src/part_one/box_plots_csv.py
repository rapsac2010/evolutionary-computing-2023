# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid", {"axes.facecolor": ".9"})

## LOAD DATA
df_di_2 = pd.read_csv("../evoman_framework/ass1_dynamic_experiments/dynamic_runs_30_9_22/best_individuals_dynamic_enemy2_0930_1035.csv")
df_di_4 = pd.read_csv("../evoman_framework/ass1_dynamic_experiments/dynamic_runs_30_9_22/best_individuals_dynamic_enemy4_0930_0929.csv")
df_di_6 = pd.read_csv("../evoman_framework/ass1_dynamic_experiments/dynamic_runs_30_9_22/best_individuals_dynamic_enemy6_0930_0921.csv")
df_st_2 = pd.read_csv("../evoman_framework/ass1_benchmark_experiments/static_runs_30_9_22/best_individuals_benchmark_enemy2_1001_1534.csv")
df_st_4 = pd.read_csv("../evoman_framework/ass1_benchmark_experiments/static_runs_30_9_22/best_individuals_benchmark_enemy4_1001_1114.csv")
df_st_6 = pd.read_csv("../evoman_framework/ass1_benchmark_experiments/static_runs_30_9_22/best_individuals_benchmark_enemy6_0930_2051.csv")
all_data_indiv = [[df_st_2, df_di_2], [df_st_4, df_di_4], [df_st_6, df_di_6]]

labels = ["Static EA", "Dynamic EA"]                #Set x-labels
enemy_arr = ['Airman', 'Heatman', 'Crashman']       #Set enemy names

# Create figure instance and adjust subplot spacing
fig, ax = plt.subplots(3,1, figsize=(5,8))
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.8,
                    wspace=0.4,
                    hspace=0.5)

# Iterate over axes to create 3 subplots, one for each boxplot
for i, ax in enumerate(axes.flatten()):

  # Create boxplot
  bplot = ax.boxplot([all_data_indiv[i][0]["individual_gain"],all_data_indiv[i][1]["individual_gain"]],
                      vert=False,  # vertical box alignment
                      patch_artist=True,  # fill with color
                      labels=labels)  # will be used to label x-ticks

  # Set title, ticks and label equal for each pane
  ax.set_title(f'enemy: {enemy_arr[i]}')
  ax.set_xlabel('Gain')
  ax.set_xticks(np.arange(-60,101,20))

  # fill with colors
  colors = ['lightblue', 'pink']
  for patch, color in zip(bplot['boxes'], colors):
      patch.set_facecolor(color)

  # adding horizontal grid lines
  ax.yaxis.grid(True)

# Save figure
plt.savefig("boxplots/boxplot.svg", bbox_inches='tight')

