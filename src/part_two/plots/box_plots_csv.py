# Imports
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import scipy

font = {'family':'normal',
        'size'   : 17}

matplotlib.rc('font', **font)
sns.set_style("darkgrid", {"axes.facecolor": ".9"})

# Evaluation functions
def simulation(env, player):
    """Runs the simulation and returns fitness value from the run
    :param env: game environment
    :param x: player controller
    :return: fitness, player life, enemy life, time
    """
    f, p, e, t = env.play(pcont=player)
    return f, p, e, t


def evaluate(env, pop):
    """Returns fitness, life and time values for an array of individuals"""
    fitness = np.array(
        list(map(lambda y: simulation(env, y), pop.reshape(-1, pop.shape[1])))
    )
    return fitness


## LOAD DATA 
exts = ['_ig.csv', '_pl.csv', '_el.csv']
df_t_257 = [pd.read_csv("boxplots/best_weights_tuned_enemies257_10132022_17_52" + ext, index_col=0) for ext in exts] 
df_t_2678 = [pd.read_csv("boxplots/best_weights_tuned_enemies2678_10122022_20_16" + ext, index_col=0) for ext in exts]
df_b_257 = [pd.read_csv("boxplots/best_weights_benchmark_enemies257_10122022_20_07" + ext, index_col=0) for ext in exts]
df_b_2678 = [pd.read_csv("boxplots/best_weights_benchmark_enemies2678_10122022_19_44" + ext, index_col=0) for ext in exts]
all_data_indiv = [[df_t_257, df_b_257], [df_t_2678, df_b_2678]]

labels = ["Tuned EA", "Benchmark EA"]                #Set x-labels
enemy_arr = ['Enemies 2, 5, and 7', 'Enemies 2, 6, 7, and 8']       #Set enemy names

# Create figure instance and adjust subplot spacing
fig, axes = plt.subplots(1, 2, figsize=(5,8))
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.8,
                    wspace=0.4,
                    hspace=0.5)

headless = True  # Set to False to visualize learning
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


# Iterate over axes to create 3 subplots, one for each boxplot
for i, ax in enumerate(axes.flatten()):
  

  # Create boxplot
  data1 = all_data_indiv[i][0][0].to_numpy()
  data2 = all_data_indiv[i][1][0].to_numpy()
  
  data1 = np.mean(data1, axis=0)
  data2 = np.mean(data2, axis=0)

  st = scipy.stats.ttest_ind(data1, data2, equal_var = False)
  print(data1.mean())
  print(data2.mean())
  print(f"stactistical test for data on {enemy_arr[i]}: {st} ")
  bplot = ax.boxplot([data1, data2],
                      vert=False,  # vertical box alignment
                      patch_artist=True,  # fill with color
                      labels=labels)  # will be used to label x-ticks

  # Set title, ticks and label equal for each pane
  ax.set_title(f'{enemy_arr[i]}')
  ax.set_xlabel('Gain')
  ax.set_xticks(np.arange(-50,5, 10))

  # fill with colors
  colors = ['lightblue', 'pink']
  for patch, color in zip(bplot['boxes'], colors):
      patch.set_facecolor(color)

  # adding horizontal grid lines
  ax.yaxis.grid(True)

# Save figure
os.chdir('../assignment_2')
plt.savefig("boxplots/boxplot.svg", bbox_inches='tight')





