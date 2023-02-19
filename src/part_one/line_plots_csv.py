# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid", {"axes.facecolor": ".9"})


# Aggregate dataframe over generations and obtain standard deviation for max and mean fitness
def prepare_df_generations(filepath):
  df = pd.read_csv(filepath)
  dfg = df.groupby('generation').mean()
  dfg["maxf_std"], dfg["meanf_std"] = df.groupby('generation').std()["max_fitness"], df.groupby('generation').std()["mean_fitness"]
  return dfg

# Load data
df_gd_2 = prepare_df_generations("../evoman_framework/ass1_dynamic_experiments/dynamic_runs_30_9_22/generations_dynamic_enemy2_0930_1035.csv")
df_gd_4 = prepare_df_generations("../evoman_framework/ass1_dynamic_experiments/dynamic_runs_30_9_22/generations_dynamic_enemy4_0930_0929.csv")
df_gd_6 = prepare_df_generations("../evoman_framework/ass1_dynamic_experiments/dynamic_runs_30_9_22/generations_dynamic_enemy6_0930_0921.csv")
df_gs_2 = prepare_df_generations("../evoman_framework/ass1_benchmark_experiments/static_runs_30_9_22/generations_benchmark_enemy2_1001_1534.csv")
df_gs_4 = prepare_df_generations("../evoman_framework/ass1_benchmark_experiments/static_runs_30_9_22/generations_benchmark_enemy4_1001_1114.csv")
df_gs_6 = prepare_df_generations("../evoman_framework/ass1_benchmark_experiments/static_runs_30_9_22/generations_benchmark_enemy6_0930_2051.csv")
all_data_gen = [[df_gd_2, df_gs_2], [df_gd_4, df_gs_4], [df_gd_6, df_gs_6]]

enemy_arr = ['Airman', 'Heatman', 'Crashman']       #Set enemy names
colors = ['red', 'royalblue']                       #Set line colors

###########################################################################
## Mean fitness averaged over 10 runs plot
###########################################################################

# Initialize figure and subplot spacing
fig, axes = plt.subplots(3,1, figsize=(6,10))
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.8,
                    wspace=0.4,
                    hspace=0.4)

# Iterate over axes, once for each enemy
for i, ax in enumerate(axes.flatten()):
  
  # Iterate twice over data, once for the static EA and once for the dynamic
  for (col,df) in zip(colors, all_data_gen[i]):

    if (df['mean_sigma'] != 1).any():
      lab = "Dynamic EA"
    else:
      lab = "Static EA"

    ax.set_title(f"enemy: {enemy_arr[i]}")
    ax.plot(range(len(df)), df['mean_fitness'], label = lab, color = col)
    ax.fill_between(range(len(df)), df['mean_fitness'] - df['meanf_std'],  df['mean_fitness'] + df['meanf_std'], color = col, alpha=0.2)

  # Set plot labels for each of the 3 plots
  ax.set_ylabel("fitness")
  ax.set_xlabel("generation")

  # Put legend in last axes on the lower right
  if i == 2:
    ax.legend(loc='lower right', prop={'size': 14}, facecolor = 'white')
plt.savefig("lineplots/mean_fitness_lineplot.svg", bbox_inches='tight')


###########################################################################
## Max fitness averaged over 10 runs plot
###########################################################################

# Create figure and axes objects
fig, axes = plt.subplots(3,1, figsize=(6,10))


plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.8,
                    wspace=0.4,
                    hspace=0.4)

# Iterate over different axes, 1 for each enemy.
for i, ax in enumerate(axes.flatten()):

  # Iterate over the static and dynamic experiment, 1 line for each.
  for (col,df) in zip(colors, all_data_gen[i]):

    if (df['mean_sigma'] != 1).any():
      lab = "Dynamic EA"
    else:
      lab = "Static EA"

    ax.set_title(f"enemy: {enemy_arr[i]}")
    ax.plot(range(len(df)), df['max_fitness'], label = lab, color = col)
    ax.fill_between(range(len(df)), df['max_fitness'] - df['maxf_std'],  df['max_fitness'] + df['maxf_std'], color = col, alpha=0.2)

  # Set labels for plot
  ax.set_ylabel("fitness")
  ax.set_xlabel("generation")

  # Put legend at corner of last plot
  if i == 2:
    ax.legend(loc='lower right', prop={'size': 14}, facecolor = 'white')

# Save figure
plt.savefig("lineplots/max_fitness_lineplot.svg", bbox_inches='tight')


###########################################################################
## Mean sigma averaged over 10 runs plot
###########################################################################

# Create figure object
fig, ax = plt.subplots(1,1, figsize=(9,6))

# Labels, colors and settings
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
colors = ['red', 'royalblue', 'mediumseagreen']
labels = [r'Airman dynamic $\sigma$',  r'Heatman dynamic $\sigma$', r'Crashman dynamic $\sigma$']
fsize = 20

x = range(len(df))

# Add line for each experiment
for i, df in enumerate([df_gd_2, df_gd_4, df_gd_6]):
  print(labels[i])
  ax.plot(x, df['mean_sigma'], label = labels[i], color = colors[i])
ax.hlines(1,0,30, color='black', label = "Constant sigma")              # Add a line for the constant sigma


# Set labels, legend and ticks
ax.set_ylabel(r"mean $\mathbf{\sigma}$", fontsize = fsize)
ax.set_xlabel("generation", fontsize = fsize)
ax.legend(prop={'size': 16}, facecolor = 'white')
ax.tick_params(axis='x', labelsize=fsize)
ax.tick_params(axis='y', labelsize=fsize)

# Save figure
plt.savefig("lineplots/sigma_plot.svg", bbox_inches='tight')
