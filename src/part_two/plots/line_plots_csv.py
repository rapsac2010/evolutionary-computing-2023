# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
sns.set_style("darkgrid", {"axes.facecolor": ".9"})


# Aggregate dataframe over generations and obtain standard deviation for max and mean fitness
def prepare_df_generations(filepath):
  df = pd.read_csv(filepath)
  dfg = df.groupby('generation').mean()
  dfg["maxf_std"], dfg["meanf_std"] = df.groupby('generation').std()["max_fitness"], df.groupby('generation').std()["mean_fitness"]
  return dfg, df[df['generation'] == 30][['max_fitness', 'mean_fitness']]


# Load data
df_3_bench, df_3_bt = prepare_df_generations("../evoman_framework/ass2_benchmark_experiments/generations_benchmark_enemies257_10122022_20_07.csv")
df_4_bench, df_4_bt = prepare_df_generations("../evoman_framework/ass2_benchmark_experiments_2678/generations_tuned_enemies2678_10122022_19_44.csv")
df_3_tuned, df_3_tt = prepare_df_generations("../evoman_framework/ass2_tuned_experiments/generations_tuned_enemies257_10122022_20_29.csv")
df_4_tuned, df_4_tt = prepare_df_generations("../evoman_framework/ass2_tuned_experiments/generations_tuned_enemies2678_10122022_20_16.csv")
all_data_gen = [[df_3_bench, df_3_tuned], [df_4_bench, df_4_tuned]]
all_data_t = [[df_3_bt, df_3_tt], [df_4_bt, df_4_tt]]
enemy_arr = ['2,5,7', '4,6,8,2']       #Set enemy names
colors = ['red', 'royalblue']                       #Set line colors

###########################################################################
## T-tests for difference in means
###########################################################################
for i, dat in enumerate(all_data_t):
  for t in ['max_fitness', 'mean_fitness']:
    test = scipy.stats.ttest_ind(dat[0][t], dat[1][t], equal_var = False)
    data_lab = '3-enemy group' if i == 0 else '4-enemy group'
    print(data_lab)
    print(f"test for difference in: {t}, result: {test}")

###########################################################################
## Mean fitness averaged over 10 runs plot
###########################################################################

# Initialize figure and subplot spacing
fig, axes = plt.subplots(2,1, figsize=(6,10))
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.8,
                    wspace=0.4,
                    hspace=0.4)

# Iterate over axes, once for each enemy
for i, ax in enumerate(axes.flatten()):
  
  # Iterate twice over data, once for the static EA and once for the dynamic
  for j, (col,df) in enumerate(zip(colors, all_data_gen[i])):
    lab = 'benchmark' if j == 0 else 'tuned'
    ax.set_title(f"enemy: {enemy_arr[i]}")
    ax.plot(range(len(df)), df['mean_fitness'], label = lab, color = col)
    ax.fill_between(range(len(df)), df['mean_fitness'] - df['meanf_std'],  df['mean_fitness'] + df['meanf_std'], color = col, alpha=0.2)

  # Set plot labels for each of the 3 plots
  ax.set_ylabel("fitness")
  ax.set_xlabel("generation")

  # Put legend in last axes on the lower right
  if i == 1:
    ax.legend(loc='lower right', prop={'size': 14}, facecolor = 'white')
plt.savefig("lineplots/mean_fitness_lineplot.svg", bbox_inches='tight')


###########################################################################
## Max fitness averaged over 10 runs plot
###########################################################################

# Create figure and axes objects
fig, axes = plt.subplots(2,1, figsize=(6,10))


plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.8,
                    wspace=0.4,
                    hspace=0.4)

# Iterate over different axes, 1 for each enemy.
for i, ax in enumerate(axes.flatten()):

  # Iterate over the static and dynamic experiment, 1 line for each.
  for j, (col,df) in enumerate(zip(colors, all_data_gen[i])):

    lab = 'benchmark' if j == 0 else 'tuned'


    ax.set_title(f"enemy: {enemy_arr[i]}")
    ax.plot(range(len(df)), df['max_fitness'], label = lab, color = col)
    ax.fill_between(range(len(df)), df['max_fitness'] - df['maxf_std'],  df['max_fitness'] + df['maxf_std'], color = col, alpha=0.2)

  # Set labels for plot
  ax.set_ylabel("fitness")
  ax.set_xlabel("generation")

  # Put legend at corner of last plot
  if i == 1:
    ax.legend(loc='lower right', prop={'size': 14}, facecolor = 'white')

# Save figure
plt.savefig("lineplots/max_fitness_lineplot.svg", bbox_inches='tight')
