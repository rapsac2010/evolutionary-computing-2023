import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os



df_t_257 = pd.read_csv("ass2_tuned_experiments/best_weights_tuned_enemies257_10122022_20_29.csv", header=None)
df_t_2678 = pd.read_csv("ass2_tuned_experiments/best_weights_tuned_enemies2678_10122022_20_16.csv", header=None)
df_b_257 = pd.read_csv("ass2_benchmark_experiments/best_weights_benchmark_enemies257_10122022_20_07.csv", header=None)
df_b_2678 = pd.read_csv("ass2_benchmark_experiments/best_weights_benchmark_enemies2678_10122022_19_44.csv", header=None)
all_data = [[df_t_257, df_b_257], [df_t_2678, df_b_2678]]
labels = ["tuned", "benchmark"]
enemy_arr = ['Enemies 2, 5, and 7', 'Enemies 2, 6, 7, and 8']  
enemies = ['2_5_7', '2_6_7_8']

for i, data in enumerate(all_data):
    for j in range(2):
        weights = data[j]
        for k in range(10):
            player = weights.iloc[[k]].to_numpy()[0]
            np.savetxt("solutions/best_" + labels[j] + "_" + enemies[i] + "_run_" + str(k) + ".txt", player)