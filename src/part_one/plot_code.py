import matplotlib.pyplot as plt
import numpy as np
import os
import benchmark_assignment_1
import sys
import pandas as pd

sys.path.insert(1, "evoman")
from environment import Environment
from demo_controller import player_controller
sys.path.insert(1, "evoman")


def run_exp(episodes, name, experiment_name, n_hidden_neurons=10, n_generations=10, pop_size=10, enemy=1):

    for i in range(episodes):
        print("NEW ITTERATION {}".format(i))
        name1 = name + "_" + str(i + 1)
        # Set the number of neurons in the 1-layer neural network
        path = experiment_name + "/" + name1

        if not os.path.exists(path):
            open(path + ".csv", 'w')
        # Initialize the game environment
        env = Environment(
            experiment_name=experiment_name,
            enemies=[enemy],
            playermode="ai",
            player_controller=player_controller(n_hidden_neurons),
            enemymode="static",
            level=2,
            speed="fastest",
        )

        # Run experiment
        N_VARS = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5
        # pop = benchmark_assignment_1.initialize_population(N_VARS=N_VARS, POP_SIZE=10)

        benchmark_assignment_1.main_game_loop(env, path, n_generations=n_generations, pop_size=pop_size)


def txt2array(itterator, experiment_name):

    means = []
    maxes = []
    populations = []
    igs = []

    for i, name in itterator:
        f = np.load(experiment_name + '/' + name + '.npy', allow_pickle=True)

        means.append(f['a'])
        maxes.append(f['b'])
        populations.append(f['c'])
        igs.append(f['d'])
    
    #     line = f.read()
    #     lists = line.split('\n')
    #     # print(lists)
    #     mean_fitness = [float(i) if i[0] != '-' else -1 * float(i[1:]) for i in lists[0][17:-2].split(', ')]
    #     max_fitness = [float(i) if i[0] != '-' else -1 * float(i[1:]) for i in lists[1][16:-2].split(', ')]
    #     lists = [i.replace("\n", "") for i in line.split(": ")]
        

    #     pop = [i.split(' ') for i in lists[3][2:-17].split(']')[:-1]]
    #     print("Pop before \n", [i.split(' ') for i in lists[3][2:-17].split(']')[:-1]])
    #     pop = [[float(j.replace('[','')) if (j[0] != '-') else -1 * float(j[1:]) for j in i if j != ' '] for i in pop]
    #     print("Pop: \n", pop)
    #     print("IG before split \n", lists[4][1:-2])
    #     ig = [float(i) if i[0] != '-' else -1 * float(i[1:]) for i in lists[4][1:-2].split(' ')]

    #     means.append(mean_fitness)
    #     maxes.append(max_fitness)
    #     populations.append(pop)
    #     igs.append(ig)
    return np.array(means), np.array(maxes), np.array(populations), np.array(igs)


def plot_fitness(fitness, names, title, legend=True, std=None):

    plt.title(title)
    for  i, (fit, name) in enumerate(zip(fitness, names)):
        plt.plot(fit, label=name)
        if std != None:
            plt.errorbar(np.arange(0,len(fit)), fit, std[i], linestyle='None', marker='^')
    plt.ylabel("episodes")
    plt.ylabel("fitness")
    if legend:
        plt.legend()
    plt.show()


def get_ig(env, individual):
    benchmark_assignment_1.simulation(env, best_indivdual)


def box_plots(populations, igs, experiment_name, n_hidden_neurons=10, enemy=1):

    

    for pop in populations:

        env = Environment(
                experiment_name=experiment_name,
                enemies=[enemy],
                playermode="ai",
                player_controller=player_controller(n_hidden_neurons),
                enemymode="static",
                level=2,
                speed="fastest",
            )

        ig = [benchmark_assignment_1.simulation(env, individual) for individual in pop]
        print(pop.shape, ig.shape)
        best_indivdual = pop[np.argmax(ig)]
       
        testing = []

        for _ in range(5):
            env = Environment(
            experiment_name=experiment_name,
            enemies=[enemy],
            playermode="ai",
            player_controller=player_controller(n_hidden_neurons),
            enemymode="static",
            level=2,
            speed="fastest",
            )
            _, p, e, _ = benchmark_assignment_1.simulation(env, best_indivdual)
            testing.append(p-e)



if __name__ == '__main__':
     # Disable visual playing to speed up training process
    headless = True  # Set to False to visualize learning
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    
    # Directory where txt files will be written to
    experiment_name = "ass1_benchmark_tryout"
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    runs = 3
    n_hidden_neurons = 10
    name = "10pop_2gen_3run"

    # uncomment to run experiments
    # run_exp(runs, name, experiment_name, n_generations=2, n_hidden_neurons=10, pop_size=10)

    names = [name + "_" + str(i + 1) for i in range(runs)]
    itterator = zip(range(runs), names)
    # Convert txt files to workable arrays
    means, maxes, population, igs = txt2array(itterator, experiment_name)
    print(means, maxes, population, igs)
    std_mean = np.std(means, axis=0)
    mean_mean = np.mean(means, axis=0)

    std_max = np.std(maxes, axis=0)
    mean_max = np.mean(maxes, axis=0)
    
    # box_plots(population, igs, experiment_name)

    plot_fitness([mean_mean], [str(1)], "mean fitness for 10 generations 5 runs", legend=False, std=[std_mean])
    plot_fitness([mean_max], [str(1)], "max fitness for 10 generations 5 runs", legend=False, std=[std_max])



