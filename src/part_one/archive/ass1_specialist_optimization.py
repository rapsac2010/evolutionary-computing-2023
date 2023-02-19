# Necessary imports
import sys

sys.path.insert(1, "evoman")

from environment import Environment
from demo_controller import player_controller
import time
import numpy as np
import os


def simulation(env, x):
    """Runs the simulation and returns fitness value from the run
    :param env: game environment
    :param x: player controller
    :return: fitness, player life, enemy life, time
    """
    f, p, e, t = env.play(pcont=x)
    return f


def evaluate(x):
    return np.array(list(map(lambda y: simulation(env, y), x)))


def initialize_population(LOWER=-1, UPPER=1, POP_SIZE=100, N_VARS=10):
    """Uniformly generates a random population with set parameters into a numpy array
    :return np.ndarray: numpy array of size (POP_SIZE, N_VARS)
    """
    population = np.random.uniform(LOWER, UPPER, (POP_SIZE, N_VARS))
    return population


def select_parents_tournament(population, fitness, N_SELECT, K):
    """Performs tournament selection to increase the selection pressure, increase K"""
    parents_idx = np.zeros(shape=(N_SELECT,), dtype=int)

    for i in range(N_SELECT):
        k_sel = np.random.randint(low=0, high=population.shape[0], size=int(K))
        parents_idx[i] = k_sel[fitness[k_sel].argmax()]

    return parents_idx


def recombination(population, fitness, alpha=0.5):
    """Performs whole arithmetic recombination between two parents and generates 2 times the
    population size"""
    offspring_shape = (population.shape[0] * 2, population.shape[1])
    offspring = np.zeros(shape=offspring_shape)
    count_offspring = 0
    while count_offspring < offspring_shape[0]:
        parents_idx = select_parents_tournament(
            population, fitness, 2, K=population.shape[0] / 3
        )
        parents = population[parents_idx, :]

        for i in range(parents_idx.shape[0]):
            offspring[count_offspring, :] = np.average(
                parents, axis=0, weights=[alpha, 1 - alpha]
            )
            count_offspring += 1

    return offspring


def mutation(population, sigma=1, p=0.8):
    """Add random noise to the p x 100% of the offspring"""
    noise = np.random.normal(0, sigma, population.shape)
    mask = np.random.rand(population.shape[0]) < p
    mutated = population + np.matmul(noise.T, np.diag(mask)).T
    return mutated


def survival_selection(population, offspring):
    """Perform (mu, lambda) selection by selecting the best mu individuals from a larger generated
    set."""
    fitness = evaluate(offspring)
    selected = (-fitness).argsort()[: population.shape[0]]
    return offspring[selected, :], fitness[selected]


def main_game_loop(env, n_generations=50, initial_pop=None):
    initial_time = time.time()

    if not initial_pop:
        N_VARS = (env.get_num_sensors() + 1) * n_hidden_neurons + (
            n_hidden_neurons + 1
        ) * 5
        pop = initialize_population(N_VARS=N_VARS, POP_SIZE=20)
    else:
        pop = initial_pop

    fitness = evaluate(pop)

    for i in range(n_generations):
        # Print the current best individual and fitness
        print(
            f"Generation {i}: best individual {np.argmax(fitness)} with fitness = {np.max(fitness):.2f} "
        )
        print(f"Generation {i}: mean fitness = {np.mean(fitness)}")

        # Apply mutation and recombination to current population to generate offspring
        offspring = mutation(recombination(pop, fitness))
        # Apply survival selection to select the next population, and calculate its fitness
        pop, fitness = survival_selection(pop, offspring)

        # Update simulation state
        solutions = [pop, fitness]
        env.update_solutions(solutions)
        env.save_state()

    # Print execution time of experiment
    end_time = time.time()
    print(
        "\nExecution time: "
        + str(round((end_time - initial_time) / 60))
        + " minutes \n"
    )

    # TODO: Keep track of best solutions and save them appropriately

    return env


if __name__ == "__main__":
    # Disable visual playing to speed up training process
    headless = True  # Set to False to visualize learning
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    experiment_name = "ass1_tryout"
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    # Set the number of neurons in the 1-layer neural network
    n_hidden_neurons = 10

    # Initialize the game environment
    env = Environment(
        experiment_name=experiment_name,
        enemies=[1],
        playermode="ai",
        player_controller=player_controller(n_hidden_neurons),
        enemymode="static",
        level=2,
        speed="fastest",
    )

    # Run experiment
    # N_VARS = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5
    # pop = initialize_population(N_VARS=N_VARS, POP_SIZE=100)

    main_game_loop(env)
