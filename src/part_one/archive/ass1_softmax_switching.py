# Necessary imports
import sys

sys.path.insert(1, "evoman")

from environment import Environment
from demo_controller import player_controller
from iteround import saferound
import time
import numpy as np
import os
import itertools


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


#########################
# SELECTION OPERATORS
#########################
def select_parents_tournament(population, fitness, N_SELECT, K):
    """Performs tournament selection to increase the selection pressure, increase K"""
    parents_idx = np.zeros(shape=(N_SELECT,), dtype=int)

    for i in range(N_SELECT):
        k_sel = np.random.randint(low=0, high=population.shape[0], size=int(K))
        parents_idx[i] = k_sel[fitness[k_sel].argmax()]

    return parents_idx


def survival_selection(population, offspring, fitness = None):
    """Perform (mu, lambda) selection by selecting the best mu individuals from a larger generated
    set."""
    if fitness.any():
        fitness = fitness
    else: 
        fitness = evaluate(offspring)
    selected = (-fitness).argsort()[:population.shape[0]]
    print(selected)
    return offspring[selected, :], fitness[selected]


#########################
# RECOMBINATION OPERATORS
#########################
# Whole arithmetic is same as simple
def whole_arithmetic_crossover(population, fitness, alpha=0.5, offspring_multiplier=2, offspring_count = None):
    """Performs whole arithmetic recombination between two parents and generates
    offspring_multiplier times the population size"""
    offspring_shape = (population.shape[0] * offspring_multiplier, population.shape[1])
    if offspring_count:
        offspring_shape = (population.shape[0] * offspring_count, population.shape[1])
    offspring = np.zeros(shape=offspring_shape)

    count = 0
    while count < offspring_shape[0]:
        # Parent selection
        parents = population

        # Generate offspring
        for _ in range(parents.shape[0]):
            offspring[count, :] = np.average(parents, axis=0, weights=[alpha, 1 - alpha])
            count += 1
    return offspring


def blx_alpha_crossover(population, fitness, alpha=0.5, offspring_multiplier=2, offspring_count = None):
    """Performs BLX-alpha between two parents
    """
    offspring_shape = (population.shape[0] * offspring_multiplier, population.shape[1])
    if offspring_count:
        offspring_shape = (population.shape[0] * offspring_count, population.shape[1])
    offspring = np.zeros(shape=offspring_shape)

    count = 0
    while count < offspring_shape[0]:
        # Parent selection
        parents = population

        # Generate offspring
        for j in range(parents.shape[1]):
            gamma = (1 - 2 * alpha) * np.random.rand() - alpha  # Page 67 Eiben
            offspring[count, j] = gamma * np.max(parents[:, j]) + (1 - gamma) * np.min(parents[:, j])

        count += 1

    return offspring


def simple_arithmetic_crossover(population, fitness, alpha=0.5, offspring_multiplier=2, offspring_count = None):
    """Performs simple arithmetic crossover between two parents. When alpha = 1 this boils down
    to copying the tail of the second parent onto the head of the first parent.
    """
    offspring_shape = (population.shape[0] * offspring_multiplier, population.shape[1])
    if offspring_count:
        offspring_shape = (population.shape[0] * offspring_count, population.shape[1])
    offspring = np.zeros(shape=offspring_shape)
    count = 0
    while count < offspring_shape[0]:

        # Parent selection
        parents = population
        for _ in range(parents.shape[0]):
            # Sample from 1 to shape - 1 to ensure at least one weight being transferred
            k = np.random.randint(low=1, high=parents.shape[1] - 1)

            # Copy first K weights from parent 1
            offspring[count, :k] = parents[0, :k]
            offspring[count, k:] = parents[1, k:]

            # # Roll parents to switch positions around
            # parents = np.roll(parents, shift=1, axis=0)

            count += 1

    return offspring


#########################
# MUTATION OPERATORS
#########################
def additive_gaussian_mutation(population, sigma=1, p=0.8):
    """Add random noise to the weights of the offspring,
    an individual has probability = p of being mutated
    """
    noise = np.random.normal(0, sigma, population.shape)
    mask = np.random.rand(population.shape[0]) < p
    mutated = population + np.matmul(noise.T, np.diag(mask)).T
    return mutated


def reset_coefficients_mutation(population, p=0.3):
    """Randomly resets weights with probability = p for each weight of an individual.
    Resets to a uniform number between LOWER and UPPER
    """
    # Create array of random numbers of same shape as the population
    reset_array = initialize_population(POP_SIZE=population.shape[0], N_VARS=population.shape[1])
    # Select which weights will be reset
    mask = np.random.rand(population.shape[0], population.shape[1]) < p
    mutated = np.where(mask, reset_array, population)
    return mutated


def total_reset_mutation(population, p=0.2):
    """(DUMB APPROACH): Resets all weights of an individual with probability = p
    """
    reset_array = initialize_population(POP_SIZE=population.shape[0], N_VARS=population.shape[1])
    mask = np.random.rand(population.shape[0]) < p
    # Use matrix multiplication to set certain rows (individuals) to te random numbers
    mutated = np.matmul(population.T, np.diag(mask)).T + np.matmul(reset_array.T, np.diag(~mask)).T
    return mutated


##############################
# Operator selection mechanism
##############################

def getCountsSoftMax(population, success_counts):
    counts = np.array([2] * len(success_counts)).astype(int) #Set minimum parents in each class to 2
    probabilities = np.exp(success_counts)/np.sum(np.exp(success_counts))

    counts = counts + np.array(saferound(probabilities * (population.shape[0] - sum(counts)), places = 0)).astype(int)
    print(counts)
    cumsum = np.cumsum(counts)
    print(f"success_counts: {success_counts}\nprobs:{probabilities}\ncumsum:{cumsum}")
    return cumsum



def combine_operators(population,
                      fitness,
                      classes_indx,
                      MUTATION_OPERATORS,
                      CROSSOVER_OPERATORS,
                      offspring_multiplier=2):
    """Combines the different mutation and crossover operators by giving the probability
    that each crossover-mutation combination is chosen, which can be optimised by Multi-Armed-Bandit
    for example."""

    # Obtain number of rows for each class (cumulative as in baseline)
    cumsum = classes_indx
    cumsum_shift = np.roll(classes_indx, shift=1)
    cumsum_shift[0] = 0

    # Create an array of parents the size of the population
    parents_idx = select_parents_tournament(population, fitness, population.shape[0], population.shape[0] / 3)


    succ_arr = np.zeros(len(MUTATION_OPERATORS) * len(CROSSOVER_OPERATORS))

    parents = population[parents_idx]
    parents_f = fitness[parents_idx]
    offspring = np.empty(shape=(0, population.shape[1]))
    offspring_f = np.empty(shape=(0, 1))
    for i, (mutation, crossover) in enumerate(itertools.product(MUTATION_OPERATORS, CROSSOVER_OPERATORS)):
        # Select individuals to apply a mutation-crossover combination on
        temp_parents = parents[cumsum_shift[i]:cumsum[i], :]
        temp_fit = parents_f[cumsum_shift[i]:cumsum[i]]

        for c in range(len(temp_parents)):
            p1, p2 = temp_parents[0], temp_parents[1]
            max_f = np.mean([temp_fit[0], temp_fit[1]]) + 0.1

            # Mutate and crossover using current set of parents with current mutation/crossover combination
            mutated = mutation(crossover(np.array([p1, p2]), temp_fit, offspring_multiplier=1))
            children_f = evaluate(mutated)

            # Append offspring and evaluated fitness of offspring
            offspring = np.vstack([offspring, mutated])
            offspring_f = np.append(offspring_f, children_f)

            # Set success counts for probabilities in next perio
            succ_arr[i] += np.count_nonzero(children_f > max_f)

            # Roll arrays to obtain new parents in case class is larger than 2
            temp_parents = np.roll(temp_parents, shift = 1)
            temp_fit = np.roll(temp_fit, shift = 1)
    print(f"223: success counts: {succ_arr}")
    print(f"226: offspring_f: {offspring_f}")        

    return offspring, offspring_f, succ_arr


#########################
# MAIN GAME LOOP
#########################
def main_game_loop(env, n_generations=50, initial_pop=None):
    initial_time = time.time()

    if not initial_pop.any():
        pop = initialize_population(N_VARS=265, POP_SIZE=20)
        print('test')
    else:
        pop = initial_pop

    fitness = evaluate(pop)

    # Define lists of operators to include
    MUTATION_OPERATORS = [
        additive_gaussian_mutation,
        # reset_coefficients_mutation,
        # total_reset_mutation
    ]
    CROSSOVER_OPERATORS = [
        # whole_arithmetic_crossover,
        # blx_alpha_crossover,
        simple_arithmetic_crossover
    ]
    # Set operator selection probabilities
    n_combinations = len(MUTATION_OPERATORS) * len(CROSSOVER_OPERATORS)
    class_sizes = getCountsSoftMax(pop, [1 for _ in range(n_combinations)])
    print(f"267: class_sizes = {class_sizes}")
    for i in range(n_generations):
        # Print the current best individual and fitness
        print(
            f"Generation {i}: best individual {np.argmax(fitness)} with fitness = {np.max(fitness):.2f} "
        )
        print(f"Generation {i}: mean fitness = {np.mean(fitness)}")

        # Apply mutation and recombination to current population to generate offspring
        offspring, off_fitness, succ_counts = combine_operators(
            population=pop,
            fitness=fitness,
            cum_indx=class_sizes,
            MUTATION_OPERATORS=MUTATION_OPERATORS,
            CROSSOVER_OPERATORS=CROSSOVER_OPERATORS
        )

        # Apply survival selection to select the next population, and calculate its fitness
        pop, fitness = survival_selection(pop, offspring, fitness = off_fitness)
        class_sizes = getCountsSoftMax(pop, succ_counts)


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
        enemies=[2],
        playermode="ai",
        player_controller=player_controller(n_hidden_neurons),
        enemymode="static",
        level=2,
        speed="fastest",
        logs = "on"
    )

    # Run experiment
    N_VARS = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5
    pop = initialize_population(N_VARS=N_VARS, POP_SIZE=20)

    main_game_loop(env, initial_pop = pop)
