# Necessary imports
from audioop import cross
import sys
import csv
import pandas as pd

sys.path.insert(1, "evoman")

from environment import Environment
from demo_controller import player_controller
import time
import numpy as np
import os
import datetime as dt
from pathlib import Path


def date_str():
    """Returns a formatted date string that can be used in file names"""
    return dt.datetime.now().strftime("%m%d_%H%M")


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


def initialize_population(LOWER=-1, UPPER=1, POP_SIZE=10, N_VARS=265):
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
        parents_idx[i] = k_sel[fitness[k_sel, 0].argmax()]

    return parents_idx


def mu_comma_lambda_selection(env, population, offspring, fitness, keep_best=True):
    """Perform (mu, lambda) selection by selecting the best mu individuals from a larger generated
    set."""
    fitness_offspring = evaluate(env, offspring)
    selected = (-fitness_offspring[:, 0]).argsort()[: population.shape[0]]

    if keep_best:
        # Get best from previous population
        best_idx = (-fitness[:, 0]).argsort()[0]

        # Take the selected individuals
        offspring = np.vstack([population[best_idx, :], offspring[selected, :][:-1]])
        fitness_offspring = np.vstack(
            [fitness[best_idx, :], fitness_offspring[selected, :][:-1]]
        )

    else:
        # Take only the selected individuals from the offspring
        offspring = offspring[selected, :]
        fitness_offspring = fitness_offspring[selected, :]

    return offspring, fitness_offspring


#########################
# RECOMBINATION OPERATORS
#########################
def whole_arithmetic_crossover(
    population, fitness, alpha=0.5, offspring_multiplier=2, k=4
):
    """Performs whole arithmetic recombination between two parents and generates
    offspring_multiplier times the population size"""
    offspring_shape = (
        int(population.shape[0] * offspring_multiplier),
        population.shape[1],
    )
    offspring = np.zeros(shape=offspring_shape)

    count = 0
    while count < offspring_shape[0]:
        # Parent selection
        parents_idx = select_parents_tournament(population, fitness, 2, K=k)
        parents = population[parents_idx, :]

        # Generate offspring
        for _ in range(offspring.shape[0]):
            offspring[count, :] = np.average(
                parents, axis=0, weights=[alpha, 1 - alpha]
            )
            count += 1

            # Roll the array so that for next offspring the weights are switched
            parents = np.roll(parents, shift=1, axis=0)

    return offspring


#########################
# MUTATION OPERATORS
#########################
def additive_gaussian_mutation(population, sigma=1, p=0.2, per_individual=False):
    """Add random noise to the weights of the offspring,
    an individual has probability = p of being mutated
    """
    noise = np.random.normal(0, sigma, population.shape)
    if per_individual:
        mask = np.random.random_sample(population.shape[0]) < p
        mutated = population + np.matmul(noise.T, np.diag(mask)).T
    else:
        mask = np.random.random_sample(population.shape) < p
        mutated = population + (noise * mask)
    return mutated


#########################
# CREATE OFFSPRING
#########################
def combine_operators(population, fitness, mutation_operator, crossover_operator):
    """Combines mutation and crossover to generate offspring"""
    offspring = mutation_operator(crossover_operator(population, fitness))
    return offspring


#########################
# MAIN GAME LOOP
#########################
def main_game_loop(
    env,
    run_idx=1,
    initial_pop=None,
    n_hidden_neurons=10,
):
    """This function combines all functionality to run the optimization loop"""
    # Initialize timers
    initial_time = time.time()
    gen_time = time.time()

    if not initial_pop:
        N_VARS = (env.get_num_sensors() + 1) * n_hidden_neurons + (
            n_hidden_neurons + 1
        ) * 5
        pop = initialize_population(N_VARS=N_VARS, POP_SIZE=POP_SIZE)
    else:
        pop = initial_pop

    # Run initial population
    fitness = evaluate(env, pop)

    # Find best initial individual
    best_idx = np.argmax(fitness[:, 0])
    best_individual = {
        "individual": best_idx,
        "fitness": fitness[best_idx, 0],
        "individual_gain": fitness[best_idx, 1] - fitness[best_idx, 2],
        "player_life": fitness[best_idx, 1],
        "enemy_life": fitness[best_idx, 2],
        "time": fitness[best_idx, 3],
    }
    best_weights = pop[best_idx, :]

    # Initialize logging dataframe
    run_logs = pd.DataFrame(
        columns=["run", "generation", "mean_fitness", "max_fitness", "mean_sigma"]
    )

    for i in range(N_GENERATIONS + 1):
        # Print the current best individual and fitness
        print(
            f"RUN: {run_idx} | GENERATION: {i} | ENEMY: {env.enemyn} | POPULATION SIZE: {POP_SIZE}"
        )
        print(
            f"Best individual {np.argmax(fitness[:, 0])} with fitness = {np.max(fitness[:, 0]):.2f}"
        )
        print(f"Mean fitness = {np.mean(fitness[:, 0])}")
        print(
            f"Keep best: {KEEP_BEST} | Mutation per individual: {PER_INDIVIDUAL} | Mutation probability = {P} | K = {K}"
        )

        # Log the results in the logging dataframe
        run_logs = pd.concat(
            [
                run_logs,
                pd.DataFrame(
                    [
                        {
                            "run": run_idx,
                            "generation": i,
                            "mean_fitness": np.mean(fitness[:, 0]),
                            "standard_deviation": np.std(fitness[:, 0]),
                            "max_fitness": np.max(fitness[:, 0]),
                            "mean_sigma": 1,
                            "duration_seconds": np.abs(gen_time - time.time()),
                            "mean_player_life": np.mean(fitness[:, 1]),
                            "mean_enemy_life": np.mean(fitness[:, 2]),
                            "mean_time": np.mean(fitness[:, 3]),
                        }
                    ]
                ),
            ],
            ignore_index=False,
        )

        gen_time = time.time()
        # Log the best individual of the generation and see if it is better than the previous
        best_idx = np.argmax(fitness[:, 0])
        if fitness[best_idx, 0] >= best_individual["fitness"]:
            best_individual = {
                "individual": best_idx,
                "fitness": fitness[best_idx, 0],
                "individual_gain": fitness[best_idx, 1] - fitness[best_idx, 2],
                "player_life": fitness[best_idx, 1],
                "enemy_life": fitness[best_idx, 2],
                "time": fitness[best_idx, 3],
            }
            best_weights = pop[best_idx, :]

        # Break out of loop when max iterations reached
        if i == N_GENERATIONS:
            break

        # Define operators to include
        def crossover(x, y):
            return whole_arithmetic_crossover(
                x,
                y,
                alpha=ALPHA,
                offspring_multiplier=OFFSPRING_MULTIPLIER,
                k=K,
            )

        def mutation(x):
            return additive_gaussian_mutation(
                x, sigma=SIGMA, p=P, per_individual=PER_INDIVIDUAL
            )

        # Apply mutation and recombination to current population to generate offspring
        offspring = combine_operators(
            population=pop,
            fitness=fitness,
            mutation_operator=mutation,
            crossover_operator=crossover,
        )

        # Apply survival selection to select the next population, and calculate its fitness
        pop, fitness = mu_comma_lambda_selection(
            env, pop, offspring, fitness, keep_best=KEEP_BEST
        )

        # Update simulation state
        solutions = [pop, fitness[:, 0]]
        env.update_solutions(solutions)
        env.save_state()

    # Print execution time of experiment
    end_time = time.time()
    print(
        "\nExecution time: "
        + str(round((end_time - initial_time) / 60))
        + " minutes \n"
    )

    return env, run_logs, pd.DataFrame([best_individual]), pd.DataFrame([best_weights])


def run_experiment(enemies, experiment_dir, experiment_name, n_runs=10):
    """Runs an experiment 10 times for a certain enemy"""
    headless = True  # Set to False to visualize learning
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    n_hidden_neurons = 10

    logging_df = pd.DataFrame(
        columns=["run", "generation", "mean_fitness", "max_fitness", "mean_sigma"]
    )
    best_individuals = pd.DataFrame(
        columns=[
            "run",
            "individual",
            "fitness",
            "individual_gain",
            "player_life",
            "enemy_life",
            "time",
        ]
    )
    best_weights_array = pd.DataFrame()

    for run_idx in range(n_runs):
        # Initialize the game environment
        env = Environment(
            experiment_name=str(experiment_dir),
            enemies=enemies,
            playermode="ai",
            player_controller=player_controller(n_hidden_neurons),
            enemymode="static",
            level=2,
            speed="fastest",
        )

        env, run_logs, best_individual, best_weights = main_game_loop(
            env=env, run_idx=(run_idx + 1)
        )

        logging_df = pd.concat([logging_df, run_logs], ignore_index=True)
        best_individuals = pd.concat(
            [best_individuals, best_individual], ignore_index=True
        )
        best_weights_array = pd.concat(
            [best_weights_array, best_weights], ignore_index=True
        )

    logging_df.to_csv(
        f"{str(experiment_dir.joinpath('generations_' + experiment_name))}.csv"
    )
    best_individuals.to_csv(
        f"{str(experiment_dir.joinpath('best_individuals_' + experiment_name))}.csv"
    )
    best_weights_array.to_csv(
        f"{str(experiment_dir.joinpath('best_weights_' + experiment_name))}.csv",
        header=False,
        index=False,
    )

    print("Saving results succesful!")

    return logging_df


if __name__ == "__main__":
    # Set hyperparameters
    POP_SIZE = 5
    ALPHA = 0.5
    OFFSPRING_MULTIPLIER = 2
    SIGMA = 1
    PER_INDIVIDUAL = False
    P = 0.2
    KEEP_BEST = True
    N_GENERATIONS = 5
    K = 4
    N_RUNS = 10

    # Select which enemy to play
    enemies = [7]

    # Create directory to save experiment results
    experiment_dir = Path("ass1_benchmark_experiments")
    experiment_dir.mkdir(exist_ok=True)
    experiment_name = f"benchmark_enemy{enemies[0]}_{date_str()}"

    # Run the experiment
    run_experiment(
        enemies=enemies,
        experiment_dir=experiment_dir,
        experiment_name=experiment_name,
        n_runs=N_RUNS,
    )
