###############################################################################
# IMPORTS
###############################################################################
# Imports EA
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.core.callback import Callback
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.visualization.scatter import Scatter
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.crossover.binx import BinomialCrossover
from pymoo.operators.crossover.pcx import ParentCentricCrossover
from pymoo.operators.crossover.dex import DEX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.core.mutation import Mutation
from pymoo.core.variable import Real, get
from pymoo.operators.repair.bounds_repair import repair_random_init

# Multiprocessing
from multiprocessing.pool import ThreadPool
from pymoo.core.problem import StarmapParallelization
from pymoo.core.problem import DaskParallelization
import multiprocessing

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
import optuna


def date_str():
    """Returns a formatted date string that can be used in file names"""
    return dt.datetime.now().strftime("%m%d%Y_%H_%M")


###############################################################################
# AUXILIARY FUNCTIONS
###############################################################################
# Environment now returns vector of achieved fitnesses
class EnvironmentMulti(Environment):
    """Subclass of the original EvoMan Enviroment class, which redefines the
    multiobjective function to accomodate tracking the performance against
    individual enemies
    """

    def cons_multi(self, values):
        """Return the array of fitness values instead of the aggregate"""
        return values


###############################################################################
# PARTIALLY OVERWRITTEN CLASSES AND FUNCTIONS
###############################################################################
class Logger(Callback):
    """Create logging Callback for the minimize() function from Pymoo. The base class
    Callback has been extended with functionality for:
        - Logging fitness metrics among the generations
        - Report recent fitness values to Optuna trial
    """

    def __init__(self, trial=None, run_idx=0) -> None:
        super().__init__()
        # Initialize empty logging arrays
        self.data["mean_fitness"] = []
        self.data["max_fitness"] = []
        self.data["standard_deviation"] = []
        self.data["mean_gain"] = []
        self.data["max_gain"] = []
        self.data["mean_player_life"] = []
        self.data["mean_enemy_life"] = []
        self.data["mean_beat"] = []
        self.data["max_beat"] = []
        self.data["run"] = []
        self.data["generation"] = []

        # Track which run of the wrapper
        self.run = run_idx

        # Define attributes for pruning callbacks with Optuna
        self.trial = trial
        self.step = 0

    def notify(self, algorithm):
        # Retrieve values from the current individual
        # NOTE: This is -1 * fitness, Shape (n_individuals, n_enemies)
        fitVal = algorithm.pop.get("F")

        # These are defined as the sum of all enemies combined, Shape (n_individuals, )
        gainVal = algorithm.pop.get("Gain")
        beatVal = algorithm.pop.get("Beat")
        time = algorithm.pop.get("time")
        # These are defined as the mean over the enemies per individual, Shape (n_individuals, )
        pVal = algorithm.pop.get("P")
        eVal = algorithm.pop.get("E")

        # Calculate mean for each individual in the population
        mean_individual = np.mean(-fitVal, axis=1)  # Shape (n_individuals,)

        # Mean fitness of all individuals and all enemies
        self.data["mean_fitness"].append(np.mean(-fitVal))  # Scalar

        # Maximum, Std. of the mean individual fitnesses
        self.data["max_fitness"].append(np.max(mean_individual))  # Scalar
        self.data["standard_deviation"].append(np.std(mean_individual))  # Scalar

        # Gain, player/enemy life, and amount beat
        self.data["mean_gain"].append(np.mean(gainVal))
        self.data["max_gain"].append(np.max(gainVal))
        self.data["mean_player_life"].append(np.mean(pVal))
        self.data["mean_enemy_life"].append(np.mean(eVal))
        self.data["mean_beat"].append(np.mean(beatVal))
        self.data["max_beat"].append(np.max(beatVal))
        self.data["run"].append(self.run)
        self.data["generation"].append(self.step)

        print(f"Generation: {self.step}")
        print(f"amount beat: {beatVal} out of {fitVal.shape[1]}")
        print(f"max gain: {np.max(gainVal)}")
        print(f"max beat: {np.max(beatVal)}")
        print(f"max fitness: {np.max(mean_individual)}")
        print(f"mean fitness: {np.mean(mean_individual)}\n")

    def report_prune(self):
        """This method reports the mean_fitness of the new generation and reports
        it to the Optuna trial, which puts it through a pruning algorithm to
        check whether this trials should be pruned.
        """
        # Break out of function if no trial
        if self.trial is None:
            self.step += 1
            return

        self.trial.report(value=self.data["mean_fitness"][-1], step=self.step)

        if self.trial.should_prune():
            raise optuna.TrialPruned(
                f"Trial has been pruned at generation: {self.step}"
            )

        self.step += 1

    def __call__(self, algorithm):
        if not self.is_initialized:
            self.initialize(algorithm)
            self.is_initialized = True

        self.notify(algorithm)
        self.update(algorithm)
        self.report_prune()

    def write_to_csv(self, fname=None, direc="pymoo_test_optuna", return_df=False):
        """Writes data to csv in specified directory"""
        df = self.get_data()
        if not fname:
            fname = "_".join(
                [
                    date_str(),
                    "results_nsga2",
                    "benchmark",
                ]
            )
        path = Path(direc)
        df.to_csv(str(path.joinpath(fname + ".csv")))

        if return_df:
            return df

    def get_data(self, columns=None):
        if columns:
            d = {k: self.data[k] for k in columns}
        else:
            d = self.data

        df = pd.DataFrame.from_dict(self.data)
        return df


# Problem definition
class GeneralistAgent(ElementwiseProblem):
    """Define subclass of ElementwiseProblem from Pymoo to accomodate generalist
    agent training"""

    def __init__(
        self, enemies=[4, 8], experiment_name="ass2_benchmark_experiments", **kwargs
    ):
        n_obj = len(enemies)
        super().__init__(n_var=265, n_obj=n_obj, n_constr=0, xl=-10, xu=10, **kwargs)
        self.env = EnvironmentMulti(
            experiment_name=experiment_name,
            enemies=enemies,
            multiplemode="yes",
            playermode="ai",
            player_controller=player_controller(10),
            enemymode="static",
            level=2,
            speed="fastest",
        )

    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate function that has to be defined for Pymoo optimization."""
        f, p, e, t = self.env.play(pcont=x)

        # This is the arrat to be minimized, i.e., the negative of the fitnesses against each enemy
        out["F"] = -f

        # Additional output to save
        out["Gain"] = np.sum(p - e)
        out["Beat"] = np.sum((p - e) > 0)
        out["P"] = np.mean(p)
        out["E"] = np.mean(e)
        out["time"] = np.sum(t)


# Redefine Gaussian mutation to solve scale < 0 error
def mut_gauss(X, xl, xu, sigma, prob):
    n, n_var = X.shape
    assert len(sigma) == n
    assert len(prob) == n

    Xp = np.full(X.shape, np.inf)

    mut = np.random.random(X.shape) < prob[:, None]

    Xp[:, :] = X

    _xl = np.repeat(xl[None, :], X.shape[0], axis=0)[mut]
    _xu = np.repeat(xu[None, :], X.shape[0], axis=0)[mut]
    sigma = sigma[:, None].repeat(n_var, axis=1)[mut]

    Xp[mut] = np.random.normal(X[mut], sigma * np.abs(_xu * _xl))

    Xp = repair_random_init(Xp, X, xl, xu)

    return Xp


class GaussianMutation(Mutation):
    def __init__(self, sigma=0.1, **kwargs):
        super().__init__(**kwargs)
        self.sigma = Real(sigma, bounds=(0.01, 0.25), strict=(0.0, 1.0))

    def _do(self, problem, X, **kwargs):
        X = X.astype(float)

        sigma = get(self.sigma, size=len(X))
        prob_var = self.get_prob_var(problem, size=len(X))

        Xp = mut_gauss(X, problem.xl, problem.xu, sigma, prob_var)

        return Xp


def multiple_tournament(pop, P, **kwargs):
    """Tournament functionality used by TournamentSelection from pymoo"""
    # P defines the amount of tournaments and amount of competitors
    n_tournaments, _ = P.shape

    # Initialize the winners array
    S = np.full(n_tournaments, -1, dtype=int)

    # execute the tournaments
    for i in range(n_tournaments):
        competitors = P[i]
        # Take the mean of fitnesses for a parent against the different enemies
        fitnesses = [np.mean(pop[j].F) for j in P[i]]
        S[i] = competitors[np.argmax(fitnesses)]

    return S


#########################
# MAIN GAME LOOP
#########################
def main_game_loop(
    run_idx=1,
    enemies=[4, 8],
    pop_size=25,
    n_gen=10,
    experiment_name="ass2_benchmark_experiments",
):
    """This function combines all functionality to run the optimization loop"""
    # Initialize timer
    initial_time = time.time()

    # Define hyperparameters and operators
    params_fixed = {"POP_SIZE": pop_size, "N_GENERATIONS": (n_gen + 1)}

    # Crossover hyperparameters and operator
    # NOTE: Comment out the the operators that you will not use in this run
    params_cross = {"eta": 20, "prob": 0.5}
    crossover_op = SBX(**params_cross)

    # params = {"eta": 0.5, "zeta": 0.5}
    # crossover_op = ParentCentricCrossover(**params)

    # Mutation hyperparameters and operator
    # NOTE: Comment out the the operators that you will not use in this run
    params_mut = {"prob": 0.1, "eta": 20}
    mutation_op = PolynomialMutation(**params_mut)

    # params = {"sigma": 1}
    # mutation_op = GaussianMutation(**params)

    # Selection hyperparameter and operator
    selection_op = TournamentSelection(
        func_comp=multiple_tournament,
        pressure=4,
    )

    operators = {
        "crossover": crossover_op,
        "mutation": mutation_op,
        "selection": selection_op,
    }

    # Instantiate algorithm NSGA2 instance using defined generated operators
    algorithm = NSGA2(pop_size=params_fixed["POP_SIZE"], **operators)

    # Optimize
    problem = GeneralistAgent(enemies=enemies, experiment_name=experiment_name)
    res = minimize(
        problem,
        algorithm,
        ("n_gen", params_fixed["N_GENERATIONS"]),
        callback=Logger(run_idx=run_idx),
        verbose=False,
    )

    # Print execution time of experiment
    end_time = time.time()
    print(
        "\nExecution time: "
        + str(round((end_time - initial_time) / 60))
        + " minutes \n"
    )

    return res


def run_experiment(
    enemies, experiment_dir, experiment_name, n_runs=10, n_gen=10, pop_size=25
):
    """Runs an experiment 10 times for a certain enemy"""
    headless = True  # Set to False to visualize learning
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    logging_df_cols = [
        "run",
        "generation",
        "mean_fitness",
        "max_fitness",
        "standard_deviation",
        "mean_player_life",
        "mean_enemy_life",
        "mean_beat",
        "max_beat",
    ]
    logging_df = pd.DataFrame(columns=logging_df_cols)
    best_weights_array = pd.DataFrame()

    for run_idx in range(n_runs):
        # Run game loop
        res = main_game_loop(
            run_idx=run_idx, enemies=enemies, n_gen=n_gen, pop_size=pop_size
        )

        # Get results from every generation in the run
        run_logs = res.algorithm.callback.get_data(logging_df_cols)
        # Get best individual from this run
        best_weights = pd.DataFrame([res.X[np.mean(-res.F, axis=1).argmax(), :]])

        # Save the results
        logging_df = pd.concat(
            [logging_df, run_logs[logging_df_cols]], ignore_index=True
        )
        best_weights_array = pd.concat(
            [best_weights_array, best_weights], ignore_index=True
        )

    logging_df.to_csv(
        f"{str(experiment_dir.joinpath('generations_' + experiment_name))}.csv"
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
    N_RUNS = 2
    POP_SIZE = 5
    N_GENERATIONS = 3

    # Select which enemy to play
    enemies = [6, 7, 8]

    # Create directory to save experiment results
    experiment_dir = Path("ass2_benchmark_experiments")
    experiment_dir.mkdir(exist_ok=True)
    experiment_name = f"tuned_enemies{''.join(map(str, enemies))}_{date_str()}"

    # Run the experiment
    run_experiment(
        enemies=enemies,
        experiment_dir=experiment_dir,
        experiment_name=experiment_name,
        n_runs=N_RUNS,
        pop_size=POP_SIZE,
        n_gen=N_GENERATIONS,
    )
