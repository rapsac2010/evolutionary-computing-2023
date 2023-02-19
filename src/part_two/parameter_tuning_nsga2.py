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

# from dask.distributed import Client

# Imports evoman
import sys
import os

os.chdir('src/part_two/')
print(sys.path[0])
sys.path.append(os.path.join(sys.path[0],'..', 'evoman_framework','evoman'))
sys.path.append(os.path.join(sys.path[0],'..', 'evoman_framework','evoman'))

# print(sys.path)
# sys.path.insert(1, "../evoman_framework")

from environment import Environment
from demo_controller import player_controller
import time
import numpy as np
import os
import itertools
from datetime import datetime
import pandas as pd
import optuna
from pathlib import Path
import joblib
import json

# Set experiment name and disable video
experiment_name = Path("pymoo_test_optuna")
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)
os.environ["SDL_VIDEODRIVER"] = "dummy"

###############################################################################
# AUXILIARY FUNCTIONS
###############################################################################
def date_str():
    """Returns a formatted date string that can be used in file names"""
    return datetime.now().strftime("%m%d%Y_%H_%M")


###############################################################################
# PARTIALLY OVERWRITTEN CLASSES AND FUNCTIONS
###############################################################################
class Logger(Callback):
    """Create logging Callback for the minimize() function from Pymoo. The base class
    Callback has been extended with functionality for:
        - Logging fitness metrics among the generations
        - Report recent fitness values to Optuna trial
    """

    def __init__(self, trial) -> None:
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
        df = pd.DataFrame.from_dict(self.data)
        if not fname:
            fname = "_".join(
                [
                    date_str(),
                    "results_nsga2",
                    "optuna_trial_",
                    str(self.trial.number),
                ]
            )
        path = Path(direc)
        df.to_csv(str(path.joinpath(fname + ".csv")))

        if return_df:
            return df


# Environment now returns vector of achieved fitnesses
class EnvironmentMulti(Environment):
    """Subclass of the original EvoMan Enviroment class, which redefines the
    multiobjective function to accomodate tracking the performance against
    individual enemies
    """

    def cons_multi(self, values):
        """Return the array of fitness values instead of the aggregate"""
        return values


# Problem definition
class GeneralistAgent(ElementwiseProblem):
    """Define subclass of ElementwiseProblem from Pymoo to accomodate generalist
    agent training"""

    def __init__(self, **kwargs):
        super().__init__(n_var=265, n_obj=2, n_constr=0, xl=-10, xu=10, **kwargs)
        self.env = EnvironmentMulti(
            experiment_name="pymoo_test_optuna",
            enemies=[4, 8],
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


######################################################################
# HYPERPARAMETER TUNING
######################################################################
def set_hyperparameter_space(trial: optuna.trial.Trial):
    """This function defines the hyperparamer space using Optuna's trial functionality."""

    # Define different choices of crossover and mutation operators
    crossover_choice = trial.suggest_categorical(
        name="crossover_operator", choices=["SBX", "PCX"]
    )

    mutation_choice = trial.suggest_categorical(
        name="mutation_operator", choices=["GAUSS", "POLY"]
    )

    # Set the hyperparameters of the chosen crossover operator
    if crossover_choice == "SBX":
        params = {
            "eta": trial.suggest_float(name="eta_sbx", low=0, high=50),
            "prob": trial.suggest_float(name="prob_sbx", low=0, high=1),
        }
        crossover_op = SBX(**params)

    if crossover_choice == "PCX":
        params = {
            "eta": trial.suggest_float(name="eta_pcx", low=0.1, high=2),
            "zeta": trial.suggest_float(name="zeta_pcx", low=0.1, high=2),
        }
        crossover_op = ParentCentricCrossover(**params)

    # Set the hyperparameters of the chosen crossover operator
    if mutation_choice == "GAUSS":
        params = {"sigma": trial.suggest_float(name="sigma_gauss", low=0.1, high=2)}
        mutation_op = GaussianMutation(**params)

    if mutation_choice == "POLY":
        params = {
            "prob": trial.suggest_float(name="prob_poly", low=0, high=1),
            "eta": trial.suggest_float(name="eta_poly", low=0, high=50),
        }
        mutation_op = PolynomialMutation(**params)

    # Define tournament selection and set hyperparameter
    selection_op = TournamentSelection(
        func_comp=multiple_tournament,
        pressure=trial.suggest_int(name="pressure_tournament", low=2, high=4),
    )

    # Return operators in dictionary for easy unpacking later
    operators = {
        "crossover": crossover_op,
        "mutation": mutation_op,
        "selection": selection_op,
    }

    return operators


def objective(trial: optuna.trial.Trial):
    """This function defines the objective required for the optimization setup
    of Optuna"""

    # Generate hyperparameters
    operators = set_hyperparameter_space(trial)
    params_fixed = {"POP_SIZE": 15, "N_GENERATIONS": 30}

    # Instantiate algorithm NSGA2 instance using defined generated operators
    algorithm = NSGA2(pop_size=params_fixed["POP_SIZE"], **operators)

    # Optimize
    problem = GeneralistAgent()
    res = minimize(
        problem,
        algorithm,
        ("n_gen", params_fixed["N_GENERATIONS"]),
        callback=Logger(trial),
        verbose=False,
    )

    data = res.algorithm.callback.data
    obj = np.max(data["mean_fitness"])

    res.algorithm.callback.write_to_csv()
    return obj


# Create and optimze Optuna study
def main():
    """Main function that runs the hyperparameter optimization"""
    pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=7, reduction_factor=3)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        pruner=pruner,
    )
    start = time.time()
    study.optimize(func=objective, n_trials=50, n_jobs=1)
    print("\nExecution time: " + str(round((time.time() - start) / 60)) + " minutes \n")

    # Dump study object in pickle
    joblib.dump(
        study,
        filename=experiment_name.joinpath("_".join([date_str(), "nsga2_study.pkl"])),
    )
    # Write best parameters
    with open(
        experiment_name.joinpath("_".join([date_str(), "nsga2_best_params.txt"])), "w"
    ) as f:
        f.write(json.dumps(study.best_params))


if __name__ == "__main__":
    main()
