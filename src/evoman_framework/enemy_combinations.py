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
from pymoo.operators.mutation.pm import PolynomialMutation
from itertools import combinations


# Multiprocessing
from multiprocessing.pool import ThreadPool
from pymoo.core.problem import StarmapParallelization
from pymoo.core.problem import DaskParallelization
import multiprocessing
from dask.distributed import Client

# Imports evoman
import sys
sys.path.insert(1, "evoman")
from environment import Environment
from demo_controller import player_controller
import time
import numpy as np
import os
import itertools
from datetime import datetime
import pandas as pd
experiment_name = 'pymoo_test_enemy_combinations'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)
os.environ["SDL_VIDEODRIVER"] = "dummy"

###############################################################################
#### Partially overwritten classes
###############################################################################

# Data logger callback
class Logger(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.data['enemy'] = []
        self.data["mean_fitness"] = []
        self.data["max_fitness"] = []
        self.data["standard_deviation"] = []
        self.data["mean_gain"] = []
        self.data["max_gain"] = []
        self.data["mean_player_life"] = []
        self.data["mean_enemy_life"] = []
        self.data["mean_beat"] = []
        self.data["max_beat"] = []

    def notify(self, algorithm):
        fitVal = algorithm.pop.get("F")
        gainVal = algorithm.pop.get("Gain")
        pVal = algorithm.pop.get("P")
        eVal = algorithm.pop.get("E")
        beatVal = algorithm.pop.get("Beat")
        time = algorithm.pop.get("time")
        mean_individual = [np.mean(f) for f in fitVal]
        global comb
        self.data['enemy'].append(str(comb))
        self.data["mean_fitness"].append(np.mean(fitVal))
        self.data["max_fitness"].append(-np.min(mean_individual))
        self.data["standard_deviation"].append(np.std(mean_individual))
        self.data["mean_gain"].append(np.mean(gainVal))
        self.data["max_gain"].append(np.max(gainVal))
        self.data["mean_player_life"].append(np.mean(pVal))
        self.data["mean_enemy_life"].append(np.mean(eVal))
        self.data["mean_beat"].append(np.mean(beatVal))
        self.data["max_beat"].append(np.max(beatVal))
        print(f"beat: {beatVal}")
        print(f"max gain: {np.max(gainVal)}")
        print(f"max beat: {np.max(beatVal)}")
        print(f"max gain: {np.max(gainVal)}")
        

# Environment now returns vector of achieved fitnesses
class EnvironmentMulti(Environment):
    def cons_multi(self, values):
        return values

# Problem definition
class GeneralistAgent(ElementwiseProblem):
    def __init__(self, **kwargs):
        super().__init__(
            n_var = 265,
            n_obj = 3,
            n_constr = 0,
            xl = -10,
            xu = 10,
            **kwargs
            )

    def _evaluate(self, x, out, return_val=False, *args, **kwargs):
        f, p, e, t = env.play(pcont=x)
        print(f)
        out["F"] = -f
        out["Gain"] = np.sum(p-e)
        out["Beat"] = np.sum((p-e) > 0)
        out["P"] = np.mean(p)
        out["E"] = np.mean(e)
        out["time"] = np.sum(t)
        if return_val:
            return out


###############################################################################
#### New functions
###############################################################################

# Combine results in dict
def combine_dicts(dict1, dict2):
    for key, value in dict2.items():
        if key in dict1:
            dict1[key] += value
        else:
            dict1[key] = value
    return dict1


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
#### Problem setup, parameter settings
######################################################################

controller_instance = player_controller(10)

ETA_CROSS = 15
ETA_MUTATION = 20
PROB_CROSS = 0.9
PROB_MUTATION = 0.9

N_GENERATIONS = 10
POP_SIZE = 15

selection_op = TournamentSelection(
        func_comp=multiple_tournament,
        pressure=2,
    )
crossover_op = SBX(prob=PROB_CROSS, eta = ETA_CROSS)
mutation_op = PolynomialMutation(prob=PROB_MUTATION, eta=ETA_MUTATION)
enemy_list = [1, 2, 3, 4, 5, 6, 7, 8]
combs = np.array([*combinations(enemy_list, 5)], dtype=object)

res_dict = {}
res_dict2 = {}


######################################################################
#### Experiment run
######################################################################

# Test all created combinations
for comb in combs:
    problem = GeneralistAgent()
    problem.n_obj = len(comb)


    # Setup env against enemy combination
    env = EnvironmentMulti(experiment_name=experiment_name,
                enemies = comb,
                multiplemode="yes",
                playermode="ai",
                player_controller=controller_instance,
                enemymode="static",
                level=2,
                speed='fastest')

    # Create NSGA2 and optimize + save results
    algorithm = NSGA2(
        pop_size=POP_SIZE,
        selection=selection_op,
        crossover=crossover_op,
        mutation=mutation_op)
    res = minimize(
        problem,
        algorithm,
        ('n_gen', N_GENERATIONS),
        callback = Logger(),
        seed = 1, verbose = True)
    res_dict = combine_dicts(res_dict, res.algorithm.callback.data)

    # Create en environment with all enemies
    env = EnvironmentMulti(experiment_name=experiment_name,
                           enemies= [1, 2, 3, 4, 5, 6, 7, 8],
                           multiplemode="yes",
                           playermode="ai",
                           player_controller=controller_instance,
                           enemymode="static",
                           level=2,
                           speed='fastest')
    problem.n_obj = 8

    # Take best solution and run against all enemies
    x = res.X[-1]
    out = problem._evaluate(x, {}, return_val=True)
    out.pop('F', None)
    avg_out = {'enemy': [str(comb)]}

    # Save evaluation output
    for key, val in out.items():
        avg_out[key] = [np.mean(val)]

    res_dict2 = combine_dicts(res_dict2, avg_out)

    # Save data inbetween runs in case of errors
    res_df = pd.DataFrame.from_dict(res_dict)
    res_df2 = pd.DataFrame.from_dict(res_dict2)

    fname = "5_temp_results_nsga2_train_combs" + "_" + str(N_GENERATIONS) + '_' + str(POP_SIZE)
    res_df.to_csv(fname + '.csv')
    fname = "5_temp_results_all_nsga2_train_combs" + "_" + str(N_GENERATIONS) + '_' + str(POP_SIZE)
    res_df2.to_csv(fname + '.csv')

# Save final results
res_df = pd.DataFrame.from_dict(res_dict)
res_df2 = pd.DataFrame.from_dict(res_dict2)
now = datetime.now()
fname = "5_results_nsga2_train_combs" + str(N_GENERATIONS) + '_' + str(POP_SIZE) + '_' + now.strftime("%m%d%Y_%H_%M")
res_df.to_csv(fname + '.csv')
fname = "5_results_all_nsga2_train_combs" + "_" + str(N_GENERATIONS) + '_' + str(POP_SIZE) + '_' + now.strftime(
        "%m%d%Y_%H_%M")
res_df2.to_csv(fname + '.csv')
print(res_df)
