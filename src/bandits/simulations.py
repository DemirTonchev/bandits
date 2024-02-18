import numpy as np
from copy import deepcopy
from joblib import Parallel, delayed
import pandas as pd
from typing import Protocol
from bandits.base import check_random_state

class Simuation(Protocol):

    def run_step(self) -> None:
        ...
    def run(self) -> None:
        ...

class SimpleSimulation:
    """Basic class for a sigle simulation for simple (non-contextual) environment and a particular policy
    Parameters:
        environment: Environment class instance
        policy : Policy class instance
        horizont: number of iterations
        seed: seed
        record_every: how often to record results from simulations
    """

    def __init__(self, environment, policy, horizont: int = 10000, seed=None, record_every: int = 1):

        self.environment = environment
        self.policy = policy
        self.horizont = horizont
        self.t = 1

        self.cumulative_reward = 0
        self.cumulative_regret = 0
        self.best_arm_pulls = 0
        self.best_arm = environment.optimal_arm
        self.best_mean = environment.optimal_mean
        self.record_every = record_every

        self.results = []

    def run_step(self) -> None:

        # pick action (arm) and observe reward
        arm_idx = self.policy.choose_action()
        _, arm_reward, *_, info = self.environment.step(arm_idx)
        # policy(agent) observes reward and updates
        self.policy.observe_reward(arm_idx, arm_reward)

        # accumulate data for regret
        self.cumulative_reward += arm_reward
        instant_regret = self.best_mean - self.environment.means[arm_idx]
        self.cumulative_regret += instant_regret
        self.best_arm_pulls += 1 if arm_idx == self.best_arm else 0
        if self.record_every and (self.t % self.record_every == 0):
            self._data = {'t': self.t,
                          'instant_regret': instant_regret,
                          'cumulative_regret': self.cumulative_regret,
                          'best_arm_pulls': self.best_arm_pulls,
                          'arm_pulled': arm_idx
            }
            self.results.append(self._data)

        self.t += 1

    def run(self) -> None:
        
        for t in range(self.horizont):
            self.run_step()

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame(self.results)

def _run_simulation(simulation, sid = None):
    """ Helper function for Parralel computation of several simulations.
    Returns list of dictionaries of results for a particular simulation with id and policy
    type added
    """
    simulation.run()
    result = simulation.results
   
    # put simulation identifier
    for step_results in result:
        step_results.update({
                    'id': sid,
                    'policyType': type(simulation.policy).__name__,
            })
    return result

class ParallelSimulation:

    def __init__(self, simulation: SimpleSimulation, n_simulations: int = 1, n_jobs:int = 1):
        """Runs an instance of simulation n_simulations times in parrarel given by n_jobs
        Params:
            simulation - instance of SimpleSimulation with loaded policy and environment variables
            n_simulations - number of simulations to run
            n_jobs - number of parralel jobs to run
            full_results - return all the data for a simulation otherwise return only the end step results
        """
        self.simulation = simulation
        self.n_simulations = n_simulations
        self.n_jobs = n_jobs
        self.ids = [f'id-{i+1}' for i in range(n_simulations)]

    def run(self, verbose = 0):
        self.results = Parallel(n_jobs=self.n_jobs, verbose=verbose)(delayed(_run_simulation)(
                deepcopy(self.simulation), self.ids[i])
                for i in range(self.n_simulations))

    def to_pandas(self) -> pd.DataFrame:
        if not hasattr(self, 'results'):
            raise Exception('Run the simulation to get results')
        else:
            return pd.concat([pd.DataFrame(res) for res in self.results])

    def get_results(self) -> list:
        return self.results

class ContextualSimulation():
    """Sigle simulation for contextual environment and a particular policy
    Parameters:
        environment: Environment class instance
        policy : Policy class instance
        horizont: number of iterations
        seed: seed
        record_every: how often to record results from simulations
    """

    def __init__(self, environment, policy, horizont=10000, seed = None, record_every = 1):

        self.environment = environment
        self.policy = policy
        self.horizont = horizont
        self.t = 0

        self.cumulative_reward = 0
        self.cumulative_regret = 0
        self.best_arm_pulls = 0
#        self.best_arm = None
#        self.best_mean = None
        self.record_every = record_every

        self.results = []

    def run_step(self):
        # environment generates context and rewards (unknown to the agent/policy)
        context = self.environment.generate_context()
        # pick action (arm) and observe reward
        arm_idx = self.policy.choose_action(context)
        arm_reward = self.environment.get_stochastic_reward(arm_idx)
        # agent/agent observes reward
        self.policy.observe_reward(arm_idx, arm_reward)

        # accumulate data for regret
        self.cumulative_reward += arm_reward
        instant_regret = self.environment.current_optimal_mean - self.environment.current_means[arm_idx]
        self.cumulative_regret += instant_regret
        self.best_arm_pulls += 1 if arm_idx == self.environment.current_optimal_arm else 0

        if (self.t) % self.record_every == 0:
            self._data = {'t': self.t,
                         'instant_regret': instant_regret,
                         'cumulative_regret': self.cumulative_regret,
                         'best_arm_pulls': self.best_arm_pulls,
                         'pulled_arm' : arm_idx,
                         'reward': arm_reward,
                         'current_best_arm': self.environment.current_optimal_arm,
                         'context': context[0]
                         }
            self.results.append(self._data)

        self.t += 1

    def run(self):
        for t in range(self.horizont):
            self.run_step()

    def to_pandas(self):
        return pd.DataFrame(self.results)