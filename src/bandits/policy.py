from typing import Optional, Self, Protocol
from copy import deepcopy

import numpy as np
from bandits.base import check_random_state, random_argmax, base_rng, softmax
from functools import partial
from numpy import linalg


def safe_min_1d(array):
    """Useful for taking min arm index for some policies
    """
    if len(array) > 0:
        return np.min(array)
    else:
        return None

class BanditPolicy(Protocol):
    
    k_arms: int

    def choose_action(self) -> int:
        ...

    def observe_reward(self, arm_idx: int, reward: float) -> None:
        ...

class ContextualBanditPolicy(Protocol):
    
    k_arms: int

    def choose_action(self, context: np.array) -> int:
        ...

    def observe_reward(self, arm_idx: int, reward: float) -> None:
        ...

class BasePolicy:
    
    def __init__(self, k_arms: int):
        """
        k_arms (int)- number of arms
        """
        assert k_arms > 0, 'Number of arms should be positive integer'
        self.k_arms = k_arms
        self.arms_data = [[] for _ in range(k_arms)]
        self.pulls = np.zeros(k_arms, dtype=int)
        self.estimated_means = np.zeros(k_arms, dtype=float)
        self.t = 1

    def state_dict(self) -> dict:
        raise NotImplementedError()
    
    def choose_action(self) -> int:
        raise NotImplementedError()

    def observe_reward(self, arm_idx: int, reward: float) -> None:
        raise NotImplementedError()

    @classmethod
    def from_state_dict(cls, state_dict: dict) -> Self:
        raise NotImplementedError()

    def __str__(self):
        return f"{type(self).__name__} with {self.k_arms} arms"

class ConstantPick(BasePolicy):
    """Pick always the same arm, mainly for testing purposes
    """

    def __init__(self, k_arms, pick_arm: int = 0):
        super().__init__(k_arms)
        assert pick_arm <= k_arms, "The index of the picked arm should be less or equal to the number of hands"
        self.pick_arm = pick_arm

    def choose_action(self):
        return self.pick_arm

    def observe_reward(self, arm_idx, reward):
        pass


class EpsilonGreedy(BasePolicy):

    def __init__(self, k_arms, epsilon = lambda t: 1./t, seed=None):

        super().__init__(k_arms)
        self.rng = check_random_state(seed)

        if not callable(epsilon):
            self.epsilon = lambda t: epsilon
        else:
            self.epsilon = epsilon

    def choose_action(self) -> int:
        if self.t > self.k_arms:
            if self.rng.uniform() < self.epsilon(self.t): # logic for exploration
                arm_idx = self.rng.choice(range(self.k_arms)).item()
            else:
                arm_idx = random_argmax(self.estimated_means, seed=self.rng).item()
        else:
            arm_idx = self.t - 1 
        return arm_idx

    def observe_reward(self, arm_idx:int , reward: float):
        self.pulls[arm_idx] += 1
        n = self.pulls[arm_idx]
        self.estimated_means[arm_idx] = reward/n + (n-1)*self.estimated_means[arm_idx]/n
        # add to history
        self.arms_data[arm_idx].append(reward)
        self.t += 1


class UCB1(BasePolicy):
    """UCB policy from Finite-time Analysis of the Multiarmed Bandit Problem, AUER, CESA-BIANCHI;
    """
    def __init__(self, k_arms: int, keep_history: bool = False):

        super().__init__(k_arms)
        
        self.radius = lambda t, n: np.sqrt(2*np.log(t)/n)
        self.UCBs = np.full(k_arms, np.inf) # start with infitinite UCB currently this is computed after at least each arm is pulled(placeholder)
        # history keeping
        self.keep_history = keep_history
        self.UCB_history = []
        self.means_history = []
        self.pulls_history = []
        self.probability_history = []

    def _save_history(self):
        self.UCB_history.append(self.UCBs.copy())
        self.means_history.append(self.estimated_means.copy())
        self.pulls_history.append(self.pulls.copy())
        self.probability_history.append(self.t**-(4))

    def choose_action(self) -> int:
        # try each arm once then compute UCBs
        if self.t > self.k_arms:
            self.UCBs = self.estimated_means + self.radius(self.t, self.pulls)
            arm_idx = random_argmax(self.UCBs).item()
            if self.keep_history:
                self._save_history()
        else:
            arm_idx = self.t - 1 
        return arm_idx

    def observe_reward(self, arm_idx: int, reward: float) -> None:
        self.pulls[arm_idx] += 1 # lets leave this here for now
        self.arms_data[arm_idx].append(reward)
        n = self.pulls[arm_idx]
        self.estimated_means[arm_idx] = reward/n + (n-1)*self.estimated_means[arm_idx]/n
        self.t += 1

    def state_dict(self) -> dict:
        return {"estimated_means": deepcopy(self.estimated_means),
                "pulls": deepcopy(self.pulls),
                "t": self.t,
                }
    
    @classmethod
    def from_state_dict(cls, state_dict: dict) -> Self:
        policy = cls(k_arms = len(state_dict["pulls"]) )
        policy.estimated_means = state_dict["estimated_means"]
        policy.pulls = state_dict["pulls"]
        policy.t = state_dict["t"]

        return policy


class GradientSoftmax(BasePolicy):
    """beta feature
    """
    def __init__(self, k_arms: int, preferences: Optional[np.ndarray] = None, baseline: bool = True, seed=None, min_lr=0.01):
        super().__init__(k_arms)
    
        self.preferences = np.zeros(k_arms) if preferences is None else np.asarray(preferences)
        self.baseline = baseline
        self.rng = check_random_state(seed)
        self.cum_rewards = 0
        self.min_lr = min_lr
    
    def choose_action(self) -> int:
        if self.t > self.k_arms:
            arm_idx = self.rng.choice(self.k_arms, p=softmax(self.preferences))
        else:
            arm_idx = self.t - 1
        return arm_idx
    
    def observe_reward(self, arm_idx:int, reward: float|int) -> None:
        self.pulls[arm_idx] += 1
        self.arms_data[arm_idx].append(reward)
        self.cum_rewards += reward
        n = self.pulls[arm_idx]
        self.lr = max(1/n, self.min_lr)
        gradient = np.where(np.arange(self.k_arms) == arm_idx, 1, 0) - softmax(self.preferences)
        # theta = theta + lr * R_t * gradient log(pi(theta)) / dtheta
        baseline = self.cum_rewards/self.pulls.sum() if self.baseline else 0
        self.preferences += self.lr * (reward - baseline) * gradient
        self.t += 1


class BetaBernoulliTS(BasePolicy):

    def __init__(self, k_arms: int,
                 prior_data: Optional[list[np.ndarray]] = None,
                 keep_history: bool = False,
                 seed=None
                 ):
        
        super().__init__(k_arms)
        
        self.arms_data = [[] for _ in range(k_arms)]
        if prior_data is None:
            self.prior_data = [np.array([1,1]) for _ in range(k_arms)] # start with uniform
        else:
            prior_data = [np.array(data) for data in prior_data]
            self.prior_data = prior_data
        self.rng = check_random_state(seed)
        self.estimated_means = np.array([prior[0]/np.sum(prior) for prior in self.prior_data])
        # history keeping
        self.keep_history = keep_history
        self.prior_data_history = []
        self.means_history = []
        self.pulls_history = []


    def _sample_from_arms(self, size = None):
        return np.array([self.rng.beta(*prior_obs, size=size) for prior_obs in self.prior_data])

    def choose_action(self) -> int:
        if self.keep_history:
            self.prior_data_history.append(self.prior_data.copy())
            self.means_history.append(self.estimated_means.copy())
            self.pulls_history.append(self.pulls.copy())
        return random_argmax(self._sample_from_arms())

    def observe_reward(self, arm_idx: int, reward: float) -> None:
        self.prior_data[arm_idx] = self.prior_data[arm_idx] + np.array([reward, 1-reward], dtype=np.int32)

        self.arms_data[arm_idx].append(reward) 
        self.estimated_means[arm_idx] = self.prior_data[arm_idx][0]/np.sum(self.prior_data[arm_idx])
        self.t += 1
        self.pulls[arm_idx] += 1
    
    def state_dict(self) -> dict:
        return {
                "k_arms": self.k_arms,
                "prior_data": deepcopy(self.prior_data),
                "t": self.t 
                }
    
    @classmethod
    def from_state_dict(cls, state_dict: dict) -> Self:
        policy = cls(k_arms = state_dict["k_arms"],
                     prior_data = state_dict["prior_data"])
        policy.t = state_dict["t"]

        return policy


class LinUCB:
    """LinUCB policy from A Contextual-Bandit Approach to Personalized News Article
    Recommendation (li et al).
    """
    def __init__(self, k_arms, dimension, delta = 0.05, alpha = None):

        self.k_arms = k_arms
        self.d = dimension
        self.delta = delta
        self.theta_hat = [np.zeros(dimension) for _ in range(k_arms)]
        self.alpha = 1 + np.sqrt(np.log(2/delta)/(2)) if alpha is None else alpha
        # using the notation from the paper
        ## TODO make those np arrays not lists
        self.As = [np.identity(dimension) for _ in range(k_arms)]
        self.bs = [np.zeros(dimension) for _ in range(k_arms)]
        
        self.t = 1
        self.arms_data = [[] for _ in range(k_arms)]
        self.pulls = np.zeros(k_arms)
        self.estimated_means = np.zeros(k_arms)
    

    def _compute_theta(self, A, b):
        theta_hat = np.dot(linalg.inv(A), b)
        return theta_hat

    def _compute_ucb(self, theta_hat, context_vector, alpha, A):
        estimated_mean = np.dot(theta_hat, context_vector)
        deviation = alpha * np.sqrt(linalg.multi_dot([context_vector,
                                                      linalg.inv(A),
                                                      context_vector]))
        return estimated_mean + deviation

    def _update_A(self, A, context_vector):
        return A + np.outer(context_vector, context_vector)

    def _update_b(self, b, context_vector, reward):
        return b + reward * context_vector

    def choose_action(self, context):
        # context contains all context_vectors for all arms
        self.context = context
        self.ucbs = np.array([self._compute_ucb(self.theta_hat[i], context[i], self.alpha, self.As[i]) for
           i in range(self.k_arms)])
        return random_argmax(self.ucbs)

    def observe_reward(self, arm_idx, reward):
        self.As[arm_idx] = self._update_A(self.As[arm_idx], self.context[arm_idx])
        self.bs[arm_idx] = self._update_b(self.bs[arm_idx], self.context[arm_idx], reward)
        self.theta_hat[arm_idx] = self._compute_theta(self.As[arm_idx], self.bs[arm_idx])
        # add history 
        self._record_and_increment_time(arm_idx, reward)

    def _record_and_increment_time(self, arm_idx, reward):
        self.arms_data[arm_idx].append(reward)
        self.pulls[arm_idx] += 1
        self.t += 1

    def state_dict(self):
        
        return {
                "As": deepcopy(self.As), # its a list so it needs a deepcopy
                "bs": deepcopy(self.bs), # same
                "alpha" : self.alpha
                }
    
    def load_state_dict(self, state_dict: dict):
        # deepcopy so that we dont change the input dict when we train further and not have side effect

        state_dict = deepcopy(state_dict)
        if self.k_arms != len(state_dict['As']):
            raise ValueError("Loaded state_dict is from a policy with different number of arms"
                             f"Expected number of arms {self.k_arms}, but dict has {len(state_dict['As'])}!")
        if np.zeros(self.d).shape != state_dict['bs'][0].shape:
            raise ValueError("Loaded state_dict has different number of dims than the loaded policy!")
        
        self.As = state_dict['As']
        self.bs = state_dict['bs']
        self.alpha = state_dict['alpha']


class LinThompsonSampling(LinUCB):
    """This policy follows the algorithm of https://arxiv.org/pdf/1209.3352.pdf
    """
    def __init__(self, k_arms, dimension, v2: float = 1, seed: int | None = None):
        super().__init__(k_arms, dimension)
    
        self.v2 = v2
        self.rng = check_random_state(seed)
        #TODO covariance could be saved and updated on need bases and thus save on compute
        self.covariances = None # this is A_inv/B_inv 
    
    # @property
    # def nu(self):
    #     return 1
    #     # return np.sqrt(9*self.dimension*np.log(self._t/self.delta))

    def _sample_mu(self, mean, covariance):
        sample = self.rng.multivariate_normal(mean, covariance)
        return sample
    
    # def _estimate_mean(self, theta_hat, context_vector, A):
    #     estimated_reward = np.dot(self._sample_mu(theta_hat, np.linalg.inv(A)),
    #                               context_vector)
    #     print(estimated_reward)
    #     return estimated_reward 
    
    def choose_action(self, context):
        # context contains all context_vectors for all arms
        self.context = context
        A_inv = self.covariances = [np.linalg.inv(self.As[i]) for i in range(self.k_arms)] # A_inv just for clarity remove later 
        mu_tilde = [self._sample_mu(self.theta_hat[i], self.v2 * self.covariances[i]) for i in range(self.k_arms)]
        self.estimated_rewards = [np.dot(mu_tilde[i], context[i]) for i in range(self.k_arms)]
        return random_argmax(self.estimated_rewards)

    def observe_reward(self, arm_idx, reward):
        self.As[arm_idx] = self._update_A(self.As[arm_idx], self.context[arm_idx])
        self.bs[arm_idx] = self._update_b(self.bs[arm_idx], self.context[arm_idx], reward)

        self.theta_hat[arm_idx] = self._compute_theta(self.As[arm_idx], self.bs[arm_idx])
        # add history 
        self._record_and_increment_time(arm_idx, reward)