from typing import Any, Callable, Protocol
import numpy as np
from functools import partial
from numpy.random import default_rng
from bandits.base import (bernuolli,
                          sigmoid,
                          constant_dist,
                          check_random_state,
                          base_rng,
)


class Arm(Protocol):

    mean: float
    distribution: Callable[[Any], float]

    def pull(self, *args, **kwargs) -> float:
        ...

class BaseArm:

    def __init__(self, seed=None) -> None:
        self.rng = check_random_state(seed)

    # def get_mean(self):
    #     return self.mean

    @property
    def expected_reward(self):
        return self.mean

    def pull(self, size = None):
        return self.distribution(size=size)
    
    def __str__(self) -> str:
        return self.__class__.__qualname__


class ConstantArm(BaseArm):

    def __init__(self, constant: float, seed=None) -> None:

        self.mean = constant
        self.distribution = partial(constant_dist, constant)


class BernulliArm(BaseArm):
    """ Simple arm with Bernoulli distribution support {0,1}"""

    def __init__(self, p: float = None, seed=None):
        super().__init__(seed=seed)

        p = round(np.random.uniform(), 2) if p is None else p
        assert 0 <= p <= 1, 'Bernulli parameter must be in [0,1]'

        self.mean = p
        self.distribution = partial(bernuolli, self.mean, seed=self.rng)

    def __repr__(self):
        return f"{self.__class__.__qualname__}(p={self.mean:.3g})"


class GaussArm(BaseArm):
    """ Simple arm with Gaussian distribution"""

    def __init__(self, mu: float = None, sigma: float = None, seed = None):
        super().__init__(seed=seed)

        self.mean = round(self.rng.uniform(),2) if mu is None else mu
        self.sigma = round(self.rng.uniform(),2) if sigma is None else sigma
        self.distribution = partial(self.rng.normal, self.mean, self.sigma)

class PoissonArm(BaseArm):
    
    def __init__(self, lam: float, seed=None):
        super().__init__(seed=seed)
        
        self.mean = lam
        self.distribution = partial(base_rng.poisson, lam=self.mean)
    

class SimpleLinearArm():
    """Class for generating rewards with linear function=(real_theta dot context)
    real reward probabilty depends on the context vector and some real Theta
    E[r_t | theta o context_t ]
    """

    def __init__(self, theta: list | np.ndarray, norm_strategy: str | None = 'le', seed=None):
        # TODO implement with custom callable
        self.norm_strategy = norm_strategy 
        self.norm_strategy_fn = self._norm_factory(norm_strategy)
        self.theta = self.norm_strategy_fn(np.array(theta))

        self.dimension = len(theta)
        self.current_mean = None


    def _norm_factory(self, strategy):
        def le(vec: np.array) -> np.array:
            norm = np.linalg.norm(vec)
            if norm <= 1:
                return vec
            else:
                return vec/norm
        
        def strict(vec: np.array) -> np.array:
            return vec/np.linalg.norm(vec)
            
        match strategy:
            case 'le':
                return le
            case 'strict':
                return strict
            case None | False | 'identity':
                return lambda x: x
            case _:
                raise ValueError("Unknown norm strategy")


    @property    
    def norm(self):
        return np.linalg.norm(self.theta)


    def pull(self, context_vector):
        context_vector = np.array(context_vector)
        self.current_mean = np.dot(self.theta, context_vector)
        self.reward = np.dot(self.theta, context_vector)
        return self.reward


    def get_current_mean(self, context_vector = None):
        if context_vector is not None:
            return np.dot(self.theta, context_vector)
        return self.current_mean


    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__['theta'].tolist()})"

class BernulliLinearArm(SimpleLinearArm):
    """Class for generating bernuli rewards with sigmiod(real_theta dot context)
    real reward probabilty depends on the context vector and some real Theta
    E[r_t | theta o context_t ]

    """

    def __init__(self, theta: list | np.ndarray, norm_strategy: str | None = None,
                 seed: int | None = None, ):
        super().__init__(theta, norm_strategy)
        self.rng = default_rng(seed)


    def pull(self, context_vector):
        """generetate random reward with fixed p
        """
        context_vector = np.array(context_vector)
        self.current_mean = sigmoid(np.dot(self.theta, context_vector))
        self.reward = int(self.rng.uniform() < self.current_mean) # essentially Bernuli with p = current_mean
        return self.reward # 0 or 1 


    def get_current_mean(self, context_vector = None):
        if context_vector is not None:
            return sigmoid(np.dot(self.theta, context_vector))
        return self.current_mean