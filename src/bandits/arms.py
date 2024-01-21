from typing import Callable, Optional, Protocol, TypeAlias
import numpy as np
from functools import partial
from numpy.random import default_rng
from bandits.base import (bernuolli,
                          sigmoid,
                          check_random_state,
                          base_rng,
)


class Arm(Protocol):

    def pull(self, *args, **kwargs) -> float:
        ...

class BaseArm:

    def __init__(self, seed=None) -> None:
        self.rng = check_random_state(seed)

    def pull(self, size=None) -> float:
        raise NotImplementedError("method pull not implemented")
    
    def __str__(self) -> str:
        return self.__class__.__qualname__
        
    def __repr__(self) -> str:
        return self.__class__.__qualname__
    

class ConstantArm(BaseArm):

    def __init__(self, constant: float, seed=None) -> None:
        super().__init__(seed=seed) # does not matter just to comply with interface

        self.mean = constant

    def pull(self):
        return self.mean
    
    def __repr__(self) -> str:
        return super().__repr__()


class BernulliArm(BaseArm):
    """ Simple arm with Bernoulli distribution support {0,1}"""

    def __init__(self, p: float, seed=None):
        super().__init__(seed=seed)

        assert 0 <= p <= 1, 'Bernulli parameter must be in [0,1]'
        self.mean = p

    def pull(self, size=None):
        return bernuolli(self.mean, seed=self.rng, size=size)

    def __repr__(self):
        return f"{self.__class__.__qualname__}(p={self.mean:.3g}, seed={self.seed})"


class GaussArm(BaseArm):
    """ Simple arm with Gaussian distribution"""

    def __init__(self, mu: float = None, sigma: float = None, seed = None):
        super().__init__(seed=seed)

        self.mean = round(self.rng.uniform(),2) if mu is None else mu
        self.sigma = round(self.rng.uniform(),2) if sigma is None else sigma
    
    def pull(self, size=None):
        return self.rng.normal(loc=self.mean, scale=self.sigma, size=size)

class PoissonArm(BaseArm):
    
    def __init__(self, lam: float, seed=None):
        super().__init__(seed=seed)
        
        self.mean = lam
    
    def pull(self, size=None):
        return self.rng.poisson(lam=self.mean, size=size)
    

class ContextualBaseArm:

    def __init__(self, seed=None) -> None:
        self.rng = check_random_state(seed)

    def pull(self, context: list[float] | np.ndarray) -> float:
        raise NotImplementedError("method pull not implemented")
    
    def __str__(self) -> str:
        return self.__class__.__qualname__

class _BaseArmWrapper:
    def __init__(self, wrapped: BaseArm) -> None:
        self.wrapped = wrapped
    
    def pull(self, new_mean, size=None):
        self.wrapped.mean = new_mean
        return self.wrapped.pull(size=size)
    
    def __repr__(self) -> str:
        return self.wrapped.__class__.__qualname__
        

Vector1d: TypeAlias = list[float] | np.ndarray

class SimpleLinearArm(ContextualBaseArm):
    """Class for generating rewards with linear function=(real_theta dot context)
    real expected reward is linear and depends on the context vector and some coefficient vector theta*
    E[r_t | theta o context_t]

    based on A Contextual-Bandit Approach to Personalized News Article Recommendation (https://arxiv.org/pdf/1003.0146.pdf)

    param: theta - real coff vector (/theta* in the paper)
    param: base_arm - the generator arm for the reward distribution. Default is ConstantArm, this would produce the same reward given the 
           context R_t | theta o context_t = theta o context_t . i.e. the reward is the dot product. But 
    param: seed - the seed object from numpy (or torch soon...)
    """

    def __init__(self, theta: Vector1d, base_arm: Optional[BaseArm] = None, seed=None):
        super().__init__(seed=seed)
        
        self.theta = np.asarray(theta)
        self.base_arm = _BaseArmWrapper( ConstantArm(None) if base_arm is None else base_arm )

        # self.dimension = len(theta)
        self.current_mean = None


    @property    
    def norm(self):
        return np.linalg.norm(self.theta)


    def pull(self, context_vector: Vector1d, size=None) -> float:
        context_vector = np.asarray(context_vector)
        self.current_mean = np.dot(self.theta, context_vector)
        self.reward = self.base_arm.pull(self.current_mean, size=size)
        return self.reward


    def get_current_mean(self, context_vector: Optional[Vector1d] = None):
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

    def __init__(self, theta: list | np.ndarray,
                 norm_strategy: Optional[str] = None,
                 seed: int | None = None,
                 ):
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