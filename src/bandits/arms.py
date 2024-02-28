from functools import partial
from typing import Callable, Optional, Protocol, TypeAlias
import numpy as np
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
        return f"<{self.__class__.__qualname__}>"

    def __repr__(self) -> str:
        return self.__class__.__qualname__

    def __call__(self):
        return self.pull()


class ConstantArm(BaseArm):

    def __init__(self, constant: float, seed=None) -> None:
        super().__init__(seed=seed)  # does not matter just to comply with interface

        self.mean = constant

    def pull(self, size=None):
        return self.mean


class BernulliArm(BaseArm):
    """ Simple arm with Bernoulli distribution support {0,1}"""

    def __init__(self, p: float, seed=None):
        super().__init__(seed=seed)

        self.mean = p

    def pull(self, size=None):
        return bernuolli(self.mean, seed=self.rng, size=size)

    def __repr__(self):
        return f"{self.__class__.__qualname__}(p={self.mean:.3g}, seed={self.seed})"


class GaussArm(BaseArm):
    """ Simple arm with Gaussian distribution"""

    def __init__(self, mu: float = None, sigma: float = None, seed=None):
        super().__init__(seed=seed)

        self.mean = round(self.rng.uniform(), 2) if mu is None else mu
        self.sigma = round(self.rng.uniform(), 2) if sigma is None else sigma

    def pull(self, size=None):
        return self.rng.normal(loc=self.mean, scale=self.sigma, size=size)


class PoissonArm(BaseArm):

    def __init__(self, lam: float, seed=None):
        super().__init__(seed=seed)

        self.mean = lam

    def pull(self, size=None):
        return self.rng.poisson(lam=self.mean, size=size)

Vector1d: TypeAlias = list[float] | np.ndarray

class ContextualBaseArm:

    def __init__(self, seed=None) -> None:
        self.rng = check_random_state(seed)

    def pull(self, context: Vector1d) -> float:
        raise NotImplementedError("method pull not implemented")

    def __str__(self) -> str:
        return self.__class__.__qualname__
    
    def __call__(self, context: Vector1d, **kwargs) -> float:
        return self.pull(context, **kwargs)
    
class LinearArm(ContextualBaseArm):

    def __init__(self, theta: Vector1d, seed=None) -> None:
        super().__init__(seed)
        self.theta = theta

    def pull(self, context: Vector1d, size=None) -> float:

        return np.dot(self.theta, context)


class LinearNormalArm(ContextualBaseArm):
    """Class for generating rewards with linear function=(real_theta dot context)
    real expected reward is linear and depends on the context vector and some coefficient vector theta*
    E[r_t | theta o context_t]

    based on A Contextual-Bandit Approach to Personalized News Article Recommendation (https://arxiv.org/pdf/1003.0146.pdf)

    param: theta - real coff vector (\theta* in the paper)
    param: base_arm - the generator arm for the reward distribution. Default is ConstantArm, this would produce the same reward given the 
           context R_t | theta o context_t = theta o context_t . i.e. the reward is the dot product. But 
    param: seed - the seed object from numpy (or torch soon...)
    """

    def __init__(self, theta: Vector1d, sigma=None, seed=None):
        super().__init__(seed=seed)

        self.theta = np.asarray(theta)
        self.sigma = sigma
        self.dimension = self.theta.shape[0]

    @property
    def norm(self):
        return np.linalg.norm(self.theta)

    def pull(self, context: Vector1d, size=None) -> float:
        context = np.asarray(context)
        self.current_mean = np.dot(self.theta, context)
        return self.rng.normal(self.current_mean, self.sigma, size=size)

    # def __call__(self, context: Vector1d, size=None) -> float:
    #     return super().__call__(context, size)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__['theta'].tolist()})"

