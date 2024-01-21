import numbers
import numpy as np

EPS = 1E-4
base_rng = np.random.default_rng()

# Copied from scipy.stats._qmc.check_random_state
def check_random_state(seed=None):
    """Turn `seed` into a `numpy.random.Generator` instance.

    Parameters
    ----------
    seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
        If `seed` is an int or None, a new `numpy.random.Generator` is
        created using ``np.random.default_rng(seed)``.
        If `seed` is already a ``Generator`` or ``RandomState`` instance, then
        the provided instance is used.

    Returns
    -------
    seed : {`numpy.random.Generator`, `numpy.random.RandomState`}
        Random number generator.

    """
    if seed is None or isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.default_rng(seed)
    elif isinstance(seed, (np.random.RandomState, np.random.Generator)):
        return seed
    else:
        raise ValueError(f'{seed!r} cannot be used to seed a'
                         ' numpy.random.Generator instance')
    
    
def bernuolli(p: float, seed=base_rng, *, size=None):
    rng = check_random_state(seed)
    return rng.binomial(1, p, size)


def random_argmax(vector, seed=base_rng, **kwargs):
    """Helper function to select argmax at random... otherwise np.max returns the first index of where the max value is encountered."""
    rng = check_random_state(seed)
    return rng.choice(np.nanargmax(vector), **kwargs)

def sigmoid(x):
    """Sigmonoid function. Accepts single number or numpy array
    """
    return 1/(1+np.exp(-x))

def softmax(x, axis=None):
    r"""
    Copied from scipy - https://github.com/scipy/scipy/blob/main/scipy/special/_logsumexp.py#L131
    
    Compute the softmax function.

    The softmax function transforms each element of a collection by
    computing the exponential of each element divided by the sum of the
    exponentials of all the elements. That is, if `x` is a one-dimensional
    numpy array::
        softmax(x) = np.exp(x)/sum(np.exp(x))

    Parameters
    ----------
    x : array_like
        Input array.
    axis : int or tuple of ints, optional
        Axis to compute values along. Default is None and softmax will be
        computed over the entire array `x`.

    Returns
    -------
    s : ndarray
        An array the same shape as `x`. The result will sum to 1 along the
        specified axis.
    """
    x_max = np.amax(x, axis=axis, keepdims=True)
    exp_x_shifted = np.exp(x - x_max)
    return exp_x_shifted / np.sum(exp_x_shifted, axis=axis, keepdims=True)

# def ucb_max_regret(T, environment):
#     deltas = environment.optimal_mean - environment.means
#     deltas = deltas[np.nonzero(deltas)]
#     return 8*np.log(T)*np.sum(1/deltas) + (1+(np.pi**2)/3)*np.sum(deltas)

def _check_kl_bernulli(p):
    if np.isclose(p,0): p = EPS
    elif np.isclose(p,1): p = 1-EPS
    else: p
    return p

def kl_bernulli(p, q):
    p = _check_kl_bernulli(p)
    q = _check_kl_bernulli(q)
    #should have the same check for near 1
    return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))

kl_bernulli = np.vectorize(kl_bernulli)