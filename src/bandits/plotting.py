import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import scipy
from math import prod


def _plot_regrets_over_time(data, y, title=None, label=None, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    ax = sns.lineplot(x='t', y=y, data=data,
                      estimator='mean',
                      errorbar=None,
                      label=label,
                      ax=ax,
                      **kwargs)
    if title is not None:
        ax.set_title(title)
    return ax


def plot_instant_regret(data, title=None, label=None, ax=None, **kwargs):
    ax = _plot_regrets_over_time(data, y='instant_regret', title=title, label=label, ax=ax, **kwargs)
    return ax


def plot_cumulative_regret(data, title=None, label=None, ax=None, **kwargs):
    ax = _plot_regrets_over_time(data, y='cumulative_regret', title=title, label=label, ax=ax, **kwargs)
    return ax


def plot_TS_priors(ts_policy, t=0, ax=None):
    X = np.linspace(0, 1, 1000)
    if ax is None:
        _, ax = plt.subplots()
    for k, params in enumerate(ts_policy.prior_data_history[t]):
        ax.plot(X, beta(*params).pdf(X), label=f'arm {k} beta({params[0]},{params[1]})')
    ax.legend(loc='best')
    ax.set_title(f't={t:<5}' + '; '.join([f'Arm {k}: {pull}' for k, pull in enumerate(ts_policy.pulls_history[t])]))
    return ax


def TS_probs3(prior_history: list[list[np.ndarray]], t: int):
    """Analitical solutions of the probability that each arm is the best at time t.
    prior_history - list of list with beta params at each t. eg
    at t=0 the list of beta params looks like
    [array([1, 1]), array([1, 1]), array([1, 1])]
    """

    a1, a2, a3 = [beta(*params) for params in prior_history[t]]

    def f1(x): return a1.pdf(x)*a2.cdf(x)*a3.cdf(x)  # noqa prob of max for a_1 
    def f2(x): return a2.pdf(x)*a1.cdf(x)*a3.cdf(x)  # noqa prob of max for a_2 
    def f3(x): return a3.pdf(x)*a1.cdf(x)*a2.cdf(x)  # noqa prob of max for a_3
    return [scipy.integrate.quad(f, 0, 1)[0] for f in [f1, f2, f3]]


def TS_probs(ts_policy, t: int):
    """Analitical solutions of the probability that each arm is the best at time t.
    Should work for any number of arms
    prior_history - list of list with beta params at each t. eg
    at t=0 the list of beta params looks like
    [array([1, 1]), array([1, 1]), array([1, 1])]

    Returns the probability of each arm being the max
    """

    def _prob(rv, rest):
        #  we want a f of x for the integration
        def prob(x):
            return rv.pdf(x) * prod([rv_.cdf(x) for rv_ in rest])
        return prob
    # --------
    rvs = [beta(*params) for params in ts_policy.prior_data_history[t]]
    fs = []
    for i in range(len(rvs)):
        rv = rvs[i]
        rest = [rvs[j] for j in range(len(rvs)) if j != i]
        fs.append(_prob(rv, rest))
    return [scipy.integrate.quad(f, 0, 1)[0] for f in fs]


def plot_ucb(estimated_means, ucbs, pulls_data, thetas, t=0, ax=None, ylim=None):
    arms = np.array(range(len(estimated_means.columns)))
    if ax is None:
        fig, ax = plt.subplots()
    ax.scatter(arms, thetas, marker='_', c='red', s=300,)
    ucb_pull_marker = (ucbs.loc[t] == ucbs.loc[t].max()).values
    ax.scatter(arms[ucb_pull_marker], ucbs.loc[t].values[ucb_pull_marker], marker=10, c='red', s=100)
    ax.errorbar(arms, estimated_means.loc[t].values, yerr=ucbs.loc[t].values - estimated_means.loc[t].values,
                fmt='o', lolims=True, color='#1f77b4')
    children = ax.get_children()
    if ylim:
        ax.set_ylim(*ylim)
    for element in children:
        if str(element).startswith('Line2D') and element.get_marker() == 10:
            element.set_marker('_')
            element.set_markersize(20)
    ax.set_xticks(arms)
    ax.set_title(f't={t:<5}' + '; '.join([f'Arm {k}: {pull}' for k, pull in enumerate(pulls_data.loc[t].values)]))
    return ax
