import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import scipy

def plot_instant_regret(data, title = None, label = None, ax = None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    ax=sns.lineplot(x='t', y='instant_regret', data = data, estimator='mean',
                 ci=None, label = label, ax=ax, **kwargs)
    if title is not None:
        ax.set_title(title)
    return ax

def plot_cumulative_regret(data, title = None, label = None, ax = None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    ax=sns.lineplot(x='t', y='cumulative_regret', data = data, estimator='mean',
                 ci=None, label = label, ax=ax, **kwargs)
    if title is not None:
        ax.set_title(title)
    return ax

X = np.linspace(0, 1, 1000)
def plot_TS_priors(prior_history, pulls_history, t=0, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    for k, params in enumerate(prior_history.loc[t].values):
        ax.plot(X, beta(*params).pdf(X), label = f'arm {k} beta({params[0]},{params[1]})')
    ax.legend(loc='best')
    ax.set_title(f't={t}   ' + '; '.join(['Arm {}: {:.0f}'.format(k, pull) for k, pull in enumerate(pulls_history.loc[t].values)]) )
    return ax

def TS_probs(prior_history, t):
    a1, a2,a3 = [beta(*params) for params in prior_history.loc[t].values]
    f1 = lambda x: a1.pdf(x)*a2.cdf(x)*a3.cdf(x) # prob of max for a_1
    f2 = lambda x: a2.pdf(x)*a1.cdf(x)*a3.cdf(x) # prob of max for a_2
    f3 = lambda x: a3.pdf(x)*a1.cdf(x)*a2.cdf(x) # prob of max for a_3
    return [scipy.integrate.quad(f,0,1)[0] for f in [f1,f2,f3]]

def plot_ucb(estimated_means, ucbs, pulls_data, thetas, t=0, ax = None, ylim = None):
    arms = np.array(range(len(estimated_means.columns)))
    if ax is None:
        fig, ax = plt.subplots()
    ax.scatter(arms, thetas, marker='_', c ='red', s=300,)
    ucb_pull_marker = (ucbs.loc[t]==ucbs.loc[t].max()).values
    ax.scatter(arms[ucb_pull_marker], ucbs.loc[t].values[ucb_pull_marker], marker = 10, c='red', s = 100)
    ax.errorbar(arms, estimated_means.loc[t].values, yerr=ucbs.loc[t].values-estimated_means.loc[t].values,
                fmt='o', lolims=True, color = '#1f77b4')
    children = ax.get_children()
    if ylim:
        ax.set_ylim(*ylim)
    for element in children:
        if str(element).startswith('Line2D') and element.get_marker()==10:
            element.set_marker('_')
            element.set_markersize(20)
    ax.set_xticks(arms)
    ax.set_title(f't={t}   ' + '; '.join(['Arm {}: {:.0f}'.format(k, pull) for k, pull in enumerate(pulls_data.loc[t].values)]) )
    return ax