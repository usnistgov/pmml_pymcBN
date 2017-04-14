"""Taken from PyMC3 documentation"""
from pymcnet.net import *
import matplotlib.pyplot as plt
# import numpy as np
# import pymc3 as pm

# Initialize random number generator
np.random.seed(123)

# True parameter values
alpha, sigma = 1, 1
beta = [1, 2.5]

# Size of dataset
size = 100

# Predictor variable
X1 = np.random.randn(size)
X2 = np.random.randn(size) * 0.2

# Simulate outcome variable
Y = alpha + beta[0]*X1 + beta[1]*X2 + np.random.randn(size)*sigma

D = BayesianNetwork()
# root nodes can be defined staticly.
D.add_node('alpha', dist = pm.Normal, mu=0, sd=10, dist_type='Normal')
D.add_node('beta', dist = pm.Normal, mu=0, sd=10, shape=2, dist_type='Normal')
D.add_node('sigma', dist = pm.HalfNormal, sd=1, dist_type='HalfNormal')

#
D.add_node('Y_obs', dist = pm.Normal,
           mu = lambda: D.d('alpha') +\
                        D.d('beta')[0]*X1 +\
                        D.d('beta')[1]*X2,
           sd = lambda: D.d('sigma'),
           observed=Y,
           dist_type='Normal')
D.add_edges_from([('alpha','Y_obs'),('beta','Y_obs')], var='mu')
D.add_edges_from([('sigma','Y_obs')], var='sd')

draw_net(D)

with pm.Model() as model:
    instantiate_pm(D)

with model:
#     step = pm.Metropolis()
    trace = pm.sample(2000)
pm.traceplot(trace)
plt.show()