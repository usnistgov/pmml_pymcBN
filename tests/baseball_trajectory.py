"""
Say I want to estimate the spread of a thrower's distance, but I only have a radar gun aimed
at the location the ball leaves his hand. I have some reasonable assumptions for aleatory things
like his height, the angle he throws at to maximize distance, etc.

I place priors on the loc/scale of his velocity distribution, and set the observations (10 of them).
Then I calculate the distance deterministically from the variables.
"""

from pymcnet.net import *
import matplotlib.pyplot as plt

from pymc3.math import sin, cos, sqrt
from theano.tensor import mean

np.random.seed(19806)  # random.org
v_obsv = np.random.normal(loc=28., scale=1., size=10.)  # m/s

# gravity
g = 9.81  # m/s**2

D = BayesianNetwork()
D.add_node('theta', dist=pm.Uniform, lower=np.pi/4.-.1, upper=np.pi/4.+.1,
           dist_type='Uniform')  # close to 45deg
D.add_node('height', dist=pm.Uniform, lower=1., upper=2.,
           dist_type='Uniform')  # about human-height

D.add_node('v_mean', dist=pm.Uniform, lower=1., upper=30.,
           dist_type='Uniform')  # from slow to best available
D.add_node('v_sd', dist=pm.HalfCauchy, beta=5., dist_type='HalfCauchy')  # something
D.add_node('velocity', dist=pm.Normal, mu=lambda: D.d('v_mean'), sd=lambda: D.d('v_sd'),
           dist_type='Normal', observed=v_obsv)
D.add_edge('v_mean', 'velocity', var='mu')
D.add_edge('v_sd', 'velocity', var='sd')

def distance_f():
    return D.d('v_mean') * cos(D.d('theta')) * \
           (D.d('v_mean') * sin(D.d('theta')) + sqrt((D.d('v_mean') * sin(D.d('theta'))) ** 2 +\
                                                     2 * g * D.d('height')) / g)
D.add_node('distance', dist=pm.Deterministic, var=distance_f, dist_type='Deterministic')
D.add_edges_from([(i, 'distance') for i in ['theta', 'height', 'v_mean']], var='var')

draw_net(D)

with pm.Model() as projectile_model:
    instantiate_pm(D)
with projectile_model:
    step = pm.Metropolis()
    proj_trace = pm.sample(5000, step=step)

pm.traceplot(proj_trace[1000:])
plt.show()