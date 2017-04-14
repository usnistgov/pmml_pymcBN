"""Taken from PyMC3 documentation"""
from pymcnet.net import *
import matplotlib.pyplot as plt

disaster_data = np.ma.masked_values([4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
                            3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
                            2, 2, 3, 4, 2, 1, 3, -999, 2, 1, 1, 1, 1, 3, 0, 0,
                            1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
                            0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
                            3, 3, 1, -999, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
                            0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1], value=-999)
year = np.arange(1851, 1962)

D = BayesianNetwork()
# root nodes can be defined staticly.
D.add_node('switchpoint', dist=pm.DiscreteUniform,
           lower=year.min(), upper=year.max(),
           dist_type='DiscreteUniform', testval=1900)

D.add_node('early_rate', dist=pm.Exponential, lam=1., dist_type='Exponential')
D.add_node('late_rate', dist=pm.Exponential, lam=1., dist_type='Exponential')

# make sure to assume strings as input (node names) and use
# node dist as the vars
rate = lambda: pm.math.switch(1 * (D.d('switchpoint') >= year),
                              D.d('early_rate'),
                              D.d('late_rate'))

# inheriting nodes need to have functions as attributes
# to be called at runtime.
### ORDER OF NODE DEFINITION MUST be the ORDER OF ARGUMENTS
### and ORDER of EDGE DEFINITION
D.add_node('disasters', dist=pm.Poisson,
           mu=rate,
           observed=disaster_data,
           dist_type='Poisson')

D.add_edges_from([('late_rate', 'disasters'),
                  ('early_rate', 'disasters'),
                  ('switchpoint', 'disasters')], var='mu')

draw_net(D)

with pm.Model() as model:
    instantiate_pm(D)

with model:
    step = pm.Metropolis()
    trace = pm.sample(10000, step=step)
pm.traceplot(trace[3000:])
plt.show()

plt.figure()
plt.plot(year, disaster_data, 'o', markersize=8);
data = trace[3000:].get_values('switchpoint')
plt.axvline(data.mean(), color='r')
plt.axvspan(np.percentile(data, 5),
            np.percentile(data, 95),
            alpha=.3, color='r')
plt.ylabel("Disaster count")
plt.xlabel("Year")
plt.title('Switchpoint year with 5th-95th Quantile Uncertainty')
plt.show()