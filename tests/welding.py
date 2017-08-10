
from pymcnet.net import *
from pmml.bn import *
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('ticks')
import pandas as pd
# Synthetic Data-set (observations)
np.random.seed(19806) #random.org, for reproduceability

num_obs = 100
l_obs = norm(8.5, 0.5).rvs(size=100)*1e-3  # m
h_obs = norm(2.6, 0.5).rvs(size=100)*1e-3  # m
g_obs = norm(1.9, .19).rvs(size=100)*1e-3  # m
E_obs = norm(225, 1).rvs(size=100)  # kJ
synth = pd.DataFrame(data=np.array([l_obs, h_obs, g_obs, E_obs]).T,
                     columns=['l','h','g', 'E'])

# for i in [l_obs, h_obs, e_obs]:
#     sns.distplot(i)
# plt.legend([
#     'l_obs',
#     'h_obs',
#     'e_obs'
# ])

n_samp = 5000


# import theano.tensor as T
D = BayesianNetwork()

D.add_node('mu_l',
           lower=8.3e-3, upper=8.6e-3,
           dist_type='Uniform')
D.add_node('sd_l',
           lower=0.2e-3, upper=0.7e-3,
           dist_type='Uniform')

D.add_node('l',
           dist_type='Normal',
           exprs = {'mu':'mu_l',
                    'sd':'sd_l'})
D.add_edge('mu_l','l', var='mu')
D.add_edge('sd_l', 'l', var='sd')

D.add_node('mu_h',
           lower=2.5e-3, upper=2.8e-3,
           dist_type='Uniform')
D.add_node('sd_h',
           lower=0.3e-3, upper=0.6e-3,
           dist_type='Uniform')

D.add_node('h',
           dist_type='Normal',
           exprs = {'mu':'mu_h',
                    'sd':'sd_h'})
D.add_edge('mu_h','h', var='mu')
D.add_edge('sd_h', 'h', var='sd')

D.add_node('mu_g',
           lower=1.6e-3, upper=2.2e-3,
           dist_type='Uniform')
D.add_node('sd_g',
           lower=0.05e-3, upper=.2e-3,
           dist_type='Uniform')

D.add_node('g',
           dist_type='Normal',
           exprs = {'mu':'mu_g',
                    'sd':'sd_g'})
D.add_edge('mu_g','g', var='mu')
D.add_edge('sd_g', 'g', var='sd')

e = .011  # m
L = .500  # m
t = .015  # m

# Volume (m^3)
D.add_node('V',
           dist_type='Deterministic',
           exprs = {'var':f'{L}*((0.75)*mu_l*mu_h + mu_g*{t} + (mu_l-mu_g)*({t}-{e})*0.5)'})

D.add_edges_from([(i,'V') for i in ['mu_l','mu_h','mu_g']],
                 var='x')

# Density (kg/m^3)
D.add_node('rho',
           mu=8250., sd=10.,
          dist_type='Normal')

H=270.     # kJ/kg
C_p=0.5    # kJ/kgK
T_i=300.   # K
T_f=1600.  # K

# D.add_node('eps_E',
#            mu=0., sd=1e4,
#            dist_type = 'Normal')

# Energy (kJ)
D.add_node('E_d',
           dist_type = 'Deterministic',
           exprs={'var':f'rho*V*({C_p}*({T_f}-{T_i}) + {H})'})
D.add_edges_from([(i, 'E_d') for i in ['rho','V']], var='var')


# Model Likelihood
D.add_node('E_L',  # kJ
           dist_type = 'Normal',
           exprs={'mu':'E_d',
                  'sd':'1.0'})
D.add_edge('E_d','E_L', var='mu')


draw_net(D, pretty=True)



O = D.copy()
O.node['l']['observed'] = synth['l']
O.node['h']['observed'] = synth['h']
O.node['g']['observed'] = synth['g']
O.node['E_L']['observed'] = synth['E']

draw_net(O, pretty=True)

n_samp = 5000
with pm.Model() as prior_model:
    instantiate_pm(D, evaluate_exprs=True)
    # trace_prior = pm.sample(n_samp)

with pm.Model() as model:
    instantiate_pm(O, evaluate_exprs=True)
    # trace = pm.sample(n_samp)

#
#
#
# lims = [(8.2e-3, 8.6e-3),
#         (0.2e-3, 0.7e-3),
#         (2.5e-3, 2.8e-3),
#         (0.3e-3, 0.6e-3),
#         (1.6e-3, 2.2e-3),
#         (.05e-3, 0.2e-3)]
# trues = [8.5, 0.5, 2.6, 0.5, 1.9, .19]
# varnames=['mu_l', 'sd_l',
#           'mu_h', 'sd_h',
#           'mu_g', 'sd_g']
# varlabs= [r'$\mu_l$', r'$\sigma_l$',
#           r'$\mu_h$', r'$\sigma_h$',
#           r'$\mu_g$', r'$\sigma_g$']
#
# def compare_dist(ax, dat, lim, true=None, scale=1.):
#     sns.distplot(scale*dat, ax=ax, kde=True, label='Posterior', kde_kws={'ls':'--'})
#     a,b = scale*lim[0], scale*lim[1]
#     y = 1./(b-a)
#     ax.hlines(y=y, xmin=a, xmax=b, linestyles=u'solid', label='Prior')
#     ax.vlines([a,b],ymin=0, ymax=y, linestyles=u'solid')
#     if true is not None:
#         ax.axvline(true, color='r')
#     ax.legend()
#
# f, axes = plt.subplots(nrows=3, ncols=2, figsize=(10,15))
#
# for n, ax in enumerate(axes.flat):
#     compare_dist(ax, trace[n_samp//2:].get_values(D.d(varnames[n])), lims[n], true=trues[n], scale=1e3)
#     ax.set_xlabel("{} (x10^{})".format(varlabs[n], -3))
# plt.show()
#
# f2 = plt.figure(figsize=(10,5))
# sns.distplot(trace[n_samp//2:].get_values('E')*1e-4, kde=True, label='Posterior', kde_kws={'ls':'--'})
# sns.distplot(trace_prior[n_samp//2:].get_values('E')*1e-4, label='Prior', kde_kws={'color':'k'})
# plt.legend()
# plt.xlabel(r'$E$ (x10^{})'.format(4))
# # plt.xlim(10,35)
# plt.show()


O.toPMML('testPMML.xml')

bnp = BayesianNetworkParser()

soG = bnp.parse('testPMML.xml')