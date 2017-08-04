
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
l_obs = norm(8.5, 0.5).rvs(size=100)*1e-3
h_obs = norm(2.6, 0.5).rvs(size=100)*1e-3
g_obs = norm(1.9, .19).rvs(size=100)*1e-3
synth = pd.DataFrame(data=np.array([l_obs, h_obs, g_obs]).T,
                     columns=['l','h','g'])
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
e=.011
# D.add_node('L',
#            mu=500e-3, sd=10e-3,
#            dist_type='Normal')
L=.500
# D.add_node('g',
#            mu=2e-3, sd=0.1e-3,
#            dist_type='Normal')
t=.015
# D.add_node('t',
#            mu=15e-3, sd=0.5e-3,
#            dist_type='Normal')

D.add_node('V',
           dist_type='Deterministic',
           exprs = {'var':f'{L}*((0.75)*mu_l*mu_h + mu_g*{t} + (mu_l-mu_g)*({t}-{e})*0.5)'})
# D.add_node('sd_V',
#            dist_type='Uniform',
#            lower=1e-3, upper=1e3)
# D.add_node('V',
#            dist_type='Normal',
#            exprs = {'mu':f'{L}*((0.75)*mu_l*mu_h + mu_g*{t} + (mu_l-mu_g)*({t}-{e})*0.5)',
#                     'sd': 'sd_V'}
# #                     'sd': '0.0001'}  # close to deterministic
#                     )

D.add_edges_from([(i,'V') for i in ['mu_l','mu_h','mu_g']],
                 var='x')
# D.add_edge('sd_V', 'V', var='sd')

D.add_node('rho',
           mu=8250., sd=10.,
          dist_type='Normal')
# D.add_node('H',
#            mu=270e3, sd=3e3,
#            dist_type='Normal')
H=270.
# D.add_node('C_p',
#            mu=500., sd=5.,
#            dist_type='Normal')
C_p=500.
# D.add_node('T_i',
#            mu=303., sd=.3,
#            dist_type='Normal'),
T_i=300.
# D.add_node('T_f',
#            mu=1628., sd=10.,
#            dist_type='Normal')
T_f=1600.

D.add_node('E',
           dist_type = 'Deterministic',
           exprs={'var':f'rho*V*({C_p}*({T_f}-{T_i}) + {H})'})
# D.add_node('sd_E',
#            dist_type = 'Uniform',
#            lower=1e-3, upper=1e3)
# D.add_node('E',
#            dist_type = 'Normal',
#            exprs={'mu':f'rho*V*({C_p}*({T_f}-{T_i}) + {H})',
#                   'sd': 'sd_E'}
# #                   'sd': '0.0001'}  # close to deterministic
#           )

D.add_edges_from([(i, 'E') for i in ['rho','V']], var='var')
# D.add_edge('sd_E', 'E', var='sd')



O = D.copy()
O.node['l']['observed'] = synth['l']
O.node['h']['observed'] = synth['h']
O.node['g']['observed'] = synth['g']

draw_net(O, pretty=True)

n_samp = 500
with pm.Model() as prior_model:
    instantiate_pm(D, evaluate_exprs=True)
    trace_prior = pm.sample(n_samp, njobs=4)

with pm.Model() as model:
    instantiate_pm(O, evaluate_exprs=True)
    trace = pm.sample(n_samp, njobs=4)




lims = [(8.2e-3, 8.6e-3),
        (0.2e-3, 0.7e-3),
        (2.5e-3, 2.8e-3),
        (0.3e-3, 0.6e-3),
        (1.6e-3, 2.2e-3),
        (.05e-3, 0.2e-3)]
trues = [8.5, 0.5, 2.6, 0.5, 1.9, .19]
varnames=['mu_l', 'sd_l',
          'mu_h', 'sd_h',
          'mu_g', 'sd_g']
varlabs= [r'$\mu_l$', r'$\sigma_l$',
          r'$\mu_h$', r'$\sigma_h$',
          r'$\mu_g$', r'$\sigma_g$']

def compare_dist(ax, dat, lim, true=None, scale=1.):
    sns.distplot(scale*dat, ax=ax, kde=True, label='Posterior', kde_kws={'ls':'--'})
    a,b = scale*lim[0], scale*lim[1]
    y = 1./(b-a)
    ax.hlines(y=y, xmin=a, xmax=b, linestyles=u'solid', label='Prior')
    ax.vlines([a,b],ymin=0, ymax=y, linestyles=u'solid')
    if true is not None:
        ax.axvline(true, color='r')
    ax.legend()

f, axes = plt.subplots(nrows=3, ncols=2, figsize=(10,15))

for n, ax in enumerate(axes.flat):
    compare_dist(ax, trace[n_samp//2:].get_values(D.d(varnames[n])), lims[n], true=trues[n], scale=1e3)
    ax.set_xlabel("{} (x10^{})".format(varlabs[n], -3))
plt.show()

f2 = plt.figure(figsize=(10,5))
sns.distplot(trace[n_samp//2:].get_values('E')*1e-4, kde=True, label='Posterior', kde_kws={'ls':'--'})
sns.distplot(trace_prior[n_samp//2:].get_values('E')*1e-4, label='Prior', kde_kws={'color':'k'})
plt.legend()
plt.xlabel(r'$E$ (x10^{})'.format(4))
# plt.xlim(10,35)
plt.show()


O.toPMML('testPMML.xml')

bnp = BayesianNetworkParser()

soG = bnp.parse('testPMML.xml')