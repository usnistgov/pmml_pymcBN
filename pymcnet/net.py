import numpy as np
import pymc3 as pm
import networkx as nx
from collections import OrderedDict

class BayesianNetwork(nx.DiGraph):
    """
    keeps track of the order in which nodes are added to the graph,
    and the order in which neighbors are declared
    """
    node_dict_factory = OrderedDict
    adjlist_dict_factory = OrderedDict

    def d(self, n):
        """
        get node distribution by name
        """
        try:
            return self.node[n]['dist']
        except:
            print 'Node either doesn\'t exist or has not been\
            assigned a distribution'
            raise NotImplementedError

    Dist_args = {'observed', 'shape', 'testval'}  # general RV keywords

    def get_args(self, n):
        """
        Dist-specific arguments for PyMC3 RV objects
        :param n: the node needed
        :return: a set of dist-specific required keywords for a corresponding PyMC3 RV
        """
        Norm_args = {'mu', 'sd'}
        HalfNorm_args = {'sd'}
        HalfCauchy_args = {'beta'}
        Expon_args = {'lam'}
        Unif_args = {'lower', 'upper'}
        Poisson_args = {'mu'}
        Det_args = {'var'}
        GRW_args = {'tau'}
        StuT_args = {'nu', 'lam'}
        try:
            dist_type = self.node[n]['dist_type']
        except:
            print """Node either doesn't exist or has not been assigned a distribution"""
            raise NotImplementedError
        # pm.StudentT()
        if dist_type == 'Normal':
            return Norm_args
        elif dist_type == 'HalfNormal':
            return HalfNorm_args
        elif dist_type == 'HalfCauchy':
            return HalfCauchy_args
        elif dist_type == 'Exponential':
            return Expon_args
        elif dist_type in ['DiscreteUniform', 'Uniform']:
            return Unif_args
        elif dist_type == 'Poisson':
            return Poisson_args
        elif dist_type in ['Deterministic', 'Potential']:
            return Det_args
        elif dist_type == 'GaussianRW':
            return GRW_args
        elif dist_type == 'StudentT':
            return StuT_args

        else:
            print "Dists of type {} are not implemented".format(self.node[n]['dist'])
            raise NotImplementedError


def draw_net(D):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print "need matplotlib to plot network..."
    try:
        from networkx.drawing.nx_pydot import graphviz_layout
        pos = graphviz_layout(D, prog='dot')
        plt.figure()
        nx.draw_networkx_nodes(D, pos,
                               nodelist=[n for n in D.nodes() if 'observed' in D.node[n]],
                               with_labels=True, node_size=1000, node_color='yellow')
        nx.draw_networkx_nodes(D, pos,
                               nodelist=[n for n in D.nodes() if not 'observed' in D.node[n]],
                               with_labels=True, node_size=1000, node_color='gray')
        nx.draw_networkx_labels(D, pos)
        nx.draw_networkx_edges(D, pos, arrows=True)
        nx.draw_networkx_edge_labels(D, pos, rotate=False,
                                     edge_labels={k:D.edge[k[0]][k[1]]['var'] for k in D.edges()})
        plt.show()
    except:
        print 'You probably need to install Graphviz or have a working "dot" command in your PATH.'
        raise

def instantiate_pm(D):
    """
    Instantiates a pymc3 model specified by a BN graph object. Must contain a node name with node attributes
    {'dist', [dist_args], 'dist_type'}, where dist_args are distribution-specific keywords a la PyMC3.

    It's important to note the difference between root and child nodes in the current implementation:
    root nodes can be defined statically, but inheriting/child nodes must have their dist_args defined as
    argument-less functions (e.g. using mu=lambda: mu_x, sd=lambda: sd_x) so that they may be called at
    instantiation.

    Currently this system uses set logic to narrow down which keywords to allow for a given node, which
    is somewhat of a non-pythonic method (hard-to-read). The ability to infer keywords by the parent nodes
    (and so allow kwargs in the dist_arg functions) is broken due to NX's handling of the predecessors()
    ordering. WIP.

    :param D: a BN object inheriting from nx.DiGraph(). Contains node definitions
    :return: Instantiated PyMC3 model object
    """
    for n in D.nodes():
        print n
        if not D.predecessors(n):  # check if the node is a root (implicit booleanness of empty set)
            varset = D.node[n].viewkeys() & (D.get_args(n) | D.Dist_args)
            args = {k: D.node[n][k] for k in varset}
            print 'root node; keys: ', args.keys()
            D.node[n]['dist'] = D.node[n]['dist'](n, **args)
        else:
            args = {k: D.node[n][k] for k in D.node[n].viewkeys() & D.Dist_args}
            #             print 'pre-args: ',args
            for var in list(D.get_args(n)):  # a set for each unique edge functional relationship
                #                 parents = [i for i in D.predecessors(n)[::-1] if D.edge[i][n]['var']==var]
                #                 print var, parents
                ### ORDER OF NODE DEFINITION MUST be the ORDER OF ARGUMENTS
                #                 args.update({var: D.node[n][var](*parents)})
                args.update({var: D.node[n][var]()})
            print 'child node; keys: ', args.keys()
            D.node[n]['dist'] = D.node[n]['dist'](n, **args)

