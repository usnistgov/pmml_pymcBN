import numpy as np
import pymc3 as pm
from pymc3.math import *
import networkx as nx
from collections import OrderedDict
from lxml import etree as ET
from lxml.etree import Element,SubElement
from .oset import OrderedSet
import getpass
import re
from functools import reduce

class BayesianNetwork(nx.DiGraph):
    """
    - keeps track of the order in which nodes are added to the graph,
      and the order in which neighbors are declared

    - Adds method to serialize supported models as PMML BayesianNetworkModel's
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
            print('Node either doesn\'t exist or has not been\
            assigned a distribution')
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
        Unif_args = OrderedSet(['lower', 'upper'])
        Poisson_args = {'mu'}
        Det_args = {'var'}
        GRW_args = {'tau'}
        StuT_args = {'nu', 'lam'}
        logN_args = {'mu', 'sd'}

        try:
            dist_type = self.node[n]['dist_type']
        except:
            print("""Node either doesn't exist or has not been assigned a distribution""")
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
        elif dist_type == 'GaussianRandomWalk':
            return GRW_args
        elif dist_type == 'StudentT':
            return StuT_args
        elif dist_type == 'Lognormal':
            return logN_args

        else:
            print("Dists of type {} are not implemented".format(self.node[n]['dist_type']))
            raise NotImplementedError


    def toPMML(self, filename):
        """Write the trained model to PMML."""

        def trans_root(description=None, copyright=None, annotation=None):
            """Some basic information about the document """
            username = str(getpass.getuser())
            py_version = "0.1"

            PMML_version = "4.3"
            xmlns = "http://www.dmg.org/PMML-4_3"
            PMML = root = Element('PMML', xmlns=xmlns, version=PMML_version)

            # pmml level
            if copyright is None:
                copyright = "Copyright (c) 2015 {0}".format(username)
            if description is None:
                description = "Bayesian Network Model"
            Header = SubElement(PMML, "Header", copyright=copyright, description=description)

            if annotation is not None:
                ET.Element(Header, "Annotation").text = annotation
            return PMML

        def trans_dataDictionary(PMML):
            """
            DataField level
            TODO: make fit BN style
            """
            toStr = "{0}".format
            node_list = self.nodes()
            DataDictionary = SubElement(PMML, "DataDictionary", numberOfFields=toStr(len(node_list)))

            # only continuous supported currently
            for node_name in node_list:
                SubElement(DataDictionary, "DataField", name=node_name,optype="continuous", dataType="double")

            # for it_name in targetName:
            #     SubElement(DataDictionary, "datafield", name=it_name,optype="continuous", datatype="double" )

            return PMML

        def trans_miningSchema(BayesianNetworkModel):
            """
            Create Mining Schema
            """
            active_nodes = [n for n in self.nodes() if 'observed' in self.node[n]]
            target_nodes = [n for n in self.nodes() if not 'observed' in self.node[n]]
            MiningSchema = SubElement(BayesianNetworkModel, "MiningSchema")

            for node_name in active_nodes:
                SubElement(MiningSchema, "MiningField", name=node_name, usageType="active")

            for it_name in target_nodes:
                SubElement(MiningSchema, "MiningField", name=it_name, usageType="target")

            return BayesianNetworkModel

        def trans_BN(PMML):
            """Create BayesianNetworkModel level"""
            BayesianNetworkModel = SubElement(PMML, "BayesianNetworkModel")
            BayesianNetworkModel.set("modelName", "Bayesian Network Model")
            BayesianNetworkModel.set("functionName", "regression")
            return BayesianNetworkModel

        def trans_map(node_data):
            dic = {
                'Uniform': 'UniformDistributionForBN',
                'Normal': 'NormalDistributionForBN',
                # 'Deterministic': 'DeterministicBN',
                'Deterministic': 'NormalDistributionForBN',  # hack
                'Lognormal': 'LognormalDistributionForBN',
                'lower': 'Lower',
                'upper': 'Upper',
                'mu': 'Mean',
                'sd': 'Variance',  # Uh-oh
                # 'var': 'StaticValue'
                'var': 'Mean'  # hack
                # 'var': 'VALUE_OF_DETERMINISTIC_NODE'
            }
            return dic[node_data]

        def trans_networkNodes(BayesianNetworkModel):
            """Create Node Levels"""
            from .expr_parse import get_det_node_xml
            from sympy import symbols

            symbols(self.nodes())

            BayesianNetworkNodes = SubElement(BayesianNetworkModel, "BayesianNetworkNodes")

            for node in self.nodes_iter(data=True):
                nodeDef = SubElement(BayesianNetworkNodes, "ContinuousNode", name=node[0])
                distDef = SubElement(nodeDef, "ContinuousDistribution")
                nodeDist = SubElement(distDef, trans_map(node[1]['dist_type']))
                toStr = "{0}".format
                print(node[0])

                ##### a hack to get deterministic nodes to work #####
                if node[1]['dist_type'] == 'Deterministic':
                    variance = SubElement(nodeDist, 'Variance')
                    value = SubElement(variance, 'Constant', dataType="double")
                    value.text = toStr(0.)
                #####################################################

                for varname in self.get_args(node[0]):

                    vardef = SubElement(nodeDist, trans_map(varname))

                    if isinstance(node[1][varname], (int, float, complex)):

                        value = SubElement(vardef, 'Constant', dataType="double")
                        if varname == 'sd':  # need variance, not SD
                            val = float(node[1][varname])**2
                            value.text = toStr(val)
                            print('\t', trans_map(varname), toStr(val))
                        else:
                            val = float(node[1][varname])
                            value.text = toStr(val)
                            print('\t', trans_map(varname), toStr(val))

                    else:
                        if varname == 'sd':  # need variance, not SD
                            func = lambda x: '({0})**2.'.format(x)
                            vardef.append(ET.fromstring(get_det_node_xml(node, varname,
                                                                         func=func)))
                            print('\t', trans_map(varname), func(node[1]['exprs'][varname]))
                        else:
                            vardef.append(ET.fromstring(get_det_node_xml(node, varname)))
                            print('\t', trans_map(varname), node[1]['exprs'][varname])
                        # value = SubElement(vardef, 'Placeholder', dataType="N/A")
                        # vardef.append(ET.fromstring(get_det_node_xml(node, varname)))
            return BayesianNetworkModel

        cw = "DMG.org"

        # Start constructing the XML Tree
        PMML = trans_root(copyright=cw)
        PMML = trans_dataDictionary(PMML)
        BNM = trans_BN(PMML)
        BNM = trans_miningSchema(BNM)
        BNM = trans_networkNodes(BNM)

        # Write the tree to file
        tree = ET.ElementTree(PMML)
        tree.write(filename, pretty_print=True, xml_declaration=False, encoding="utf-8")
        print('Wrote PMML file to {}'.format(filename))


def draw_net(D, pretty=False):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("need matplotlib to plot network...")

    try:
        from networkx.drawing.nx_pydot import graphviz_layout
        pos = graphviz_layout(D, prog='dot')
        plt.figure(figsize=(12,6))

        node_colors = ['#d3d3d3' if ('observed' in D.node[n]) else 'white' for n in D.nodes()]
        node_shapes = ['s' if (D.node[n]['dist_type'] == 'Deterministic') else 'o' for n in D.nodes()]
        node_edges = [{
            'Deterministic': '#1e90ff',
            'Normal': '#f89406',
            'Uniform': '#46a546'
        }.get(D.node[n]['dist_type'], 'k') for n in D.nodes()]
        for _n, _s, _c, _e, _p in zip(D.nodes(), node_shapes, node_colors, node_edges, [{i: pos[i]} for i in D.nodes()]):
            draw_nodes = nx.draw_networkx_nodes(D, _p,
                                                nodelist=[_n],
                                                with_labels=True,
                                                node_size=1500,
                                                node_color=_c,
                                                node_shape=_s)
            draw_nodes.set_edgecolor(_e)

        nx.draw_networkx_edges(D, pos, arrows=True)

        if pretty:
            try:
                from sympy import Symbol, latex
                from matplotlib import rc_context
                # rc("font", family="serif", size=12)
                # rc("text", usetex=True)
                with rc_context(rc={'text.usetex': True,
                                    'font.family': 'serif',
                                    'font.size': 16}):
                    repls = ('lam', 'lambda'), ('sd', 'sigma'), ('var', 'x')  # latexify some pymc3 vars
                    nx.draw_networkx_labels(D, pos, labels=dict(
                        (n, r'${}$'.format(latex(Symbol(reduce(lambda a, kv: a.replace(*kv), repls, n))))) for n in D.nodes()))
                    nx.draw_networkx_edge_labels(D, pos, rotate=False,
                                                 label_pos=.7,
                                                 edge_labels={(k[0], k[1]): r'${}$'.format(latex(
                                                     Symbol(reduce(lambda a, kv: a.replace(*kv), repls, k[2]['var']))))
                                                              for k in D.edges_iter(data=True)})
            except ImportError:
                "need sympy and LaTeX for pretty variables"
                raise

        else:
            nx.draw_networkx_labels(D, pos)
            nx.draw_networkx_edge_labels(D, pos, rotate=False,
                                         label_pos=.7,
                                         edge_labels={k:D.edge[k[0]][k[1]]['var'] for k in D.edges()})
        # plt.legend(loc=0)
        plt.gca().axis('off')
        plt.show()

    except:
        print('You probably need to install Graphviz or have a working "dot" command in your PATH.')
        raise


def instantiate_pm(D, evaluate_exprs=False):
    """
    Instantiates a pymc3 model specified by a BN graph object. Must contain a node name with node attributes
    {'exprs', 'dist_type'}, where exprs is a dict containing string expressions for the mathematical
    definitions of the needed RV parameters of some pymc3 distribution, 'dist_type'.

    It's important to note the difference between root and child nodes in the current implementation:
    root nodes can be defined statically, but inheriting/child nodes by design use one-time created
    argument-less functions (e.g. using mu=lambda: mu_x, sd=lambda: sd_x) so that they may be called at
    instantiation. This will be (scarily) done with eval until another way can be found.

    Currently this system uses set logic to narrow down which keywords to allow for a given node, which
    is somewhat of a non-pythonic method (hard-to-read). The ability to infer keywords by the parent nodes
    (and so allow kwargs in the dist_arg functions) is broken due to NX's handling of the predecessors()
    ordering. WIP.

    :param D: a BN object inheriting from nx.DiGraph(). Contains node definitions
    :return: Instantiated PyMC3 model object
    """
    def regex_repl(expression):
        """
        Since the internal node-refs are done using D.d(node), which needs to reference some node's
        associated PyMC3 RandomVariable object, I can't figure out a good way to parse some kind of
        instruction at instantiation, other than input strings and use eval. (D is BayesianNet)

        This function uses regex to find node variables in an expression and replace them with the
        RV reference in that node, gracefully ignoring subscript instances (ignores matches starting
        with "_".

        :param expression: expression string to be modified
        :return: string containing uses of D.d("var") instead of "var".
        """
        r = re.compile(r"(?<!_)(" + '|'.join(D.nodes()) + r")")
        eval_str = r.sub('D.d(\'\\1\')', expression)
        # print eval_str
        return eval_str

    for n in D.nodes():
        print(n)
        if not D.predecessors(n):  # check if the node is a root (implicit booleanness of empty set)
            varset = D.node[n].keys() & (D.get_args(n) | D.Dist_args)
            args = {k: D.node[n][k] for k in varset}
            print('root node; keys: ', list(args.keys()))
            # D.node[n]['dist'] = D.node[n]['dist'](n, **args)
            D.node[n]['dist'] = getattr(pm, D.node[n]['dist_type'])(n, **args)
        else:
            # need to define value functions for each RV argument

            args = {k: D.node[n][k] for k in D.node[n].keys() & D.Dist_args}
            #             print 'pre-args: ',args
            for var in list(D.get_args(n)):  # a set for each unique edge functional relationship
               ### DANGER ZONE ###
                if evaluate_exprs:  # allow users to write custom lambda functions in nodes
                    if var in list(D.node[n]['exprs'].keys()):
                        D.node[n][var] = lambda: eval(regex_repl(D.node[n]['exprs'][var]))
               ###################
                args.update({var: D.node[n][var]()})
            print('child node; keys: ', list(args.keys()))
            # D.node[n]['dist'] = D.node[n]['dist'](n, **args)
            try:
                D.node[n]['dist'] = getattr(pm, D.node[n]['dist_type'])(n, **args)
            except AttributeError:
                print("The distribution you want must not be in the standard PyMC3 list...")
                print("going to try a time-series dist...")
                try: D.node[n]['dist'] = getattr(pm.distributions.timeseries,
                                                 D.node[n]['dist_type'])(n, **args)
                except AttributeError:
                    print("others are NOT supported at this time")
                    raise


