import numpy as np
import pymc3 as pm
import networkx as nx
from collections import OrderedDict
from lxml import etree as ET
from lxml.etree import Element,SubElement
import getpass

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


    def toPMML(self, filename):
        """Write the trained model to PMML."""

        def trans_root(description=None, copyright=None, annotation=None):
            """Some basic information about the document """
            username = str(getpass.getuser())
            py_version = "0.1"

            PMML_version = "4.3"
            xmlns = "http://www.dmg.org/PMML-4_2"
            PMML = root = Element('pmml', xmlns=xmlns, version=PMML_version)

            # pmml level
            if copyright is None:
                copyright = "Copyright (c) 2015 {0}".format(username)
            if description is None:
                description = "Bayesian Network Model"
            Header = SubElement(PMML, "header", copyright=copyright, description=description)

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
            DataDictionary = SubElement(PMML, "DataDictionary", numberoffields=toStr(len(node_list)))

            # only continuous supported currently
            for node_name in node_list:
                SubElement(DataDictionary, "DataField", name=node_name,optype="continuous", datatype="double")

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
                SubElement(MiningSchema, "MiningField", name=node_name, usagetype="active")

            for it_name in target_nodes:
                SubElement(MiningSchema, "MiningField", name=it_name, usagetype="target")

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
                'Deterministic': 'NormalDistributionForBN',
                # 'Deterministic': 'DETERMINISTIC_NODE_NEEDED',
                'lower': 'Lower',
                'upper': 'Upper',
                'mu': 'Mean',
                'sd': 'Variance',
                'var': 'Mean'
                # 'var': 'VALUE_OF_DETERMINISTIC_NODE'
            }
            return dic[node_data]

        def trans_networkNodes(BayesianNetworkModel):
            """Create Node Levels"""
            from expr_parse import get_det_node_xml
            from sympy import symbols

            symbols(self.nodes())

            BayesianNetworkNodes = SubElement(BayesianNetworkModel, "BayesianNetworkNodes")

            for node in self.nodes_iter(data=True):
                nodeDef = SubElement(BayesianNetworkNodes, "ContinuousNode", name=node[0])
                distDef = SubElement(nodeDef, "ContinuousDistribution")
                nodeDist = SubElement(distDef, trans_map(node[1]['dist_type']))
                toStr = "{0}".format
                print node[0]

                ##### a hack to get deterministic nodes to work #####
                if node[1]['dist_type'] == 'Deterministic':
                    variance = SubElement(nodeDist, 'Variance')
                    value = SubElement(variance, 'Constant', dataType="double")
                    value.text = toStr(0.)
                #####################################################

                for varname in self.get_args(node[0]):

                    vardef = SubElement(nodeDist, trans_map(varname))

                    if isinstance(node[1][varname], (int, long, float, complex)):
                        print '\t', varname, toStr(float(node[1][varname]))
                        value = SubElement(vardef, 'Constant', dataType="double")
                        if varname == 'sd':  # need variance, not SD
                            val = float(node[1][varname])**2
                            value.text = toStr(val)
                        else:
                            val = float(node[1][varname])
                            value.text = toStr(val)
                    else:

                        print '\t', varname, node[1]['exprs'][varname]
                        # value = SubElement(vardef, 'Placeholder', dataType="N/A")
                        vardef.append(ET.fromstring(get_det_node_xml(node, varname)))
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
        tree.write(filename, pretty_print=True, xml_declaration=True, encoding="utf-8")
        print 'Wrote PMML file to {}'.format(filename)


def draw_net(D, pretty=False):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print "need matplotlib to plot network..."

    try:
        from networkx.drawing.nx_pydot import graphviz_layout
        pos = graphviz_layout(D, prog='dot')
        plt.figure(figsize=(12,6))
        nx.draw_networkx_nodes(D, pos,
                               with_labels=True, node_size=1500, node_color='gray')
        nx.draw_networkx_nodes(D, pos,
                               nodelist=[n for n in D.nodes() if 'observed' in D.node[n]],
                               with_labels=True, node_size=1500, node_color='#FFBF00')
        nx.draw_networkx_nodes(D, pos,
                               nodelist=[n for n in D.nodes() if D.node[n]["dist_type"]=='Deterministic'],
                               with_labels=True, node_size=1500, node_color='#578FA4')
        nx.draw_networkx_edges(D, pos, arrows=True)

        if pretty:
            try:
                from sympy import Symbol, latex
            except ImportError:
                "need sympy for pretty variables"
                raise
            repls = ('lam', 'lambda'), ('sd', 'sigma')  # latexify some pymc3 vars
            nx.draw_networkx_labels(D, pos, labels=dict((n, r'${}$'.format(latex(Symbol(reduce(lambda a, kv: a.replace(*kv), repls, n))))) for n in D.nodes()))
            nx.draw_networkx_edge_labels(D, pos, rotate=False,
                                         edge_labels={(k[0],k[1]): r'${}$'.format(latex(Symbol(reduce(lambda a, kv: a.replace(*kv), repls, k[2]['var'])))) for k in D.edges_iter(data=True)})
        else:
            nx.draw_networkx_labels(D, pos)
            nx.draw_networkx_edge_labels(D, pos, rotate=False,
                                         edge_labels={k:D.edge[k[0]][k[1]]['var'] for k in D.edges()})
        plt.gca().axis('off')
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

