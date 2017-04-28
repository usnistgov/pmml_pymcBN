import numpy as np
import lxml.etree as et
import networkx as nx

class BayesianNetworkParser():
    def __init__(self):
        """Create a BayesianNetwork Parser that can parse PMML files"""
        self.nsp = "{http://www.dmg.org/PMML-4_3}";

    def parse(self, filename):
        """Parse a Bayesian Network PMML file. Return a BayesianNetwork DiGraph object"""
        BN = self._parse_BN(filename)
        featureName, targetName = self._parse_name(BN)
        graph = self._parse_graph(BN)
        for observed in featureName:
            graph.node[observed]['observed']="PLACEHOLDER"
        # kernelName,k_lambda,nugget,gamma = self._parse_kernel(GPM)
        # xTrain,yTrain = self._parse_training_values(GPM)
        # xTrain = np.array(xTrain)
        # yTrain = np.array(yTrain)
        # return GaussianProcessModel(gamma=gamma,beta=0,nugget=nugget,k_lambda=k_lambda,
        #     kernelName=kernelName,xTrain=xTrain,yTrain=yTrain)
        # return {'bn': BN, 'feature': featureName, 'target': targetName}
        return graph

    def _parse_BN(self, filename):
        """Return the PMML document as an etree element"""
        tree = et.parse(filename)
        root = tree.getroot()
        # print root[2].tag, root
        return self._findElementByName(root, "BayesianNetworkModel")

    def _findElementByName(self, root, tagname):
        """Find an element by Name. Throw an error if the element does not exist"""
        element = root.find(self.nsp + tagname)
        if element is None:
            raise MissingElementError("Missing tag %s" % tagname)
        return element

    def _parse_name(self, BN):
        """parse MiningSchema for features and targets"""
        # Will get a list of target name and feature name
        tagname = "MiningSchema"
        MS = self._findElementByName(BN, tagname)
        targetName = []
        featureName = []
        for MF in MS:
            MF_name = MF.attrib["name"]
            MF_type = MF.attrib["usageType"]
            if MF_type == "active":
                featureName.append(MF_name)
            elif MF_type == "target":
                targetName.append(MF_name)

        return featureName, targetName

    def _parse_graph(self, BN):
        from pymcnet.net import BayesianNetwork


        dist_dic = {
            'UniformDistributionForBN': {
                'dist_type': 'Uniform',
                'vars': {
                    'Lower': 'lower',
                    'Upper': 'upper'
                }
            },
            'NormalDistributionForBN': {
                'dist_type': 'Normal',
                'vars': {
                    'Mean': 'mu',
                    'StDev': 'sd'
                }
            },
            'DeterministicBN': {
                'dist_type': 'Deterministic',
                'vars': {
                    'StaticValue': 'var'
                }
            }
        }


        G = BayesianNetwork()

        tagname = "BayesianNetworkNodes"

        BNN = self._findElementByName(BN, tagname)



        for elem in BNN.iterchildren():
            # DETERMINISTIC_flag = False
            n = elem.attrib['name']
            G.add_node(n)

            dist_elem = elem[0][0]
            dist_name = dist_elem.tag.replace(self.nsp,'')
            print n
            G.node[n]['dist_type'] = dist_dic[dist_name]['dist_type']
            node_refs = elem.findall(".//"+self.nsp+'FieldRef')
            # print [i.attrib['field'] for i in node_refs]

            for varname, repl in dist_dic[dist_name]['vars'].items():

                BNNVar = self._findElementByName(dist_elem, varname)

                # if DETERMINISTIC_flag:
                #     G.node[n]['exprs']['var'] = et.tostring(BNNVar)
                #     continue

                var_refs = BNNVar.findall(".//"+self.nsp+'FieldRef')  # no fieldref => root node
                if not var_refs+node_refs:
                    var_val = BNNVar.findtext(self.nsp+'Constant')
                    print '\t' + repl, var_val

                    #######Hack for Det Nodes########
                    # if float(var_val)==0.:
                    #     G.node[n]['dist_type'] = 'Deterministic'
                    #     DETERMINISTIC_flag = True
                    #     continue
                    #################################

                    G.node[n][repl] = float(var_val)
                else:
                    print '\t' + repl, [i.attrib['field'] for i in var_refs]

                    if not 'exprs' in G.node[n]:
                        G.node[n]['exprs'] = dict()

                    G.node[n]['exprs'][repl] = "PLACEHOLDER"
                    ref_names = [i.attrib['field'] for i in var_refs]
                    # print ref_names
                    G.add_edges_from([(i, n) for i in ref_names], var=repl)
                    pass

        return G


class MissingElementError(Exception):
    """Thrown when an element is not found"""
    pass

#
# class Node():
#
#     def __init__(self, name, ):
#
#
#
#
#
#
#

