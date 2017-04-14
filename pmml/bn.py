import numpy as np
import lxml.etree as et
import networkx as nx

class BayesianNetworkParser():
    def __init__(self):
        """Create a GaussianProcess Parser that can parse PMML files"""
        self.nsp = "{http://www.dmg.org/PMML-4_3}";

    def parse(self, filename):
        """Parse a Gaussian Process PMML file. Return a GaussianProcessModel"""
        BN = self._parse_BN(filename)
        featureName, targetName = self._parse_name(BN)
        graph = self._parse_graph(BN)
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

        G = nx.DiGraph()

        tagname = "BayesianNetworkNodes"
        nodelist = self._findElementByName(BN, tagname)

        node_types = ['ContinuousNode', 'DiscreteNode']
        node = None
        discrete_nodes = nodelist.findall(self.nsp + 'DiscreteNode')  # unsupported
        continuous_nodes = nodelist.findall(self.nsp + 'ContinuousNode')
        print(continuous_nodes)
        for node in continuous_nodes:
            print(node.attrib['name'])
            G.add_node(node.attrib['name'])
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

