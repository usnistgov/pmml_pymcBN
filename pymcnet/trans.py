"""
Taken largely from a GPR translator by Max Ferguson (2016)

PROBABLY should move these to be class methods of BayesianNetwork() under pymcnet.net
"""

import numpy as np
import datetime
import getpass
from lxml import etree as ET
from lxml.etree import Element,SubElement


def toPMML( filename):
    """Write the trained model to PMML. Return PMML as string"""
    # X = self.xTrain;
    # Y = self.yTrain;
    # gamma = self.gamma
    # nugget = self.nugget
    # k_lambda = self.k_lambda
    copywrite = "DMG.org"
    # xrow, yrow, xcol, ycol = trans_get_dimension(X, Y)
    featureName, targetName = None, None #trans_name(xcol, ycol)
    # Start constructing the XML Tree
    PMML = trans_root(None, copywrite, None)
    PMML = trans_dataDictionary(PMML, featureName, targetName)
    BNM = trans_BN(PMML)
    BNM = trans_miningSchema(BNM, featureName, targetName)
    # GPM = trans_output(GPM)
    # GPM = trans_kernel(GPM, k_lambda, nugget, gamma, xcol, 'squared_exponential')
    # GPData = trans_traininginstances(GPM, xrow, xcol + ycol)
    # trans_instancefields(GPData, featureName, targetName)
    # trans_inlinetable(GPData, featureName, targetName, X, Y)
    # Write the tree to file
    tree = ET.ElementTree(PMML)
    tree.write(filename, pretty_print=True, xml_declaration=True, encoding="utf-8")
    print 'Wrote PMML file to %s' % filename

def trans_root(description,copyright,Annotation):
    """Some basic information about the document """
    username = str(getpass.getuser())
    py_version = "0.1"

    PMML_version = "4.3"
    xmlns = "http://www.dmg.org/PMML-4_2"
    PMML = root = Element('pmml',xmlns=xmlns, version=PMML_version)

    # pmml level
    if copyright is None:
        copyright = "Copyright (c) 2015 {0}".format(username)
    if description is None:
        description = "Bayesian Network Model"
    Header = SubElement(PMML, "header", copyright=copyright, description=description)

    if Annotation is not None:
        ET.Element(Header,"Annotation").text=Annotation
    return PMML

def trans_dataDictionary(PMML,nodeName):
    """
    DataField level
    TODO: make fit BN style
    """
    toStr = "{0}".format
    DataDictionary = SubElement(PMML, "datadictionary", numberoffields=toStr(len(nodeName)))
    # for it_name in featureName:
    #     SubElement(DataDictionary, "datafield", name=it_name,optype="continuous", datatype="double" )
    #
    # for it_name in targetName:
    #     SubElement(DataDictionary, "datafield", name=it_name,optype="continuous", datatype="double" )

    return PMML


def trans_BN(PMML):
    """Create BayesianNetworkModel level"""
    BayesianNetworkModel = SubElement(PMML, "BayesianNetworkModel")
    BayesianNetworkModel.set("modelName", "Bayesian Network Model")
    BayesianNetworkModel.set("functionName", "regression")
    return BayesianNetworkModel


def trans_miningSchema(BayesianNetworkModel,featureName,targetName):
    """Create Mining Schema"""
    MiningSchema = SubElement(BayesianNetworkModel,"miningschema")
    for it_name in featureName:
        SubElement(MiningSchema, "MiningField", name=it_name, usagetype="active")

    for it_name in targetName:
        SubElement(MiningSchema, "MiningField", name=it_name, usagetype="target")

    return BayesianNetworkModel


def trans_networkNodes(BayesianNetworkModel):
    """Create Node Level"""
    # MiningSchema = SubElement(BayesianNetworkModel,"miningschema")
    # for it_name in featureName:
    #     SubElement(MiningSchema, "MiningField", name=it_name, usagetype="active")
    #
    # for it_name in targetName:
    #     SubElement(MiningSchema, "MiningField", name=it_name, usagetype="target")

    return BayesianNetworkModel
