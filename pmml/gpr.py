"""
Taken from a GPR translator by Max Ferguson (2016)
"""

import numpy as np
import lxml.etree as ET
# from models.gpr import GaussianProcessModel

class GaussianProcessParser():

    def __init__(self):
        """Create a GaussianProcess Parser that can parse PMML files"""
        self.nsp = "{http://www.dmg.org/PMML-4_3}";


    def parse(self,filename):
        """Parse a Gaussian Process PMML file. Return a GaussianProcessModel"""
        GPM = self._parse_GPM(filename)
        featureName,targetName = self._parse_name(GPM)
        kernelName,k_lambda,nugget,gamma = self._parse_kernel(GPM)
        xTrain,yTrain = self._parse_training_values(GPM)
        xTrain = np.array(xTrain)
        yTrain = np.array(yTrain)
        return None #GaussianProcessModel(gamma=gamma,beta=0,nugget=nugget,k_lambda=k_lambda,
            #kernelName=kernelName,xTrain=xTrain,yTrain=yTrain)


    def _findElementByName(self,root,tagname):
        """Find an element by Name. Throw an error if the element does not exist"""
        element = root.find(self.nsp+tagname)
        if element is None:
            raise MissingElementError("Missing tag %s"%tagname)
        return element

    def _parse_GPM(self,filename):
        """Return the PMML document as an etree element"""
        tree = ET.parse(filename)
        root = tree.getroot()
        return self._findElementByName(root,"GaussianProcessModel");
        

    def _parse_name(self,GPM):
        """parse MiningSchema for features and targets"""
        # Will get a list of target name and feature name
        tagname = "MiningSchema"
        MS = self._findElementByName(GPM,tagname)
        targetName = []
        featureName = []
        for MF in MS:
            MF_name = MF.attrib["name"]
            MF_type = MF.attrib["usageType"]
            if MF_type == "active":
                featureName.append(MF_name)
            elif MF_type == "predicted":
                targetName.append(MF_name)

        return featureName,targetName


    def _parse_kernel(self,GPM):
        """Return kernel parameters"""
        kernels = ["ARDSquaredExponentialKernel","AbsoluteExponentialKernel"]
        kernel = None

        for kernelName in kernels:
            try:
                kernel = self._findElementByName(GPM,kernelName)
                break
            except MissingElementError:
                pass

        if kernel is None:
            raise MissingElementError("Unable to find valid kernel tag")

        nugget = float(kernel.attrib["noiseVariance"])
        gamma = float(kernel.attrib["gamma"])
        lamb = self._findElementByName(kernel,"Lambda")
        array = self._findElementByName(lamb,"array").text
        array = array.strip()
        k_lambda = [float(i) for i in array.split(" ")]
        hasKernel = True

        return kernelName,k_lambda,nugget,gamma


    def _parse_training_values(self,GPM):
        """Return the training values"""
        traininginstances = self._findElementByName(GPM,"TrainingInstances")
        inlinetable = self._findElementByName(traininginstances,"InlineTable")
        instancefields = self._findElementByName(traininginstances,"InstanceFields")

        [features,targets] = self._parse_name(GPM)

        nrows = int(traininginstances.attrib['recordCount'])
        fcols = len(features)
        tcols = len(targets)

        xTrain = np.zeros([nrows,fcols]);
        yTrain = np.zeros([nrows,tcols]);

        for i,row in enumerate(inlinetable.findall(self.nsp+"row")):
            for j,featureName in enumerate(features):
                col = row.find(self.nsp+featureName)
                xTrain[i][j] = float(col.text)

            for j,featureName in enumerate(targets):
                col = row.find(self.nsp+featureName)
                yTrain[i][j] = float(col.text)

        return xTrain,yTrain



class MissingElementError(Exception):
    """Thrown when an element is not found"""
    pass
