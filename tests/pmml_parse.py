from pmml.bn import BayesianNetworkParser

bnp = BayesianNetworkParser()
soG = bnp.parse('../WeldModelPMML.xml')
print(soG.node)