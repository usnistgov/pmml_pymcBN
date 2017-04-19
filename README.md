# BayesNetMC

## Why?

This is a python package that extends PyMC3 to enable the definition of Bayesian Network models
without immediate compilation in Theano, using a NetworkX  directed graph to store model information.

Additionally, this package allows the serialization of a pymc3-style bayes-net to the PMML format, at
least for supported node types. Currently, only continuous nodes are supported, though in theory any
distribution allowed by PyMC3 could be added in the future. This parsing is done with the help of SymPy
to allow complex node function definitions as combinations of other node values (i.e. inheritance).


## Usage
Return a Model object that contains all of the information about the 
machine learning model. Models for each supported machine learning model
are stored in the model directory.

## Examples
The motivating example, a welding model taken from (**paper-ref**), can be found in the
[Weld example Notebook](./PMML_Weld_example.ipynb).

A more complete look at how this network design paradigm might be used with PyMC3 can be found in
[this notebook](./NX_pymc3_BayesNets.ipynb), where you can go through the PyMC3 docs' original
example models in the BayesNet format.

## Future
The next major steps for this project are:
- Add the ability to read in PMML bayes net models to ready-to-sample PyMC3 networks.
- Add more distribution functionality, esp. to the PMML serialization.