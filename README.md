# BayesNetMC

## Why?

This is a python package that extends PyMC3 to enable the definition of Bayesian Network models
without immediate compilation in Theano, using a NetworkX  directed graph to store model information.

Additionally, this package allows the serialization of a pymc3-style bayes-net to the PMML format, at
least for supported node types. Currently, only continuous nodes are supported, though in theory any
distribution allowed by PyMC3 could be added in the future. This parsing is done with the help of SymPy
to allow complex node function definitions as combinations of other node values (i.e. inheritance).


## Usage
To get a feel for the way the package works, see the Examples linked below. 

### Limitations
Please be aware, the current implementation has limitations on compatibility with PMML, like 
- Only supports ContinuousNode instances to/from PMML
- Limited PyMC3 RV distributions (see [source code](/pymcnet/net.py) for now)
- Use of `eval()` (!) to compile theano graph containing RV references, via un-linked sympy expression (see [#1](/../../issues/1))

### Requirements
pymcBN currently runs on the following: 
- PyMC3 (for model instantiation and sampling)
- NetworkX (for model creation and transferrability)
- SymPy (for parsing mathematical expressions to/from PMML)

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
