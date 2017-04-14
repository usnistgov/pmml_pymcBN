# Parsers

Parsers understand different formats for predictive models.
Most predictive models will be in the PMML format, but there may be new formats for 
predictive models developed in the future. All parsers should support the Parser.parse method

## Parser.parse(filename)
Return a Model object that contains all of the information about the 
machine learning model. Models for each supported machine learning model
are stored in the model directory.