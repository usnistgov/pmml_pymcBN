# MODIFY THIS TO MAKE TREE STUFF
from __future__ import division

try:
    import sympy as sy
except ImportError:
    'Sympy is needed to parse expression as xml for PMML'


def xml_tree(node, pretty=False):
    if pretty:
        delim = '\n'
        sep = '\t'
    else:
        delim = ''
        sep = ''

    def pprint_nodes(subtrees):
        """
        Prettyprints systems of nodes.
        """

        def indent(s):
            x = s.split("\n")
            r = "{}{}".format(x[0], delim)
            for a in x[1:]:
                if a == "":
                    continue
                else:
                    r += "{}{}{}".format(sep, a, delim)
            return r

        if len(subtrees) == 0:
            return ""
        f = ""
        for a in subtrees[:-1]:
            f += indent(a)
        f += indent(subtrees[-1])
        f += '</Apply>'
        return f

    def sympy_pmml_map(sy_func):
        dic = {
            'Mul': '*',
            'Add': '+',
            'Pow': 'pow',
            'log': 'ln'
        }
        return dic[sy_func]

    def print_node(node):
        """
        Returns the "node" class name string representation.
        """
        if node.__class__.__name__ in ['Float', 'NegativeOne']:
            s = '<Constant dataType=\"double\">'
            s += "{}".format(str(node))
            s += '</Constant>'
        elif node.__class__.__name__ == 'Symbol':
            s = '<FieldRef field=\"{}\"/>'.format(str(node))
        else:
            s = "<Apply function=\"{}\">{}".format(sympy_pmml_map(node.__class__.__name__), delim)
        return s

    def tree(node):
        """
        Returns a tree representation of "node" as a string.
        It uses print_node() together with pprint_nodes() on node.args recursively.
        """
        subtrees = []
        for arg in node.args:
            subtrees.append(tree(arg))
        s = print_node(node) + pprint_nodes(subtrees)
        return s

    return tree(node)


def get_xml_expr(str_expr):
    """
    for parsing a string a BN node value function (required for PMML serialization of non-root BN nodes)
    :param str_expr: a string containing a parsable mathematical expression,
    :return: string containing PMML-style XML expression for node value
    """


    expr = sy.sympify(str_expr)
    return xml_tree(expr)


