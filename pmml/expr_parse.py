from __future__ import division
import lxml.etree as et
try:
    from sympy.parsing.sympy_parser import parse_expr
except ImportError:
    'Sympy is needed to parse expression as xml for PMML'


def pmml_to_srepr(xml_expr, nsp=''):
    def sympy_pmml_map(pmml_func):
        dic = {
            '*': 'Mul',
            '+': 'Add',
            'pow': 'Pow',
            'ln': 'log'
        }
        return dic[pmml_func]

    def tree(n):
        r = ''

        def print_node(node, last=False):

            """
            Returns the "node" string representation.
            """
            if node.getchildren() or (last == True):
                delim = ''
            else:
                delim = ', '

            if node.tag in [nsp+'Apply']:
                if node.getchildren() and (last == False):
                    delim = ', '
                s = "{}(".format(sympy_pmml_map(node.attrib['function'])) + \
                    tree(node) + '){}'.format(delim)
            elif node.tag in [nsp+'FieldRef']:
                s = 'Symbol(\'{}\'){}'.format(node.attrib['field'], delim)
            elif "{:g}".format(float(node.text)) == '-1':

                s = 'Integer(-1){}'.format(delim)
            else:
                s = "Float({:g}){}".format(float(node.text), delim)
            return s

        children = list(n.iterchildren())

        for child in children[:-1]:
            r += print_node(child)
        r += print_node(children[-1], last=True)
        return r

    res = tree(xml_expr)
    return res


def parsed_math(xml_expr, nsp='', func=None):
    if func is None:
        return parse_expr(pmml_to_srepr(xml_expr, nsp=nsp))
    else:
        return parse_expr(func(pmml_to_srepr(xml_expr, nsp=nsp)))

if __name__=='__main__':
    import sympy as sy
    import lxml.etree as et

    expr = sy.sympify('rho*V*(H + C_p*(T_f-T_i))')
    # expr = sy.sympify('L*((0.75)*mu_l*mu_h + g*t + (mu_l-g)*(t-mu_e)*0.5)')
    print 'The result should look like this: \n\n\t', sy.srepr(expr)
    print '\nwhich parses to this: \n\t', parse_expr(sy.srepr(expr)), '\n----'
    # sy.parsing.sympy_parser.parse_expr(sy.srepr(expr))

    xml_expr = """
    <StaticValue>
                  <Apply function="*">
                    <FieldRef field="V"/>
                    <FieldRef field="rho"/>
                    <Apply function="+">
                      <FieldRef field="H"/>
                      <Apply function="*">
                        <FieldRef field="C_p"/>
                        <Apply function="+">
                          <FieldRef field="T_f"/>
                          <Apply function="*">
                            <Constant dataType="double">-1</Constant>
                            <FieldRef field="T_i"/>
                          </Apply>
                        </Apply>
                      </Apply>
                    </Apply>
                  </Apply>
                </StaticValue>
    """
    xml_expr = et.fromstring(xml_expr)

    res = pmml_to_srepr(xml_expr)
    # print res
    print "instead we get this: \n\n\t", res
    print '\nwhich parses to this:\n\t ', parse_expr(res)