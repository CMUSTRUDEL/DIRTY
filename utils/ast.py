class AbstractSyntaxNode(object):
    """represent a node on an AST"""
    pass


class TerminalNode(AbstractSyntaxNode):
    """a terminal AST node representing variables or other terminal syntax tokens"""
    pass


class AbstractSyntaxTree(object):
    pass