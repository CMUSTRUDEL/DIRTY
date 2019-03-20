import re

from utils.ast import SyntaxNode
from utils.lexer import Lexer


def canonicalize_code(code):
    code = re.sub('//.*?\\n|/\\*.*?\\*/', '\\n', code, flags=re.S)
    lines = [l.rstrip() for l in code.split('\\n')]
    code = '\\n'.join(lines)
    code = re.sub('@@\\w+@@(\\w+)@@\\w+', '\\g<1>', code)

    return code


def canonicalize_constants(root: SyntaxNode) -> None:
    def _visit(node):
        if node.node_type == 'obj' and node.type == 'char *':
            node.name = 'STRING'
        elif node.node_type == 'num':
            node.name = 'NUMBER'
        elif node.node_type == 'fnum':
            node.name = 'FLOAT'

        for child in node.member_nodes:
            _visit(child)

    _visit(root)


def annotate_type(root: SyntaxNode) -> None:
    def _visit(node):
        if hasattr(node, 'type'):
            type_tokens = [t[1].lstrip('_') for t in Lexer(node.type).get_tokens()]
            type_tokens = [t for t in type_tokens if t not in ('(', ')')]
            setattr(node, 'type_tokens', type_tokens)
            node.named_fields['type_tokens'] = type_tokens

        for child in node.member_nodes:
            _visit(child)

    _visit(root)