import re

from dirty.utils.lexer import Lexer, Token

VARIABLE_ANNOTATION = re.compile(r"@@\w+@@(\w+)@@\w+")


def canonicalize_code(code):
    code = re.sub("//.*?\\n|/\\*.*?\\*/", "\\n", code, flags=re.S)
    lines = [line.rstrip() for line in code.split("\\n")]
    code = "\\n".join(lines)
    code = re.sub("@@\\w+@@(\\w+)@@\\w+", "\\g<1>", code)

    return code


def tokenize_raw_code(raw_code):
    lexer = Lexer(raw_code)
    tokens = []
    for token_type, token in lexer.get_tokens():
        if token_type in Token.Literal:
            token = str(token_type).split(".")[2]

        tokens.append(token)

    return tokens
