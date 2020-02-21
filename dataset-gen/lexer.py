# Lexes (some of) the output of the Hex-Rays decompiler and outputs in a format
# appropriate for SMT-based experiments. There are two problems with using the
# base C Lexer from Pygments:
# - Pygments does not lex two-character operators (e.g., >>, ->) as single
#   tokens. This is a problem when tokens are treated as "words".
# - Hex-Rays augments C syntax with the '::' operator borrowed from C++ to
#   reference shadowed global variables. We throw away this operator.

# Additionally, adds variable placeholders of the form '@@VAR_[id], where [id]
# is an integer.

import csv
from enum import Enum, auto
from hashlib import sha256
from pygments import lex
from pygments.token import Token
from pygments.token import is_token_subtype
from pygments.lexers.c_cpp import CLexer, inherit
from typing import Any, List, Tuple

Token.Placeholder = Token.Token.Placeholder


def hash_line(line):
    '''Hashes placeholders in a line passed as a list of (token_type, token_name)
    pairs.  A hash combines the hash of the tokens to the left of a placeholder
    (excluding other placeholders) with the hash of the tokens to the right of a
    placeholder (including itself. This encodes position in addition to the
    contents of the rest of the line.
    '''
    names = [name if not is_token_subtype(token_type, Token.Placeholder)
             else '@@PLACEHOLDER' for (token_type, name) in line]

    hashed_line = []
    for index, (token_type, name) in enumerate(line):
        if is_token_subtype(token_type, Token.Placeholder):
            fst = names[:index]
            snd = names[index:]
            hashed_name = sha256()
            hashed_name.update(str(fst).encode('utf-8'))
            hashed_name.update(hashed_name.digest())
            hashed_name.update(str(snd).encode('utf-8'))
            hashed_line.append((token_type, hashed_name.hexdigest()))
        else:
            hashed_line.append((token_type, name))
    return hashed_line


class VarNaming(Enum):
    NONE = auto()
    HASH = auto()
    TABLE = auto()


class Lexer:
    def __init__(self, file_path, var_table=None):
        self.program_text = open(file_path, 'r').read()
        self.tokens = list(lex(self.program_text, HexRaysCLexer()))
        # Maps a placeholder id to a dict of variable names
        self.var_table = dict()
        if var_table:
            with open(var_table, newline='') as tablefile:
                reader = csv.DictReader(tablefile, delimiter=',',
                                        quotechar='|')
                for row in reader:
                    self.var_table[row.pop('var_id')] = row

    # Generator that returns the next line with placeholder tokens replaced.
    # Use a generator here so that the list of tokens only has to live in memory
    # once.
    def get_lines(self, var_naming=VarNaming.NONE, var_table=None):
        line = []
        for (token_type, token) in self.tokens:
            if is_token_subtype(token_type, Token.Comment) and len(line) > 0:
                if var_naming == VarNaming.HASH:
                    line = hash_line(line)
                yield line
                line = []
            elif is_token_subtype(token_type, Token.String):
                # Pygments breaks up strings into individual tokens representing
                # things like opening quotes and escaped characters. We want to
                # collapse all of these into a single string literal token.
                if line != [] and line[-1] == (Token.String, '<LITERAL_STRING>'):
                    continue
                line.append((Token.String, '<LITERAL_STRING>'))
            elif is_token_subtype(token_type, Token.Number):
                line.append((Token.String, '<LITERAL_NUMBER>'))
            # Skip the :: token
            elif is_token_subtype(token_type, Token.Operator) and token == '::':
                continue
            # Replace placeholders if using table renaming
            elif var_naming == VarNaming.TABLE \
                 and is_token_subtype(token_type, Token.Placeholder.Var):
                if not var_table:
                    raise KeyError('var_table must be set with table renaming')
                # Remove the '@@VAR_' from the beginning of the placeholder
                var_id = token[6:]
                line.append((Token.Placeholder.Var,
                             self.var_table[var_id][var_table]))
            elif not is_token_subtype(token_type, Token.Text):
                line.append((token_type, token.strip()))
            elif '\n' in token and len(line) > 0:
                if var_naming == VarNaming.HASH:
                    line = hash_line(line)
                yield line
                line = []

    def write_lines(self, out_file, var_names=None):
        lines = '\n'.join([' '.join([tok for (_, tok) in line])
                           for line in self.get_lines(var_names)]) + '\n'
        out_file.write(lines.encode('utf8'))


class HexRaysCLexer(CLexer):
    tokens = {
        'statements' : [
            (r'->', Token.Operator),
            (r'\+\+', Token.Operator),
            (r'--', Token.Operator),
            (r'==', Token.Operator),
            (r'!=', Token.Operator),
            (r'>=', Token.Operator),
            (r'<=', Token.Operator),
            (r'&&', Token.Operator),
            (r'\|\|', Token.Operator),
            (r'\+=', Token.Operator),
            (r'-=', Token.Operator),
            (r'\*=', Token.Operator),
            (r'/=', Token.Operator),
            (r'%=', Token.Operator),
            (r'&=', Token.Operator),
            (r'\^=', Token.Operator),
            (r'\|=', Token.Operator),
            (r'<<=', Token.Operator),
            (r'>>=', Token.Operator),
            (r'<<', Token.Operator),
            (r'>>', Token.Operator),
            (r'\.\.\.', Token.Operator),
            (r'##', Token.Operator),
            (r'::', Token.Operator),
            (r'@@VAR_[0-9]+', Token.Placeholder.Var),
            inherit
        ]
    }
