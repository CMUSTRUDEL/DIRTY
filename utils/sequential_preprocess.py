#!/usr/bin/env python
"""
Usage:
    preprocess.py [options] TAR_FILES OUTPUT_CODE_FILE

Options:
    -h --help                  Show this screen.
"""

from docopt import docopt
from utils.code_processing import VARIABLE_ANNOTATION
from utils.lexer import *
from utils.dataset import Dataset
import ujson as json


def processor(input_queue, output_queue):
    while True:
        payload = input_queue.get()
        if payload is None: break

        examples = []
        for json_str, meta in payload:
            tree_json_dict = json.loads(json_str)
            code_tokens = tokenize_raw_code(tree_json_dict['raw_code'])

            tree_json_dict['code_tokens'] = code_tokens
            json_str = json.dumps(tree_json_dict)

            preprocess_ast(root, code=tree_json_dict['raw_code'])

            # add function name to the name field of the root block
            root.name = tree_json_dict['function']
            root.named_fields.add('name')

            new_json_dict = root.to_json_dict()
            tree_json_dict['ast'] = new_json_dict


            example = Example.from_json_dict(tree_json_dict, binary_file=meta, json_str=json_str)

            if is_valid_example(example):
                canonical_code = canonicalize_code(example.ast.code)
                example.canonical_code = canonical_code
                examples.append(example)

        example_queue.put(examples)

    for i in range(consumer_num):
        example_queue.put(None)

    print('example generator quited!')


def main(args):
    dataset = Dataset(args['TAR_FILES'])
    code_line_file = open(args['OUTPUT_CODE_FILE'], 'w')
    all_preserved_tokens = set()
    for example in dataset.get_iterator(num_workers=5):
        code = example.ast.code
        # code_tokens = tokenize_raw_code(code)
        # preserved_tokens = [token for token in code_tokens if token.startswith('@@') and token.endswith('@@')]
        # all_preserved_tokens.update(preserved_tokens)

        # code_line_file.write(' '.join(code_tokens) + '\n')

    code_line_file.close()

    with open(args['OUTPUT_CODE_FILE'] + '.preserved_tokens.txt', 'w') as f:
        for token in all_preserved_tokens:
            f.write(token + '\n')


if __name__ == '__main__':
    args = docopt(__doc__)
    main(args)
