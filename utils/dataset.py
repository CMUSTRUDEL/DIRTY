import sys
import ujson as json
import tarfile
from typing import Iterable, List

from tqdm import tqdm
import numpy as np
from utils.ast import AbstractSyntaxTree, SyntaxNode


class Example(object):
    def __init__(self, ast: AbstractSyntaxTree, variable_name_map: dict, binary_file: str = None):
        self.ast = ast
        self.binary_file = binary_file
        self.variable_name_map = variable_name_map

    @classmethod
    def from_json_dict(cls, json_dict, **kwargs):
        tree = AbstractSyntaxTree.from_json_dict(json_dict)
        variable_name_map = dict()
        for variable_node in tree.variable_nodes:
            variable_name_map[variable_node.node_id] = variable_node.new_name

        return cls(tree, variable_name_map, **kwargs)

class Dataset(object):
    def __init__(self, examples):
        self.examples = examples

    @staticmethod
    def iter_from_compressed_file(file_name: str, shuffle=False, progress=False) -> Iterable[Example]:
        with tarfile.open(file_name, 'r') as f:
            files = list(f.getmembers())
            if shuffle:
                np.random.shuffle(files)

            if progress: file_iter = tqdm(files, file=sys.stdout)
            else: file_iter = files

            for filename in filter(lambda x: x.name.endswith('.jsonl'), file_iter):
                jsonl_file = f.extractfile(filename)
                if jsonl_file is not None:
                    for tree_encoding_line in jsonl_file:
                        if tree_encoding_line.decode().startswith('{'):
                            tree_json_dict = json.loads(tree_encoding_line)

                            example = Example.from_json_dict(tree_json_dict, binary_file=filename.name)
                            yield example

    @staticmethod
    def batch_iter_from_compressed_file(file_name: str, batch_size=32, shuffle=True) -> Iterable[List[Example]]:
        example_iter = Dataset.iter_from_compressed_file(file_name, shuffle=shuffle)
        batch = []

        while True:
            try:
                example = next(example_iter)
                batch.append(example)

                if len(batch) == batch_size:
                    yield batch
                    batch = []
            except StopIteration:
                break
        if batch: yield batch


if __name__ == '__main__':
    for example in Dataset.iter_from_compressed_file('data/0-trees.tar.gz'):
        # print(example.root)
        pass
