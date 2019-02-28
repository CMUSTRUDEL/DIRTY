import sys
import ujson as json
import tarfile
from typing import Iterable, List, Dict

from tqdm import tqdm
import numpy as np
from utils.ast import AbstractSyntaxTree, SyntaxNode
from utils.vocab import VocabEntry, SAME_VARIABLE_TOKEN

import torch


class Example(object):
    def __init__(self, ast: AbstractSyntaxTree, variable_name_map: dict, binary_file: str = None):
        self.ast = ast
        self.binary_file = binary_file
        self.variable_name_map = variable_name_map

    @classmethod
    def from_json_dict(cls, json_dict, **kwargs):
        tree = AbstractSyntaxTree.from_json_dict(json_dict)
        variable_name_map = dict()

        for var_name, var_nodes in tree.variables.items():
            variable_name_map[var_name] = var_nodes[0].new_name

        return cls(tree, variable_name_map, **kwargs)


class BatchUtil(object):
    @staticmethod
    def to_batched_prediction_target(source_asts: List['AbstractSyntaxTree'],
                                     variable_name_maps: List[Dict[int, str]],
                                     vocab: VocabEntry):
        batch_size = len(source_asts)
        max_node_size = max(len(tree.variable_nodes) for tree in source_asts)
        tgt_var_node_ids = np.zeros((batch_size, max_node_size), dtype=np.int64)
        tgt_name_ids = np.zeros((batch_size, max_node_size), dtype=np.int64)
        tgt_var_node_mask = torch.zeros(batch_size, max_node_size)

        for e_id, (ast, var_name_map) in enumerate(zip(source_asts, variable_name_maps)):
            _var_node_ids = []
            _tgt_name_ids = []
            for var_name, var_nodes, in ast.variables.items():
                new_var_name = var_name_map[var_name]
                if new_var_name == new_var_name:
                    new_name_token_id = vocab[SAME_VARIABLE_TOKEN]
                else:
                    new_name_token_id = vocab[new_var_name]

                for node in var_nodes:
                    _var_node_ids.append(node.node_id)
                    _tgt_name_ids.append(new_name_token_id)

            tgt_var_node_ids[e_id, :len(_var_node_ids)] = _var_node_ids
            tgt_name_ids[e_id, :len(_var_node_ids)] = _tgt_name_ids
            tgt_var_node_mask[e_id, :len(_var_node_ids)] = 1.

        tgt_var_node_ids = torch.from_numpy(tgt_var_node_ids)
        tgt_name_ids = torch.from_numpy(tgt_name_ids)

        return dict(tgt_variable_node_ids=tgt_var_node_ids,
                    tgt_name_ids=tgt_name_ids,
                    tgt_variable_node_mask=tgt_var_node_mask)


class Dataset(object):
    def __init__(self, dataset_file_path):
        self.file_path = dataset_file_path

        with tarfile.open(dataset_file_path, 'r') as f:
            self.files = [x.name for x in f.getmembers() if x.name.endswith('.jsonl')]

    def __len__(self):
        return len(self.files)

    def iter_from_compressed_file(self, shuffle=False, progress=False) -> Iterable[Example]:
        with tarfile.open(self.file_path, 'r') as f:
            files = list(self.files)
            if shuffle:
                np.random.shuffle(files)

            if progress: file_iter = tqdm(files, file=sys.stdout)
            else: file_iter = files

            for filename in file_iter:
                jsonl_file = f.extractfile(filename)
                if jsonl_file is not None:
                    for tree_encoding_line in jsonl_file:
                        if tree_encoding_line.decode().startswith('{'):
                            tree_json_dict = json.loads(tree_encoding_line)

                            example = Example.from_json_dict(tree_json_dict, binary_file=filename)
                            yield example

    def batch_iter_from_compressed_file(self, batch_size=1024, shuffle=False, progress=True) -> Iterable[List[Example]]:
        example_iter = self.iter_from_compressed_file(shuffle=shuffle, progress=progress)
        batch = []
        batch_node_num = 0

        while True:
            try:
                example = next(example_iter)
                batch.append(example)
                batch_node_num += example.ast.size

                if batch_node_num >= batch_size:
                    yield batch

                    batch = []
                    batch_node_num = 0
            except StopIteration:
                break
        if batch: yield batch


if __name__ == '__main__':
    for _example in Dataset.iter_from_compressed_file('data/0-trees.tar.gz'):
        if _example.ast.size > 200:
            print(_example.binary_file, _example.variable_name_map)
