import sys
import ujson as json
import tarfile
from typing import Iterable, List, Dict
import multiprocessing

from tqdm import tqdm
import numpy as np
from utils.ast import AbstractSyntaxTree, SyntaxNode
from utils.vocab import VocabEntry, SAME_VARIABLE_TOKEN

import torch


class Example(object):
    def __init__(self, ast: AbstractSyntaxTree, variable_name_map: dict, **kwargs):
        self.ast = ast
        self.variable_name_map = variable_name_map

        for key, val in kwargs.items():
            setattr(self, key, val)

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
                                     variable_name_maps: List[Dict],
                                     context_encoding: Dict,
                                     vocab: VocabEntry):
        batch_size = len(source_asts)
        var_node_to_packed_pos_map: dict = context_encoding['var_node_to_packed_pos_map']
        variable_master_node_maps = context_encoding['variable_master_node_maps']

        packed_variable_tgt_name_id = torch.zeros(context_encoding['prediction_node_encoding'].size(0), dtype=torch.long)
        pred_node_ptr = 0
        for e_id, (ast, var_name_map) in enumerate(zip(source_asts, variable_name_maps)):
            variable_master_node_map = variable_master_node_maps[e_id]
            _var_node_ids = []
            _tgt_name_ids = []
            for var_name, var_nodes, in ast.variables.items():
                new_var_name = var_name_map[var_name]
                if var_name == new_var_name:
                    new_name_token_id = vocab[SAME_VARIABLE_TOKEN]
                else:
                    new_name_token_id = vocab[new_var_name]

                # for node in var_nodes:
                #     packed_node_id = var_node_to_packed_pos_map[e_id][node.node_id]
                #     packed_variable_tgt_name_id[packed_node_id] = new_name_token_id

                packed_variable_tgt_name_id[pred_node_ptr] = new_name_token_id
                pred_node_ptr += 1

        return dict(packed_variable_tgt_name_id=packed_variable_tgt_name_id)


def get_json_iterator(file_path, shuffle=False, progress=False) -> Iterable:
    with tarfile.open(file_path, 'r') as f:
        files = [x.name for x in f.getmembers() if x.name.endswith('.jsonl')]
        if shuffle:
            np.random.shuffle(files)

        if progress: file_iter = tqdm(files, file=sys.stdout)
        else: file_iter = files

        for filename in file_iter:
            jsonl_file = f.extractfile(filename)
            if jsonl_file is not None:
                for line_no, tree_encoding_line in enumerate(jsonl_file):
                    if tree_encoding_line.decode().startswith('{'):
                        # tree_json_dict = json.loads(tree_encoding_line)
                        yield tree_encoding_line, dict(file_name=filename, line_num=line_no)


def json_line_reader(file_path, queue, worker_num, shuffle, progress):
    for json_str in get_json_iterator(file_path, shuffle, progress):
        queue.put(json_str)

    for i in range(worker_num):
        queue.put(None)


def example_generator(json_queue, example_queue):
    while True:
        payload = json_queue.get()
        if payload is None: break
        json_str, meta = payload

        tree_json_dict = json.loads(json_str)
        example = Example.from_json_dict(tree_json_dict, binary_file=meta)

        if example.ast.size != max(node.node_id for node in example.ast) + 1:
            continue

        example_queue.put(example)

    example_queue.put(None)


class Dataset(object):
    def __init__(self, dataset_file_path):
        self.file_path = dataset_file_path

        print(f'reading dataset {dataset_file_path}', file=sys.stderr)
        example_num = 0
        for _ in get_json_iterator(dataset_file_path):
            example_num += 1
        self.size = example_num

    def __len__(self):
        return self.size

    def __iter__(self):
        return self.get_iterator(progress=True)

    def get_single_process_iterator(self, shuffle=False, progress=False) -> Iterable[Example]:
        json_str_iter = get_json_iterator(self.file_path, shuffle, progress)
        for json_str in json_str_iter:
            tree_json_dict = json.loads(json_str)
            example = Example.from_json_dict(tree_json_dict, binary_file=tree_json_dict['file_name'])

            if example.ast.size != max(node.node_id for node in example.ast) + 1:
                continue

            yield example

    def _get_iterator(self, shuffle=False, num_workers=1):
        json_enc_queue = multiprocessing.Queue()
        example_queue = multiprocessing.Queue()

        json_loader = multiprocessing.Process(target=json_line_reader, args=(self.file_path, json_enc_queue, num_workers,
                                                                             shuffle, False))
        json_loader.daemon = True
        example_generators = []
        for i in range(num_workers):
            p = multiprocessing.Process(target=example_generator, args=(json_enc_queue, example_queue))
            p.daemon = True
            example_generators.append(p)

        json_loader.start()
        for p in example_generators: p.start()

        num_finished_workers = 0
        while True:
            example = example_queue.get()
            if example is not None:
                yield example
            else:
                num_finished_workers += 1
                if num_finished_workers == num_workers: break

        json_loader.join()
        for p in example_generators: p.join()

    def get_iterator(self, shuffle=False, progress=True, num_workers=1):
        if progress:
            it_func = lambda x: tqdm(x, total=len(self), file=sys.stdout)
        else:
            it_func = lambda x: x

        return it_func(self._get_iterator(shuffle, num_workers))

    def batch_iterator(self, batch_size=1024, num_workers=1, shuffle=False, progress=True) -> Iterable[List[Example]]:
        example_iter = self.get_iterator(shuffle=shuffle, progress=progress, num_workers=num_workers)
        batch = []
        batch_node_num = 0

        for example in example_iter:
            if example.ast.size < 300 and len(example.variable_name_map) > 0:
                batch.append(example)
                batch_node_num += example.ast.size

                if batch_node_num >= batch_size:
                    yield batch

                    batch = []
                    batch_node_num = 0

        if batch: yield batch


if __name__ == '__main__':
    for _example in Dataset('data/0-trees.tar.gz'):
        if _example.ast.size > 200:
            print(_example.binary_file, _example.variable_name_map)
