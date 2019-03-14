import pickle
import sys
import time
import ujson as json
import tarfile
from typing import Iterable, List, Dict, Union
import multiprocessing

from tqdm import tqdm
import numpy as np
from utils.ast import AbstractSyntaxTree, SyntaxNode
from utils.graph import PackedGraph
from utils.vocab import VocabEntry, SAME_VARIABLE_TOKEN, Vocab
from model.encoder import GraphASTEncoder
import sentencepiece as spm
import random

import torch
import torch.multiprocessing as torch_mp


batcher_sync_msg = None


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


class Batch(object):
    __slots__ = ('examples', 'tensor_dict')

    def __init__(self, examples, tensor_dict):
        self.examples = examples
        self.tensor_dict = tensor_dict

    @property
    def size(self):
        return self.tensor_dict['batch_size']


class Batcher(object):
    def __init__(self, config, return_examples=True):
        self.config = config
        self.vocab = Vocab.load(config['data']['vocab_file'])
        self.grammar = self.vocab.grammar

        self.return_examples = return_examples

    def to_tensor_dict(self, examples: List[Example] = None, source_asts: List[AbstractSyntaxTree] = None) -> Dict[str, torch.Tensor]:
        if examples:
            source_asts = [e.ast for e in examples]

        packed_graph = GraphASTEncoder.to_packed_graph(source_asts,
                                                       connections=self.config['encoder']['connections'])

        tensor_dict = {'adj_lists': packed_graph.adj_lists,
                       'variable_master_node_restoration_indices': packed_graph.variable_master_node_restoration_indices,
                       'variable_master_node_restoration_indices_mask': packed_graph.variable_master_node_restoration_indices_mask,
                       'variable_master_node_ids': [_id for node, _id in
                                                    packed_graph.get_nodes_by_group('variable_master_nodes')]}

        packed_graph.adj_lists = None
        packed_graph.variable_master_node_restoration_indices = None
        packed_graph.variable_master_node_restoration_indices_mask = None

        _tensors = GraphASTEncoder.to_tensor_dict(packed_graph,
                                                  self.grammar, self.vocab)
        tensor_dict.update(_tensors)

        if examples:
            prediction_target = self.to_batched_prediction_target(source_asts, [e.variable_name_map for e in examples],
                                                                  packed_graph)
            tensor_dict['prediction_target'] = prediction_target

        tensor_dict['batch_size'] = len(source_asts)
        tensor_dict['packed_graph_size'] = packed_graph.size

        return tensor_dict

    def to_batch(self, examples: List[Example]) -> Batch:
        tensor_dict = self.to_tensor_dict(examples)
        if not self.return_examples:
            examples = None

        batch = Batch(examples, tensor_dict)

        return batch

    def to_batched_prediction_target(self,
                                     source_asts: List['AbstractSyntaxTree'],
                                     variable_name_maps: List[Dict],
                                     packed_graph: PackedGraph):
        batch_size = len(source_asts)
        unchanged_var_weight = self.config['train']['unchanged_variable_weight']

        use_bpe_for_var_name = self.config['decoder']['type'] == 'RecurrentSubtokenDecoder'

        variable_name_subtoken_maps = []
        if use_bpe_for_var_name:
            eov_id = self.vocab.target.subtoken_model.eos_id()
            for var_name_map in variable_name_maps:
                var_name_subtoken_map = dict()
                for old_name, new_name in var_name_map.items():
                    if old_name == new_name:
                        subtoken_ids = [self.vocab.target[SAME_VARIABLE_TOKEN], eov_id]
                    else:
                        subtoken_ids = self.vocab.target.subtoken_model.encode_as_ids(new_name) + [eov_id]
                    var_name_subtoken_map[old_name] = subtoken_ids
                variable_name_subtoken_maps.append(var_name_subtoken_map)
        else:
            for var_name_map in variable_name_maps:
                var_name_subtoken_map = dict()
                for old_name, new_name in var_name_map.items():
                    if old_name == new_name:
                        subtoken_ids = [self.vocab.target[SAME_VARIABLE_TOKEN]]
                    else:
                        subtoken_ids = [self.vocab.target[new_name]]
                    var_name_subtoken_map[old_name] = subtoken_ids
                variable_name_subtoken_maps.append(var_name_subtoken_map)

        max_pred_timestep = max(sum(len(val) for val in x.values()) for x in variable_name_subtoken_maps)

        var_encoding_restoration_indices = torch.zeros(batch_size, max_pred_timestep, dtype=torch.long)
        var_encoding_restoration_indices_mask = torch.zeros(batch_size, max_pred_timestep)

        variable_tgt_name_id = torch.zeros(batch_size, max_pred_timestep, dtype=torch.long)
        variable_tgt_name_weight = torch.zeros(batch_size, max_pred_timestep)
        var_with_new_name_mask = torch.zeros(batch_size, max_pred_timestep)
        auxiliary_var_mask = torch.zeros(batch_size, max_pred_timestep)

        variable_master_node_ptr = 0
        for e_id, (ast, var_name_map) in enumerate(zip(source_asts, variable_name_maps)):
            _var_node_ids = []
            _tgt_name_ids = []
            variable_ptr = 0
            for var_name in ast.variables:
                new_var_name_subtoken_ids = variable_name_subtoken_maps[e_id][var_name]
                variable_end_ptr = variable_ptr + len(new_var_name_subtoken_ids)

                variable_tgt_name_id[e_id, variable_ptr: variable_end_ptr] = torch.tensor(new_var_name_subtoken_ids, dtype=torch.long)

                if var_name == var_name_map[var_name]:
                    auxiliary_var_mask[e_id, variable_ptr: variable_end_ptr] = 1.
                    variable_tgt_name_weight[e_id, variable_ptr: variable_end_ptr] = unchanged_var_weight
                else:
                    var_with_new_name_mask[e_id, variable_ptr: variable_end_ptr] = 1.
                    variable_tgt_name_weight[e_id, variable_ptr: variable_end_ptr] = 1.

                var_encoding_restoration_indices[e_id, variable_ptr: variable_end_ptr] = variable_master_node_ptr

                variable_master_node_ptr += 1
                variable_ptr = variable_end_ptr

            var_encoding_restoration_indices_mask[e_id, :variable_ptr] = 1.

        return dict(variable_tgt_name_id=variable_tgt_name_id,
                    variable_tgt_name_weight=variable_tgt_name_weight,
                    var_with_new_name_mask=var_with_new_name_mask,
                    auxiliary_var_mask=auxiliary_var_mask,
                    variable_encoding_restoration_indices=var_encoding_restoration_indices,
                    variable_encoding_restoration_indices_mask=var_encoding_restoration_indices_mask)


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
    # if bpe_model_path:
    #     sp = spm.SentencePieceProcessor()
    #     sp.Load(bpe_model_path)
    while True:
        payload = json_queue.get()
        if payload is None: break
        json_str, meta = payload

        tree_json_dict = json.loads(json_str)
        example = Example.from_json_dict(tree_json_dict, binary_file=meta)

        if example.ast.size != max(node.node_id for node in example.ast) + 1:
            continue

        # if bpe_model_path:
        #     for node in example.ast:
        #         if node.node_type == 'obj':
        #             setattr(node, 'sub_tokens', sp.EncodeAsPieces(node.name))

        example_queue.put(example)
        # print('Push one example', file=sys.stderr)

    example_queue.put(None)


def is_valid_example(example):
    return example.ast.size < 300 and \
           len(example.variable_name_map) > 0 and \
           any(k != v for k, v in example.variable_name_map.items())


def examples_to_batch(example_queue, batch_queue, batch_size, batcher):
    batch_examples = []
    batch_node_num = 0

    while True:
        # print('Example queue size {}'.format(example_queue.qsize()), file=sys.stderr)
        example = example_queue.get()
        if example is None:
            break

        if is_valid_example(example):
            batch_examples.append(example)
            batch_node_num += example.ast.size

        if batch_node_num >= batch_size:
            # t1 = time.time()
            batch = batcher.to_batch(batch_examples)
            # print(f'[Batcher] {time.time() - t1}s took for tensorization', file=sys.stderr)

            # t1 = time.time()
            batch_queue.put(batch)
            # print(f'[Batcher] {time.time() - t1}s took to push one batch', file=sys.stderr)

            batch_examples = []
            batch_node_num = 0

    if batch_examples:
        batch = batcher.to_batch(batch_examples)
        batch_queue.put(batch)

    batch_queue.put(None)
    while batcher_sync_msg.value == 0:
        time.sleep(1)

    # print('[Batcher] Quit current batcher', file=sys.stderr)
    sys.stderr.flush()


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
            p = multiprocessing.Process(target=example_generator,
                                        args=(json_enc_queue, example_queue))
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

    def batch_iterator(self, batch_size: int, config: Dict,
                       return_examples=True,
                       num_readers=2,
                       progress=True, shuffle=False, single_batcher=False) -> Iterable[Union[Batch, Dict[str, torch.Tensor]]]:
        if progress:
            it_func = lambda x: tqdm(x, file=sys.stdout)
        else:
            it_func = lambda x: x

        if single_batcher:
            return it_func(self._single_process_batch_iter(batch_size, config, num_readers, shuffle))
        else:
            return it_func(self._batch_iterator(batch_size, config, num_readers, shuffle, return_examples))

    def _batch_iterator(self, batch_size: int, config: Dict, num_readers=2, shuffle=False, return_examples=True) -> Iterable[Batch]:
        global batcher_sync_msg
        batcher_sync_msg = multiprocessing.Value('i', 0)
        json_enc_queue = multiprocessing.Queue(maxsize=30000)
        example_queue = multiprocessing.Queue(maxsize=30000)

        json_loader = multiprocessing.Process(target=json_line_reader,
                                              args=(self.file_path, json_enc_queue, num_readers,
                                                    shuffle, False))
        json_loader.daemon = True
        example_generators = []
        for i in range(num_readers):
            p = multiprocessing.Process(target=example_generator,
                                        args=(json_enc_queue, example_queue))
            p.daemon = True
            example_generators.append(p)

        json_loader.start()
        for p in example_generators: p.start()

        batch_queue = torch_mp.Queue()
        batcher = Batcher(config, return_examples=return_examples)
        batchers = []
        num_batchers = num_readers
        for i in range(num_batchers):
            p = torch_mp.Process(target=examples_to_batch,
                                 args=(example_queue, batch_queue, batch_size, batcher))
            p.daemon = True
            batchers.append(p)

        for p in batchers: p.start()

        num_finished_batchers = 0
        while True:
            # t1 = time.time()
            batch = batch_queue.get()
            # print(f'{time.time() - t1} took to load a batch', file=sys.stderr)
            if batch is not None:
                yield batch
            else:
                # print('One batcher finished!', file=sys.stderr)
                num_finished_batchers += 1
                if num_finished_batchers == num_batchers: break

        batcher_sync_msg.value = 1
        json_loader.join()
        for p in example_generators: p.join()
        for p in batchers: p.join()

    def _single_process_batch_iter(self, batch_size: int, config: Dict, num_readers=2, shuffle=False) -> Iterable[Batch]:
        batcher = Batcher(config)
        example_iter = self.get_iterator(shuffle=shuffle, progress=False, num_workers=num_readers)
        # t1 = time.time()
        batch_examples = []
        batch_node_num = 0

        # if example.ast.size < 300 and len(example.variable_name_map) > 0:
        for example in filter(is_valid_example, example_iter):
            batch_examples.append(example)
            batch_node_num += example.ast.size

            if batch_node_num >= batch_size:
                batch = batcher.to_batch(batch_examples)
                # print(f'[Dataset] {time.time() - t1} took to load a batch', file=sys.stderr)
                yield batch

                batch_examples = []
                batch_node_num = 0
                # t1 = time.time()

        if batch_examples:
            batch = batcher.to_batch(batch_examples)
            yield batch


if __name__ == '__main__':
    for _example in Dataset('data/0-trees.tar.gz'):
        if _example.ast.size > 200:
            print(_example.binary_file, _example.variable_name_map)
