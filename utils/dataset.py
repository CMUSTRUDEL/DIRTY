import glob
import pickle
import sys
import os
import time
import ujson as json
import tarfile
from typing import Iterable, List, Dict, Union, Tuple
import multiprocessing
import threading
import queue

from tqdm import tqdm
import numpy as np

from utils import nn_util
from utils.ast import AbstractSyntaxTree, SyntaxNode
from utils.code_processing import annotate_type
from utils.graph import PackedGraph
from utils.vocab import VocabEntry, SAME_VARIABLE_TOKEN, Vocab
from model.encoder import GraphASTEncoder
import sentencepiece as spm
import random

import torch
import torch.multiprocessing as torch_mp


batcher_sync_msg = None

torch.multiprocessing.set_sharing_strategy('file_system')

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


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

        if 'test_meta' in json_dict:
            kwargs['test_meta'] = json_dict['test_meta']

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

        packed_graph, graph_tensor_dict = GraphASTEncoder.to_packed_graph(source_asts,
                                                                          connections=self.config['encoder']['connections'])

        tensor_dict = graph_tensor_dict

        _tensors = GraphASTEncoder.to_tensor_dict(packed_graph,
                                                  self.grammar, self.vocab)
        tensor_dict.update(_tensors)

        if examples:
            prediction_target = self.to_batched_prediction_target(examples, packed_graph)
            tensor_dict['prediction_target'] = prediction_target

        tensor_dict['batch_size'] = len(source_asts)
        tensor_dict['packed_graph_size'] = packed_graph.size

        num_elements = nn_util.get_tensor_dict_size(tensor_dict)
        tensor_dict['num_elements'] = num_elements

        return tensor_dict

    def to_batch(self, examples: List[Example]) -> Batch:
        with torch.no_grad():
            tensor_dict = self.to_tensor_dict(examples)

        if not self.return_examples:
            examples = None

        batch = Batch(examples, tensor_dict)

        return batch

    def to_batched_prediction_target(self,
                                     examples: List[Example],
                                     packed_graph: PackedGraph):
        batch_size = len(examples)
        unchanged_var_weight = self.config['train']['unchanged_variable_weight']

        use_bpe_for_var_name = self.config['decoder']['type'] == 'RecurrentSubtokenDecoder'

        variable_name_subtoken_maps = []
        if use_bpe_for_var_name:
            # eov_id = self.vocab.target.subtoken_model.eos_id()
            # for var_name_map in variable_name_maps:
            #     var_name_subtoken_map = dict()
            #     for old_name, new_name in var_name_map.items():
            #         if old_name == new_name:
            #             subtoken_ids = [self.vocab.target[SAME_VARIABLE_TOKEN], eov_id]
            #         else:
            #             subtoken_ids = self.vocab.target.subtoken_model.encode_as_ids(new_name) + [eov_id]
            #         var_name_subtoken_map[old_name] = subtoken_ids
            variable_name_subtoken_maps = [e.variable_name_subtoken_map for e in examples]
        else:
            for example in examples:
                var_name_map = example.variable_name_map
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
        for e_id, example in enumerate(examples):
            ast = example.ast
            var_name_map = example.variable_name_map
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


def get_json_iterator_from_tar_file(file_paths, shuffle=False, progress=False, group_by=None) -> Iterable:
    assert group_by in (None, 'binary_file')

    if isinstance(file_paths, str):
        file_paths = [file_paths]

    if shuffle:
        np.random.shuffle(file_paths)

    for file_path in file_paths:
        payloads = []
        t1 = time.time()
        with tarfile.open(file_path, 'r') as f:
            files = [x.name for x in f.getmembers() if x.name.endswith('.jsonl')]
            # if shuffle:
            #     np.random.shuffle(files)

            if progress: file_iter = tqdm(files, file=sys.stdout)
            else: file_iter = files

            for filename in file_iter:
                jsonl_file = f.extractfile(filename)
                if jsonl_file is not None:
                    if group_by is None:
                        for line_no, tree_encoding_line in enumerate(jsonl_file):
                            # if tree_encoding_line.decode().startswith('{'):
                            # tree_json_dict = json.loads(tree_encoding_line)
                            payload = tree_encoding_line, dict(file_name=filename, line_num=line_no)
                            # yield payload
                            payloads.append(payload)
                    elif group_by == 'binary_file':
                        lines = [(l.decode().strip(), dict(file_name=filename, line_num=line_no))
                                 for line_no, l in enumerate(jsonl_file)]
                        yield lines

        if shuffle:
            np.random.shuffle(payloads)

        print(f'load shard {file_path} took {time.time() - t1:.4f}s', file=sys.stderr)

        for payload in payloads:
            yield payload


def json_line_reader(file_path, queue, worker_num, shuffle, progress, group_by=None):
    for json_str in get_json_iterator_from_tar_file(file_path, shuffle, progress, group_by=group_by):
        queue.put(json_str)

    for i in range(worker_num):
        queue.put(None)


def is_valid_training_example(example):
    if hasattr(example, 'target_prediction_seq_length'):
        if example.target_prediction_seq_length >= 200: return False

    return example.ast.size < 300 and \
           len(example.variable_name_map) > 0 and \
           any(k != v for k, v in example.variable_name_map.items())


def example_generator(json_queue, example_queue, consumer_num=1, train=False, config=None):
    if config:
        vocab = Vocab.load(config['data']['vocab_file'])
        tgt_bpe_model = vocab.target.subtoken_model
    else:
        tgt_bpe_model = None

    while True:
        payload = json_queue.get()
        if payload is None: break
        json_str, meta = payload

        tree_json_dict = json.loads(json_str)
        example = Example.from_json_dict(tree_json_dict, binary_file=meta)

        # if example.ast.size != max(node.node_id for node in example.ast) + 1:
        #     continue

        if tgt_bpe_model:
            eov_id = tgt_bpe_model.eos_id()
            variable_name_subtoken_map = dict()
            tgt_pred_seq_len = 0
            for old_name, new_name in example.variable_name_map.items():
                if old_name == new_name:
                    subtoken_ids = [vocab.target[SAME_VARIABLE_TOKEN], eov_id]
                else:
                    subtoken_ids = tgt_bpe_model.encode_as_ids(new_name) + [eov_id]
                variable_name_subtoken_map[old_name] = subtoken_ids
                tgt_pred_seq_len += len(subtoken_ids)

            setattr(example, 'variable_name_subtoken_map', variable_name_subtoken_map)
            setattr(example, 'target_prediction_seq_length', tgt_pred_seq_len)

        if train:
            if is_valid_example(example):
                example_queue.put(example)
        else:
            example_queue.put(example)

        # print('Push one example', file=sys.stderr)

    # if buffer:
    #     _sort_and_push(buffer)

    for i in range(consumer_num):
        example_queue.put(None)

    # print('[Example Generator] example generator process quit!')


def get_batch_size(batch_examples):
    return len(batch_examples) * max(e.target_prediction_seq_length for e in batch_examples)


def train_example_sort_key(example):
    return example.target_prediction_seq_length


def example_to_batch(json_queue, batched_examples_queue, batch_size, train, config, worker_manager_lock):
    batcher = Batcher(config, return_examples=not train)
    tgt_bpe_model = batcher.vocab.target.subtoken_model
    vocab = batcher.vocab

    buffer_size = config['train']['buffer_size']
    buffer = []
    print(f'[ExampleToBatch] pid={os.getpid()}', file=sys.stderr)

    def _generate_batches():
        buffer.sort(key=train_example_sort_key)

        batches = []
        batch_examples = []

        for example in buffer:
            batch_size_with_example = get_batch_size(batch_examples + [example])
            if batch_examples and batch_size_with_example > batch_size:
                batches.append(batch_examples)
                batch_examples = []

            batch_examples.append(example)

        if batch_examples:
            batches.append(batch_examples)

        if train:
            random.shuffle(batches)

        for batch_examples in batches:
            batch = batcher.to_batch(batch_examples)
            # while batched_examples_queue.qsize() > 100:
            #     time.sleep(10)
            # print(batch.tensor_dict['num_elements'])
            while worker_manager_lock.value == 1:
                time.sleep(0.2)
            batched_examples_queue.put(batch)
            # print(f'[ExampleToBatch] batched examples queue size {batched_examples_queue.qsize()}', file=sys.stderr)

        buffer.clear()

    finished = False
    while True:
        t1 = time.time()

        while len(buffer) < buffer_size:
            payload = json_queue.get()
            if payload is None:
                finished = True
                break

            json_str, meta = payload
            tree_json_dict = json.loads(json_str)
            example = Example.from_json_dict(tree_json_dict, binary_file=meta)

            # if example.ast.size != max(node.node_id for node in example.ast) + 1:
            #     continue

            if tgt_bpe_model:
                eov_id = tgt_bpe_model.eos_id()
                variable_name_subtoken_map = dict()
                tgt_pred_seq_len = 0
                for old_name, new_name in example.variable_name_map.items():
                    if old_name == new_name:
                        subtoken_ids = [vocab.target[SAME_VARIABLE_TOKEN], eov_id]
                    else:
                        subtoken_ids = tgt_bpe_model.encode_as_ids(new_name) + [eov_id]
                    variable_name_subtoken_map[old_name] = subtoken_ids
                    tgt_pred_seq_len += len(subtoken_ids)

                setattr(example, 'variable_name_subtoken_map', variable_name_subtoken_map)
                setattr(example, 'target_prediction_seq_length', tgt_pred_seq_len)

            if train:
                if is_valid_training_example(example):
                    buffer.append(example)
            else:
                buffer.append(example)

        # print(f'[ExampleToBatch] {time.time() - t1}s took for loading {buffer_size} examples to buffer', file=sys.stderr)
        _generate_batches()
        # print(f'[ExampleToBatch] {time.time() - t1}s took for batching', file=sys.stderr)

        if finished:
            break

    batched_examples_queue.put(None)

    while batcher_sync_msg.value == 0:
        time.sleep(1)

    print(f'[ExampleToBatch] quit', file=sys.stderr)
    sys.stderr.flush()


def batch_generator(batched_examples_queue, batch_queue, return_examples, config):
    batcher = Batcher(config, return_examples)

    print(f'[BatchGenerator] pid={os.getpid()}', file=sys.stderr)

    while True:
        batched_examples = batched_examples_queue.get()
        if batched_examples is None:
            break
        else:
            batch = batcher.to_batch(batched_examples)
            # print(f'[Batcher] Batch queue size {batch_queue.qsize()}', file=sys.stderr)
            batch_queue.put(batch)

    batch_queue.put(None)
    while batcher_sync_msg.value == 0:
        time.sleep(1)

    print('[Batcher] Quit current batcher', file=sys.stderr)
    sys.stderr.flush()


def worker_manager(worker_result_queue, out_queue, num_workers, worker_manager_lock, buffer_size):
    num_finished_workers = 0
    patience = 0
    prev_queue_size = -1

    while True:
        finished = False
        # t0 = time.time()
        queue_size = worker_result_queue.qsize()
        # print(f'[LocalWorkerManager] queue size={queue_size}, patience={patience}', file=sys.stderr)
        if (queue_size > buffer_size or patience >= 10) and out_queue.qsize() < buffer_size:
            worker_manager_lock.value = 1
            patience = 0
            print(f'[LocalWorkerManager] start loading {queue_size} batches...', file=sys.stderr)

            i = 0
            while not worker_result_queue.empty() and i < buffer_size:
                batch = worker_result_queue.get()

                # print(f'[LocalWorkerManager] {time.time() - t0} took to load a batch, size={worker_result_queue.qsize()}', file=sys.stderr)
                if batch is not None:
                    out_queue.put(batch)
                else:
                    num_finished_workers += 1
                    if num_finished_workers == num_workers:
                        finished = True
                        break
                i += 1

            print(f'[LocalWorkerManager] loaded {i} batches...', file=sys.stderr)
            worker_manager_lock.value = 0
        else:
            if queue_size == prev_queue_size:
                patience += 1

        prev_queue_size = queue_size
        time.sleep(0.2)
        if finished: break

    out_queue.put(None)


class Dataset(object):
    def __init__(self, file_paths):
        if isinstance(file_paths, list):
            self.file_paths = file_paths
        else:
            assert isinstance(file_paths, str)
            self.file_paths = glob.glob(file_paths)

        print(f'reading data files {self.file_paths}', file=sys.stderr)
        example_num = 0
        for _ in get_json_iterator_from_tar_file(self.file_paths):
            example_num += 1
        self.size = example_num

    def __len__(self):
        return self.size

    def __iter__(self):
        return self.get_iterator(progress=True)

    def get_single_process_iterator(self, shuffle=False, progress=False) -> Iterable[Example]:
        json_str_iter = get_json_iterator_from_tar_file(self.file_paths, shuffle, progress)
        for json_str, meta in json_str_iter:
            tree_json_dict = json.loads(json_str)
            example = Example.from_json_dict(tree_json_dict, binary_file=meta)

            if example.ast.size != max(node.node_id for node in example.ast) + 1:
                continue

            yield example

    def _get_iterator(self, shuffle=False, num_workers=1):
        json_enc_queue = multiprocessing.Queue()
        example_queue = multiprocessing.Queue(maxsize=5000)

        json_loader = multiprocessing.Process(target=json_line_reader, args=(self.file_paths, json_enc_queue, num_workers,
                                                                             shuffle, False))
        json_loader.daemon = True
        example_generators = []
        for i in range(num_workers):
            p = multiprocessing.Process(target=example_generator,
                                        args=(json_enc_queue, example_queue, 1, None))
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
                       num_readers=3,
                       num_batchers=3,
                       progress=True, train=False, single_batcher=False) -> Iterable[Union[Batch, Dict[str, torch.Tensor]]]:
        if progress:
            it_func = lambda x: tqdm(x, file=sys.stdout)
        else:
            it_func = lambda x: x

        if single_batcher:
            return it_func(self._single_process_batch_iter(batch_size, config, num_readers, train))
        else:
            return it_func(self._batch_iterator(batch_size, config, num_readers, num_batchers, train, return_examples))

    def _batch_iterator(self, batch_size: int, config: Dict, num_readers, num_batchers, train=False, return_examples=True) -> Iterable[Batch]:
        global batcher_sync_msg
        batcher_sync_msg = multiprocessing.Value('i', 0)
        json_enc_queue = multiprocessing.Queue(maxsize=10000)
        worker_manager_lock = multiprocessing.Value('i', 0)

        json_loader = multiprocessing.Process(target=json_line_reader,
                                              args=(self.file_paths, json_enc_queue, num_readers,
                                                    train, False))
        json_loader.daemon = True
        example_generators = []
        worker_result_queue = torch_mp.Queue(maxsize=200)
        for i in range(num_readers):
            p = multiprocessing.Process(target=example_to_batch,
                                        args=(json_enc_queue, worker_result_queue, batch_size, train, config, worker_manager_lock))
            p.daemon = True
            example_generators.append(p)

        json_loader.start()
        for p in example_generators: p.start()

        batch_queue = queue.Queue()
        worker_manager_thread = threading.Thread(target=worker_manager, args=(worker_result_queue, batch_queue, num_readers, worker_manager_lock, 100))
        worker_manager_thread.start()

        while True:
            # t1 = time.time()
            main_process_queue_get_lock = 1
            batch = batch_queue.get()
            # print(f'[MainThread] local batch queue size {batch_queue.qsize()}', file=sys.stderr)
            # print(f'{time.time() - t1} took to load a batch', file=sys.stderr)
            if batch is None:
                break
            else:
                yield batch

        worker_result_queue.close()
        # print('start joining...')
        batcher_sync_msg.value = 1
        json_loader.join()
        # print('json_loader quitted')
        for p in example_generators: p.join()
        worker_manager_thread.join()
        # print('example generators quitted')
        # print('batchers quiteed')
        sys.stdout.flush()
        sys.stderr.flush()

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
