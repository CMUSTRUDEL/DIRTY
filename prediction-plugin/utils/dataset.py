import gc
import glob
import multiprocessing
import os
import queue
import resource
import sys
import tarfile
import threading
import time
import torch

import numpy as np
import torch.multiprocessing as torch_mp
import ujson as json

import random
from tqdm import tqdm
from typing import Iterable, List, Dict, Union
from utils import nn_util
from utils.ast import AbstractSyntaxTree
from utils.vocab import SAME_VARIABLE_TOKEN, Vocab


batcher_sync_msg = None
torch.multiprocessing.set_sharing_strategy('file_system')
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


class Example(object):
    def __init__(self,
                 ast: AbstractSyntaxTree,
                 variable_name_map: dict,
                 **kwargs):
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
    def __init__(self, config, train=True):
        self.config = config
        self.train = train

        # model specific config
        self.is_ensemble = config['encoder']['type'] == 'EnsembleModel'
        if not self.is_ensemble:
            self.vocab = Vocab.load(config['data']['vocab_file'])
            self.grammar = self.vocab.grammar

        self.use_seq_encoder = config['encoder']['type'] == 'SequentialEncoder'
        self.use_hybrid_encoder = config['encoder']['type'] == 'HybridEncoder'
        self.init_gnn_with_seq_encoding = \
            config['encoder']['type'] == 'GraphASTEncoder' \
            and config['encoder']['init_with_seq_encoding']

    @property
    def annotate_sequential_input(self):
        return self.use_seq_encoder \
            or self.use_hybrid_encoder \
            or self.init_gnn_with_seq_encoding

    def annotate_example(self, example) -> Example:
        """Annotate examples by populating specific fields, useful for sorting
        examples or batching.
        """
        # for ensemble models, it will be annotated by the batcher for each
        # specific class
        if self.is_ensemble:
            return example

        if self.annotate_sequential_input:
            src_bpe_model = self.vocab.source_tokens.subtoken_model
            snippet = example.code_tokens
            snippet = ' '.join(snippet)
            sub_tokens = \
                ['<s>'] + src_bpe_model.encode_as_pieces(snippet) + ['</s>']
            sub_token_ids = \
                [src_bpe_model.bos_id()] \
                + src_bpe_model.encode_as_ids(snippet) \
                + [src_bpe_model.eos_id()]
            setattr(example, 'sub_tokens', sub_tokens)
            setattr(example, 'sub_token_ids', sub_token_ids)
            setattr(example, 'source_seq_length', len(sub_tokens))

        tgt_bpe_model = self.vocab.target.subtoken_model
        eov_id = tgt_bpe_model.eos_id()
        var_name_subtoken_map = dict()
        tgt_pred_seq_len = 0
        for old_name, new_name in example.variable_name_map.items():
            if old_name == new_name:
                subtoken_ids = [self.vocab.target[SAME_VARIABLE_TOKEN], eov_id]
            else:
                subtoken_ids = tgt_bpe_model.encode_as_ids(new_name) + [eov_id]
            var_name_subtoken_map[old_name] = subtoken_ids
            tgt_pred_seq_len += len(subtoken_ids)

        setattr(example, 'variable_name_subtoken_map', var_name_subtoken_map)
        setattr(example, 'target_prediction_seq_length', tgt_pred_seq_len)

        return example

    def sort_training_examples(self, examples):
        def _key(_example):
            if self.use_seq_encoder:
                return _example.source_seq_length
            elif self.is_ensemble:
                return len(_example.ast.code)
            else:
                return _example.target_prediction_seq_length

        examples.sort(key=_key)

        return examples

    def get_batch_size(self, examples: List[Example]):
        if self.is_ensemble:
            return len(examples)

        if self.annotate_sequential_input:
            return len(examples) * max(e.source_seq_length for e in examples)
        else:
            return len(examples) * \
                max(e.target_prediction_seq_length for e in examples)

    def to_tensor_dict(self,
                       examples: List[Example],
                       return_prediction_target=True):
        # type: (...) -> Dict[str, torch.Tensor]
        from model.sequential_encoder import SequentialEncoder
        from model.graph_encoder import GraphASTEncoder

        if not hasattr(examples[0], 'target_prediction_seq_length'):
            for example in examples:
                self.annotate_example(example)

        if self.config['encoder']['type'] == 'GraphASTEncoder':
            init_with_seq_encoding = \
                self.config['encoder']['init_with_seq_encoding']
            connections = self.config['encoder']['connections']
            packed_graph, tensor_dict = \
                GraphASTEncoder.to_packed_graph(
                    [e.ast for e in examples],
                    connections=connections,
                    init_with_seq_encoding=init_with_seq_encoding
                )

            if init_with_seq_encoding:
                seq_tensor_dict = SequentialEncoder.to_tensor_dict(examples)
                tensor_dict['seq_encoder_input'] = seq_tensor_dict

            _tensors = GraphASTEncoder.to_tensor_dict(packed_graph,
                                                      self.grammar, self.vocab)
            tensor_dict.update(_tensors)
        elif self.config['encoder']['type'] == 'SequentialEncoder':
            tensor_dict = SequentialEncoder.to_tensor_dict(examples)
        elif self.config['encoder']['type'] == 'HybridEncoder':
            connections = \
                self.config['encoder']['graph_encoder']['connections']
            packed_graph, gnn_tensor_dict = \
                GraphASTEncoder.to_packed_graph([e.ast for e in examples],
                                                connections=connections)
            gnn_tensors = GraphASTEncoder.to_tensor_dict(packed_graph,
                                                         self.grammar,
                                                         self.vocab)
            gnn_tensor_dict.update(gnn_tensors)

            seq_tensor_dict = SequentialEncoder.to_tensor_dict(examples)

            tensor_dict = {'graph_encoder_input': gnn_tensor_dict,
                           'seq_encoder_input': seq_tensor_dict}
        else:
            raise ValueError('UnknownEncoderType')

        if self.train or return_prediction_target:
            prediction_target = self.to_batched_prediction_target(examples)
            tensor_dict['prediction_target'] = prediction_target

        if not self.train:
            if hasattr(examples[0], 'test_meta'):
                tensor_dict['test_meta'] = [e.test_meta for e in examples]

        tensor_dict['batch_size'] = len(examples)
        num_elements = nn_util.get_tensor_dict_size(tensor_dict)
        tensor_dict['num_elements'] = num_elements

        return tensor_dict

    def to_batch(self,
                 examples: List[Example],
                 return_examples=False,
                 return_prediction_target=True) -> Batch:
        if self.is_ensemble:
            # do not perform tensorization for the parent ensemble model
            tensor_dict = None
        else:
            with torch.no_grad():
                tensor_dict = \
                    self.to_tensor_dict(examples, return_prediction_target)

        if not return_examples:
            batch = Batch(None, tensor_dict)
            del examples[:]
        else:
            batch = Batch(examples, tensor_dict)

        return batch

    def to_batched_prediction_target(self, examples: List[Example]):
        batch_size = len(examples)
        unchanged_var_weight = \
            self.config['train']['unchanged_variable_weight']
        use_bpe_for_var_name = self.vocab.target.subtoken_model is not None

        var_name_subtoken_maps = []
        if use_bpe_for_var_name:
            var_name_subtoken_maps = \
                [e.variable_name_subtoken_map for e in examples]
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
                var_name_subtoken_maps.append(var_name_subtoken_map)

        max_pred_timestep = max(
            sum(len(val) for val in x.values())
            for x in var_name_subtoken_maps
        )
        tgt_var_enc_indices = \
            torch.zeros(batch_size, max_pred_timestep, dtype=torch.long)
        tgt_var_enc_indices_mask = \
            torch.zeros(batch_size, max_pred_timestep)
        var_tgt_name_id = \
            torch.zeros(batch_size, max_pred_timestep, dtype=torch.long)
        var_tgt_name_weight = torch.zeros(batch_size, max_pred_timestep)
        var_with_new_name_mask = torch.zeros(batch_size, max_pred_timestep)
        auxiliary_var_mask = torch.zeros(batch_size, max_pred_timestep)

        var_master_node_ptr = 0
        for e_id, example in enumerate(examples):
            ast = example.ast
            var_name_map = example.variable_name_map
            var_ptr = 0
            for var_id, var_name in enumerate(ast.variables):
                new_var_name_subtoken_ids = \
                    var_name_subtoken_maps[e_id][var_name]
                var_end_ptr = \
                    var_ptr + len(new_var_name_subtoken_ids)
                var_tgt_name_id[e_id, var_ptr: var_end_ptr] = \
                    torch.tensor(new_var_name_subtoken_ids, dtype=torch.long)
                if var_name == var_name_map[var_name]:
                    auxiliary_var_mask[e_id, var_ptr: var_end_ptr] = 1.
                    var_tgt_name_weight[e_id, var_ptr: var_end_ptr] = \
                        unchanged_var_weight
                else:
                    var_with_new_name_mask[e_id, var_ptr: var_end_ptr] = 1.
                    var_tgt_name_weight[e_id, var_ptr: var_end_ptr] = 1.

                tgt_var_enc_indices[e_id, var_ptr: var_end_ptr] = var_id

                var_master_node_ptr += 1
                var_ptr = var_end_ptr

            tgt_var_enc_indices_mask[e_id, :var_ptr] = 1.

        return dict(var_tgt_name_id=var_tgt_name_id,
                    var_tgt_name_weight=var_tgt_name_weight,
                    var_with_new_name_mask=var_with_new_name_mask,
                    auxiliary_var_mask=auxiliary_var_mask,
                    target_var_encoding_indices=tgt_var_enc_indices,
                    target_var_encoding_indices_mask=tgt_var_enc_indices_mask)


def get_json_iterator_from_tar_file(file_paths,
                                    shuffle=False,
                                    progress=False,
                                    group_by=None,
                                    buffer=True) -> Iterable:
    assert group_by in (None, 'binary_file')

    # if shuffle:
    #     assert buffer is False

    if isinstance(file_paths, str):
        file_paths = [file_paths]

    if shuffle:
        np.random.shuffle(file_paths)
    for file_path in file_paths:
        payloads = []
        t1 = time.time()
        with tarfile.open(file_path, 'r') as f:
            files = [x.name for x in f.getmembers()
                     if x.name.endswith('.jsonl')]
            if progress:
                file_iter = tqdm(files, file=sys.stdout)
            else:
                file_iter = files

            for filename in file_iter:
                jsonl_file = f.extractfile(filename)
                if jsonl_file is not None:
                    if group_by is None:
                        for line, tree_encoding_line in enumerate(jsonl_file):
                            payload = tree_encoding_line, \
                                dict(file_name=filename, line_num=line)
                            if buffer:
                                payloads.append(payload)
                            else:
                                yield payload

                    elif group_by == 'binary_file':
                        lines = [(l.decode().strip(),
                                  dict(file_name=filename, line_num=line_no))
                                 for line_no, l in enumerate(jsonl_file)]
                        yield lines

        if shuffle:
            np.random.shuffle(payloads)

        print(f'load shard {file_path} took {time.time() - t1:.4f}s',
              file=sys.stderr)

        for payload in payloads:
            yield payload


def json_line_reader(file_path,
                     queue,
                     worker_num,
                     shuffle,
                     progress,
                     group_by=None,
                     buffer=True):
    json_iterator = get_json_iterator_from_tar_file(file_path,
                                                    shuffle,
                                                    progress,
                                                    group_by=group_by,
                                                    buffer=buffer)
    for json_str in json_iterator:
        queue.put(json_str)

    for i in range(worker_num):
        queue.put(None)


def is_valid_training_example(example):
    if hasattr(example, 'target_prediction_seq_length'):
        if example.target_prediction_seq_length >= 200:
            return False

    return example.ast.size < 300 \
        and len(example.variable_name_map) > 0 \
        and any(k != v for k, v in example.variable_name_map.items())


def example_generator(json_queue, example_queue, consumer_num=1):
    while True:
        payload = json_queue.get()
        if payload is None:
            break
        json_str, meta = payload

        tree_json_dict = json.loads(json_str)
        if 'code_tokens' in tree_json_dict:
            code_tokens = tree_json_dict['code_tokens']
            example = Example.from_json_dict(tree_json_dict,
                                             binary_file=meta,
                                             code_tokens=code_tokens)
        else:
            example = Example.from_json_dict(tree_json_dict, binary_file=meta)

        example_queue.put(example)

    for i in range(consumer_num):
        example_queue.put(None)

    # print('[Example Generator] example generator process quit!')


def example_to_batch(json_queue,
                     batched_examples_queue,
                     batch_size,
                     train,
                     config,
                     worker_manager_lock,
                     return_examples=False,
                     return_prediction_target=True):
    batcher = Batcher(config, train)

    buffer_size = config['train']['buffer_size']
    buffer = []
    print(f'[ExampleToBatch] pid={os.getpid()}', file=sys.stderr)

    def _generate_batches():
        # buffer.sort(key=batcher.train_example_sort_key)
        batcher.sort_training_examples(buffer)

        batches = []
        batch_examples = []

        for example in buffer:
            batch_size_with_example = \
                batcher.get_batch_size(batch_examples + [example])
            if batch_examples and batch_size_with_example > batch_size:
                batches.append(batch_examples)
                batch_examples = []

            batch_examples.append(example)

        if batch_examples:
            batches.append(batch_examples)

        if train:
            random.shuffle(batches)

        for batch_examples in batches:
            batch = batcher.to_batch(
                batch_examples,
                return_examples=return_examples,
                return_prediction_target=return_prediction_target
            )
            # while batched_examples_queue.qsize() > 100:
            #     time.sleep(10)
            # print(batch.tensor_dict['num_elements'])
            while worker_manager_lock.value == 1:
                time.sleep(0.2)
            batched_examples_queue.put(batch)

        buffer.clear()
        gc.collect()

    finished = False
    while True:
        while len(buffer) < buffer_size:
            payload = json_queue.get()
            if payload is None:
                finished = True
                break

            json_str, meta = payload
            tree_json_dict = json.loads(json_str)

            if 'code_tokens' in tree_json_dict:
                code_tokens = tree_json_dict['code_tokens']
                example = Example.from_json_dict(tree_json_dict,
                                                 binary_file=meta,
                                                 code_tokens=code_tokens)
            else:
                example = \
                    Example.from_json_dict(tree_json_dict, binary_file=meta)
            batcher.annotate_example(example)

            if train:
                if is_valid_training_example(example):
                    buffer.append(example)
            else:
                buffer.append(example)

        _generate_batches()

        if finished:
            break

    batched_examples_queue.put(None)

    while batcher_sync_msg.value == 0:
        time.sleep(1)

    print(f'[ExampleToBatch] quit', file=sys.stderr)
    sys.stderr.flush()


def worker_manager(worker_result_queue,
                   out_queue,
                   num_workers,
                   worker_manager_lock,
                   buffer_size):
    num_finished_workers = 0
    patience = 0
    prev_queue_size = -1

    while True:
        finished = False
        # t0 = time.time()
        try:
            queue_size = worker_result_queue.qsize()
        except Exception:
            # just trigger data loading, for max os X
            queue_size = 999999
        if (queue_size > buffer_size or patience >= 10) \
           and out_queue.qsize() < buffer_size:
            worker_manager_lock.value = 1
            patience = 0

            i = 0
            while not worker_result_queue.empty() and i < buffer_size:
                batch = worker_result_queue.get()

                if batch is not None:
                    out_queue.put(batch)
                else:
                    num_finished_workers += 1
                    if num_finished_workers == num_workers:
                        finished = True
                        break
                i += 1

            worker_manager_lock.value = 0
        else:
            if queue_size == prev_queue_size:
                patience += 1

        prev_queue_size = queue_size
        time.sleep(0.2)
        if finished:
            break

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

    def get_single_process_iterator(self,
                                    shuffle=False,
                                    progress=False) -> Iterable[Example]:
        json_str_iter = \
            get_json_iterator_from_tar_file(self.file_paths, shuffle, progress)
        for json_str, meta in json_str_iter:
            tree_json_dict = json.loads(json_str)
            example = Example.from_json_dict(tree_json_dict, binary_file=meta)
            if example.ast.size != max(n.node_id for n in example.ast) + 1:
                continue
            yield example

    def _get_iterator(self, shuffle=False, num_workers=1):
        enc_queue = multiprocessing.Queue()
        example_queue = multiprocessing.Queue(maxsize=5000)
        args = (self.file_paths,
                enc_queue,
                num_workers,
                shuffle,
                False, None, False)
        json_loader = \
            multiprocessing.Process(target=json_line_reader, args=args)
        json_loader.daemon = True
        example_generators = []
        for i in range(num_workers):
            p = multiprocessing.Process(target=example_generator,
                                        args=(enc_queue, example_queue, 1))
            p.daemon = True
            example_generators.append(p)

        json_loader.start()
        for p in example_generators:
            p.start()

        num_finished_workers = 0
        while True:
            example = example_queue.get()
            if example is not None:
                yield example
            else:
                num_finished_workers += 1
                if num_finished_workers == num_workers:
                    break

        json_loader.join()
        for p in example_generators:
            p.join()

    def get_iterator(self, shuffle=False, progress=True, num_workers=1):
        iterator = self._get_iterator(shuffle, num_workers)
        if progress:
            return tqdm(iterator, total=len(self), file=sys.stdout)
        return iterator

    def batch_iterator(self, batch_size: int, config: Dict,
                       return_examples=False,
                       return_prediction_target=None,
                       num_readers=3,
                       num_batchers=3,
                       progress=True,
                       train=False,
                       single_batcher=False):
        # type: (...) -> Iterable[Union[Batch, Dict[str, torch.Tensor]]]
        if progress:
            it_func = lambda x: tqdm(x, file=sys.stdout)
        else:
            it_func = lambda x: x

        if single_batcher:
            return it_func(
                self._single_process_batch_iter(batch_size,
                                                config,
                                                num_readers,
                                                train)
            )
        else:
            return it_func(
                self._batch_iterator(batch_size,
                                     config,
                                     num_readers,
                                     num_batchers,
                                     train,
                                     return_examples,
                                     return_prediction_target)
            )

    def _batch_iterator(self,
                        batch_size: int,
                        config: Dict,
                        num_readers,
                        num_batchers,
                        train=False,
                        return_examples=False,
                        return_prediction_target=None) -> Iterable[Batch]:
        global batcher_sync_msg
        batcher_sync_msg = multiprocessing.Value('i', 0)
        json_enc_queue = multiprocessing.Queue(maxsize=10000)
        worker_manager_lock = multiprocessing.Value('i', 0)
        args = (self.file_paths, json_enc_queue, num_readers, train, False)
        json_loader = \
            multiprocessing.Process(target=json_line_reader, args=args)
        json_loader.daemon = True
        example_generators = []
        worker_result_queue = torch_mp.Queue(maxsize=150)

        if return_prediction_target is None:
            return_prediction_target = train

        for i in range(num_readers):
            args = (json_enc_queue,
                    worker_result_queue,
                    batch_size,
                    train,
                    config,
                    worker_manager_lock,
                    return_examples,
                    return_prediction_target)
            p = multiprocessing.Process(target=example_to_batch, args=args)
            p.daemon = True
            example_generators.append(p)

        json_loader.start()
        for p in example_generators:
            p.start()

        batch_queue = queue.Queue(maxsize=100)
        args = (worker_result_queue,
                batch_queue,
                num_readers,
                worker_manager_lock,
                100)
        worker_manager_thread = \
            threading.Thread(target=worker_manager, args=args)
        worker_manager_thread.start()

        while True:
            batch = batch_queue.get()
            if batch is None:
                break
            else:
                yield batch

        worker_result_queue.close()
        batcher_sync_msg.value = 1
        json_loader.join()
        for p in example_generators:
            p.join()
        worker_manager_thread.join()
        sys.stdout.flush()
        sys.stderr.flush()

    def _single_process_batch_iter(self,
                                   batch_size: int,
                                   config: Dict,
                                   num_readers=2,
                                   shuffle=False) -> Iterable[Batch]:
        batcher = Batcher(config)
        example_iter = self.get_iterator(shuffle=shuffle,
                                         progress=False,
                                         num_workers=num_readers)
        # t1 = time.time()
        batch_examples = []
        batch_node_num = 0

        # if example.ast.size < 300 and len(example.variable_name_map) > 0:
        for example in filter(is_valid_training_example, example_iter):
            batch_examples.append(example)
            batch_node_num += example.ast.size

            if batch_node_num >= batch_size:
                batch = batcher.to_batch(batch_examples)
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
