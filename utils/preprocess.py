#!/usr/bin/env python
"""
Usage:
    preprocess.py [options] TAR_FILES TARGET_FOLDER

Options:
    -h --help                  Show this screen.
    --shard-size=<int>         shard size [default: 2500]
"""

import glob
import multiprocessing
import tarfile
from collections import Iterable
from typing import Tuple
import ujson as json

from docopt import docopt
import os, sys
from multiprocessing import Process
import numpy as np

from utils.dataset import Example, json_line_reader
from tqdm import tqdm


train_functions = dict()


def is_valid_example(example):
    return example.ast.size < 300 and \
           len(example.variable_name_map) > 0 and \
           any(k != v for k, v in example.variable_name_map.items())


def example_generator(json_queue, example_queue, consumer_num=1):
    while True:
        payload = json_queue.get()
        if payload is None: break

        examples = []
        for json_str, meta in payload:
            tree_json_dict = json.loads(json_str)
            example = Example.from_json_dict(tree_json_dict, binary_file=meta, json_str=json_str)

            if is_valid_example(example):
                examples.append(example)

        example_queue.put(examples)

    for i in range(consumer_num):
        example_queue.put(None)

    print('example generator quited!')


def main(args):
    tgt_folder = args['TARGET_FOLDER']
    tar_files = glob.glob(args['TAR_FILES'])
    print(tar_files)
    shard_size = int(args['--shard-size'])

    os.system(f'mkdir -p {tgt_folder}')
    os.system(f'mkdir -p {tgt_folder}/files')
    num_workers = 5

    for tar_file in tar_files:
        print(f'read {tar_file}')
        valid_example_count = 0

        json_enc_queue = multiprocessing.Queue()
        example_queue = multiprocessing.Queue(maxsize=2000)

        json_loader = multiprocessing.Process(target=json_line_reader,
                                              args=(os.path.expanduser(tar_file), json_enc_queue, num_workers, False, False, 'binary_file'))
        json_loader.daemon = True
        json_loader.start()

        example_generators = []
        for i in range(num_workers):
            p = multiprocessing.Process(target=example_generator, args=(json_enc_queue, example_queue, 1))
            p.daemon = True
            p.start()
            example_generators.append(p)

        n_finished = 0
        while True:
            payload = example_queue.get()
            if payload is None:
                print('received None!')
                n_finished += 1
                if n_finished == num_workers: break
                continue

            examples = payload
            json_strs = []
            for example in examples:
                json_strs.append(example.json_str)

            if json_strs:
                json_file_name = examples[0].binary_file['file_name'].split('/')[-1]
                with open(os.path.join(tgt_folder, 'files/', json_file_name), 'w') as f:
                    for line in json_strs:
                        f.write(line + '\n')

                valid_example_count += len(json_strs)

        print('valid examples: ', valid_example_count)

        json_enc_queue.close()
        example_queue.close()

        json_loader.join()
        for p in example_generators: p.join()

    cur_dir = os.getcwd()
    all_files = glob.glob(os.path.join(tgt_folder, 'files/*.jsonl'))
    all_files = list(all_files)
    np.random.shuffle(all_files)
    print('Total valid binary file num: ', len(all_files))

    shards = [all_files[i:i + shard_size] for i in range(0, len(all_files), shard_size)]
    for shard_id, shard_files in enumerate(shards):
        print(f'Preparing shard {shard_id}, {len(shard_files)} files: ', len(all_files))
        with open(os.path.join(tgt_folder, 'file_list.txt'), 'w') as f:
            for file_name in shard_files:
                f.write(file_name.split('/')[-1] + '\n')

        os.chdir(os.path.join(tgt_folder, 'files'))
        print('creating tar file...')
        os.system(f'tar cf ../shard-{shard_id}.tar -T ../file_list.txt')
        os.chdir(cur_dir)


if __name__ == '__main__':
    args = docopt(__doc__)
    main(args)
