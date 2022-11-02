#!/usr/bin/env python
"""
Usage:
    preprocess.py [options] INPUT_FOLDER TARGET_FOLDER

Options:
    -h --help                  Show this screen.
    --max=<int>                max dataset size [default: 10000]
    --shard-size=<int>         shard size [default: 5000]
    --test-file=<file>         test file
    --no-filtering             do not filter files
    --preprocess               only preprocess
"""

import glob
import gzip
import multiprocessing
import os
import random
import shutil
import sys
import tarfile
from json import dumps
from multiprocessing import Process
from typing import Tuple

import numpy as np
import ujson as json
from docopt import docopt
from tqdm import tqdm

from utils.dataset import Example
from utils.dire_types import TypeInfo, TypeLib, TypeLibCodec
from utils.function import CollectedFunction
from utils.code_processing import canonicalize_code

all_functions = dict()  # indexed by binaries


def example_generator(json_str_list):
    examples = []
    for json_str, meta in json_str_list:
        try:
            json_dict = json.loads(json_str)
        except ValueError:
            continue
        cf = CollectedFunction.from_json(json_dict)
        example = Example.from_cf(
            cf, binary_file=meta, max_stack_length=1024, max_type_size=1024
        )

        if example.is_valid_example:
            canonical_code = canonicalize_code(example.raw_code)
            example.canonical_code = canonical_code
            examples.append(example)

    return examples


def json_line_reader(args):
    max_files, bin_number, bin_path = args
    func_json_list = []
    try:
        with gzip.open(bin_path, "rt") as bin_file:
            for line_no, line in enumerate(bin_file):
                json_str = line.strip()
                fname = os.path.basename(bin_path)[:-3]
                if json_str:
                    func_json_list.append(
                        (json_str, dict(file_name=fname, line_num=line_no))
                    )
    except (gzip.BadGzipFile, EOFError):
        print(f"Bad Gzip file {bin_path}")
    except Exception as e:
        print(f"Bad Gzip file {bin_path}, {e}")

    return func_json_list


def type_dumper(args):
    tgt_folder, fname = args
    typelib = TypeLib()
    with open(fname, "r") as f:
        for line in f:
            e = Example.from_json(json.loads(line))
            for var in e.target.values():
                typelib.add(var.typ)
    typelib.sort()
    with open(
        os.path.join(tgt_folder, "types", fname.split("/")[-1]), "w"
    ) as type_lib_file:
        encoded = TypeLibCodec.encode(typelib)
        type_lib_file.write(encoded)


def main(args):
    np.random.seed(1234)
    random.seed(1992)

    tgt_folder = args["TARGET_FOLDER"]
    input_folder = args["INPUT_FOLDER"]

    bins_dir_path = os.path.join(input_folder, "bins")
    bins = (f.path for f in os.scandir(bins_dir_path) if f.is_file())
    max_files = int(args["--max"])
    shard_size = int(args["--shard-size"])

    if os.path.exists(tgt_folder):
        op = input(f"{tgt_folder} exists. remove? (y/n) ")
        if op == "y":
            shutil.rmtree(tgt_folder)

    os.system(f"mkdir -p {tgt_folder}")
    os.system(f"mkdir -p {tgt_folder}/files")
    os.system(f"mkdir -p {tgt_folder}/types")
    num_workers = 16

    valid_example_count = 0

    print("loading examples")
    with multiprocessing.Pool(num_workers) as pool:
        json_iter = pool.imap(
            json_line_reader,
            (
                (max_files, bin_number, bin_path)
                for bin_number, bin_path in enumerate(bins)
            ),
            chunksize=64,
        )

        example_iter = pool.imap(example_generator, json_iter, chunksize=64)

        for examples in tqdm(example_iter):
            if not examples:
                continue
            json_file_name = examples[0].binary_file["file_name"].split("/")[-1]
            with open(os.path.join(tgt_folder, "files/", json_file_name), "w") as f:
                for example in examples:
                    f.write(dumps(example.to_json()) + "\n")
                    all_functions.setdefault(json_file_name, dict())[
                        example.name
                    ] = example.canonical_code

            valid_example_count += len(examples)

    print("valid examples: ", valid_example_count)
    if args["--preprocess"]:
        return

    cur_dir = os.getcwd()
    all_files = glob.glob(os.path.join(tgt_folder, "files/*.jsonl"))
    file_prefix = os.path.join(tgt_folder, "files/")
    sorted(all_files)  # sort all files by names
    all_files = list(all_files)
    file_num = len(all_files)
    print("Total valid binary file num: ", file_num)

    test_file = args["--test-file"]
    if test_file:
        print(f"using test file {test_file}")
        with tarfile.open(test_file, "r") as f:
            test_files = [
                os.path.join(file_prefix, x.name.split("/")[-1])
                for x in f.getmembers()
                if x.name.endswith(".jsonl")
            ]
        dev_file_num = 0
    else:
        print(f"randomly sample test file {test_file}")
        test_file_num = int(file_num * 0.1)
        dev_file_num = int(file_num * 0.1)
        test_files = list(
            np.random.choice(all_files, size=test_file_num, replace=False)
        )

    test_files_set = set(test_files)
    train_files = [fname for fname in all_files if fname not in test_files_set]

    if dev_file_num == 0:
        dev_file_num = int(len(train_files) * 0.1)

    np.random.shuffle(train_files)
    dev_files = train_files[-dev_file_num:]
    train_files = train_files[:-dev_file_num]

    # Create types from filtered training set
    with multiprocessing.Pool(num_workers) as pool:
        pool.map(
            type_dumper, ((tgt_folder, fname) for fname in train_files), chunksize=64,
        )
    print("reading typelib")
    typelib = TypeLib()
    for fname in tqdm(train_files):
        fname = os.path.basename(fname)
        fname = fname[: fname.index(".")] + ".jsonl"
        typelib.add_json_file(os.path.join(tgt_folder, "types", fname))
    typelib.prune(5)
    typelib.sort()

    print("dumping typelib")
    with open(os.path.join(tgt_folder, "typelib.json"), "w") as type_lib_file:
        encoded = TypeLibCodec.encode(typelib)
        type_lib_file.write(encoded)

    train_functions = dict()
    for train_file in train_files:
        file_name = train_file.split("/")[-1]
        for func_name, func in all_functions[file_name].items():
            train_functions.setdefault(func_name, set()).add(func)

    print(
        f"number training: {len(train_files)}, number dev: {len(dev_files)}, number test: {len(test_files)}"
    )
    print("dump training files")
    shards = [
        train_files[i : i + shard_size] for i in range(0, len(train_files), shard_size)
    ]
    for shard_id, shard_files in enumerate(shards):
        print(f"Preparing shard {shard_id}, {len(shard_files)} files: ")
        with open(os.path.join(tgt_folder, "file_list.txt"), "w") as f:
            for file_name in shard_files:
                f.write(file_name.split("/")[-1] + "\n")

        os.chdir(os.path.join(tgt_folder, "files"))
        print("creating tar file...")
        os.system(f"tar cf ../train-shard-{shard_id}.tar -T ../file_list.txt")
        os.chdir(cur_dir)

    def _dump_dev_file(tgt_file_name, file_names):
        with open(os.path.join(tgt_folder, "file_list.txt"), "w") as f:
            for file_name in file_names:
                last_file_name = file_name.split("/")[-1]
                f.write(last_file_name + "\n")

                with open(file_name) as fr:
                    all_lines = fr.readlines()

                replace_lines = []
                for line in all_lines:
                    json_dict = json.loads(line.strip())
                    func_name = json_dict["name"]
                    canonical_code = all_functions[last_file_name][func_name]
                    func_name_in_train = False
                    func_body_in_train = False
                    if func_name in train_functions:
                        func_name_in_train = True
                        if canonical_code in train_functions[func_name]:
                            func_body_in_train = True

                    json_dict["test_meta"] = dict(
                        function_name_in_train=func_name_in_train,
                        function_body_in_train=func_body_in_train,
                    )
                    new_json_str = json.dumps(json_dict)
                    replace_lines.append(new_json_str.strip())

                with open(file_name, "w") as fw:
                    for line in replace_lines:
                        fw.write(line + "\n")

        os.chdir(os.path.join(tgt_folder, "files"))
        print("creating tar file...")
        os.system(f"tar cf ../{tgt_file_name} -T ../file_list.txt")
        os.chdir(cur_dir)

    print("dump dev files")
    _dump_dev_file("dev.tar", dev_files)
    print("dump test files")
    _dump_dev_file("test.tar", test_files)


if __name__ == "__main__":
    args = docopt(__doc__)
    main(args)
