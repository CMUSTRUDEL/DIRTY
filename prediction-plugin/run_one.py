#!/usr/bin/env python
# Runs the decompiler to dump ASTs for a binary
# Requires Python 3

from model.model import RenamingModel
from typing import Dict, List, Any
from utils.dataset import Dataset
from utils.evaluation import Evaluator
from utils.preprocess import generate_example
import argparse
import datetime
import errno
import json
import numpy as np
import os
import pickle
import pprint
import random
import subprocess
import sys
import tempfile
import torch

dire_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'decompiler')
DUMP_TREES = os.path.join(dire_dir, 'decompiler_scripts', 'dump_trees.py')

parser = argparse.ArgumentParser(description="Run the decompiler and dump AST.")
parser.add_argument('--model',
                    help="location of the model"
)
parser.add_argument('--ida',
                    metavar='IDA',
                    help="location of the idat64 binary",
                    default='/home/jlacomis/bin/ida/idat64',
)
parser.add_argument('binary',
                    metavar='BINARY',
                    help="the binary",
)

args = parser.parse_args()
env = os.environ.copy()
env['IDALOG'] = '/dev/stdout'

def make_directory(dir_path):
    """Make a directory, with clean error messages."""
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if not os.path.isdir(dir_path):
            raise NotADirectoryError(f"'{dir_path}' is not a directory")
        if e.errno != errno.EEXIST:
            raise

def run_decompiler(file_name, env, script, timeout=None):
    """Run a decompiler script.

    Keyword arguments:
    file_name -- the binary to be decompiled
    env -- an os.environ mapping, useful for passing arguments
    script -- the script file to run
    timeout -- timeout in seconds (default no timeout)
    """
    idacall = [args.ida, '-B', f'-S{script}', file_name]
    output = ''
    try:
        output = subprocess.check_output(idacall, env=env, timeout=timeout)
    except subprocess.CalledProcessError as e:
        output = e.output
        subprocess.call(['rm', '-f', f'{file_name}.i64'])
    return output

# Create a temporary directory, since the decompiler makes a lot of additional
# files that we can't clean up from here
with tempfile.TemporaryDirectory() as tempdir:
    env['OUTPUT_DIR'] = tempdir
    tempfile.tempdir = tempdir

    file_count = 1

    binary = args.binary
    binary_name = os.path.basename(binary)
    start = datetime.datetime.now()
    env['PREFIX'] = binary_name
    file_path = binary

    with tempfile.NamedTemporaryFile() as orig:
        subprocess.check_output(['cp', file_path, orig.name])
        # Timeout after 10 minutes
        output = run_decompiler(orig.name, env, DUMP_TREES, timeout=600)
        jsonl_filename = os.path.join(tempdir, binary_name+".jsonl")
        if os.path.isfile(jsonl_filename):
            jsonl = open(jsonl_filename, "r").readlines()
        else:
            print("Something bad happened: ", output)
            assert False

        end = datetime.datetime.now()
        duration = end-start
        print(f"Duration: {duration}\n")

print("Preprocessing...")

examples = map(lambda x: generate_example(x, binary), jsonl)
examples = filter(None, examples)

def seed_stuff():
    # seed the RNG
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed * 13 // 7)
    random.seed(seed * 17 // 7)
    sys.setrecursionlimit(7000)
    default_config = '{"decoder": {"remove_duplicates_in_prediction": true} }'
    global extra_config
    extra_config = json.loads(default_config)

def decode(model: RenamingModel,
           examples,
           config: Dict):

    model.eval()
    all_examples = dict()

    with torch.no_grad():
        for line_num, example in enumerate(examples):
            rename_result = model.predict([example])[0]
            example_pred_accs = []
            top_rename_result = rename_result[0]
            for old_name, gold_new_name \
                in example.variable_name_map.items():
                pred = top_rename_result[old_name]
                pred_new_name = pred['new_name']
                var_metric = Evaluator.get_soft_metrics(pred_new_name,
                                                        gold_new_name)
                example_pred_accs.append(var_metric)
            file_name = binary_name + ".jsonl"
            fun_name = example.ast.compilation_unit
            all_examples[f'{file_name}_{line_num}_{fun_name}'] = \
                (rename_result, Evaluator.average(example_pred_accs))

    return all_examples

seed_stuff()

model = RenamingModel.load(args.model,
                           use_cuda=False,
                           new_config=extra_config)

decode_results = \
    decode(model, examples, model.config)
pp = pprint.PrettyPrinter(stream=sys.stderr)
pp.pprint(decode_results)
